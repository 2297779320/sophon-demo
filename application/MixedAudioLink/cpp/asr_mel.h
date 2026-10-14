//===----------------------------------------------------------------------===//
//
// asr_mel.h - 128 维 mel 特征（WhisperFeatureExtractor 复现，Qwen3-ASR 前端）
// 参数对齐 LLM-TPU/models/Qwen3_ASR/config/preprocessor_config.json：
//   feature_size=128, n_fft=400, hop_length=160, dither=0, padding right
// 流程：hann 窗 STFT(400) → 功率谱 → mel 滤波(128) → log10(clamp 1e-10)
//       → clamp(max-8) → (x+4)/4 → 按 AUDIO_LENGTH 分块（整秒补零截断）
// STFT 说明：n_fft=400 非 2 的幂，FFT 走 Bluestein/chirp-z（内部复用 radix-2，
// 频点与标准 DFT 严格一致），保证 201 个频点与 whisper mel_filters.npy 对齐；
// 中心对齐 pad 量 = n_fft/2（whisper center=True 语义）。
// mel 滤波器组：JSON asr.mel_filters 指定 .npy(128x201) 时优先加载；
//   否则用 librosa slaney 惯例现算（数值可能略有差异，docs/dataflow.md）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_ASR_MEL_H
#define AL_ASR_MEL_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "al_feature.h"   /* 复用 readNpyF32 / FFT 思想 */

namespace al {

class AsrMel {
public:
    struct Config {
        int sample_rate = 16000;
        int feature_size = 128;   /* n_mels */
        int n_fft = 400;
        int hop_length = 160;
        int audio_length = 200;   /* AUDIO_LENGTH：audio 子网每块 mel 帧数（net_info 推导） */
        std::string mel_filters_npy;   /* 可选 */
    };

    void init(const Config &cfg) {
        m_cfg = cfg;
        m_nfft = cfg.n_fft;
        m_win.resize(m_nfft);
        for (int i = 0; i < m_nfft; i++)
            m_win[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / m_nfft); /* hann（torch 默认 periodic） */
        if ((m_nfft > 0) && (m_nfft & (m_nfft - 1)) != 0)
            initBluestein();   /* n_fft 非 2 的幂（400）时准备 chirp-z 卷积核 */
        if (!loadFiltersNpy(cfg.mel_filters_npy)) {
            buildSlaneyFilterbank(cfg.feature_size, cfg.sample_rate, m_nfft);
            m_filters_loaded = false;
        } else {
            m_filters_loaded = true;
        }
    }

    int bins() const { return m_cfg.feature_size; }
    int framesPerSec() const { return m_cfg.sample_rate / m_cfg.hop_length; }   /* 100 */
    int audioLength() const { return m_cfg.audio_length; }
    /** mel_filters.npy 是否真的加载成功；false = 走了 slaney 回退（形状不符时静默发生） */
    bool filtersLoaded() const { return m_filters_loaded; }

    /**
     * pcm → mel [n_chunks, 128, AUDIO_LENGTH]（行优先展平进 out）。
     * 整秒对齐（pipeline.py: y = y[:len//sr*sr]），尾部不足一个 AUDIO_LENGTH 的帧丢弃。
     * @return 块数 n_times（0 表示音频过短）
     */
    int compute(const std::vector<int16_t> &pcm, std::vector<float> &out) {
        int n = (int)pcm.size();
        int whole_sec = n / m_cfg.sample_rate;
        int max_frames = whole_sec * framesPerSec();
        int n_times = max_frames / m_cfg.audio_length;
        out.clear();
        if (n_times <= 0) return 0;
        int use_frames = n_times * m_cfg.audio_length;

        /* 全段 STFT + mel + log */
        int num_freqs = m_nfft / 2 + 1;
        std::vector<float> logmel((size_t)use_frames * m_cfg.feature_size);
        std::vector<std::complex<float>> buf(m_nfft);
        std::vector<float> frame(m_nfft);
        std::vector<float> spec(num_freqs);   /* 功率谱（先算一次，mel 40 个 bin 复用） */
        for (int f = 0; f < use_frames; f++) {
            int off = f * m_cfg.hop_length;
            for (int i = 0; i < m_nfft; i++) {
                int idx = off + i - m_nfft / 2; /* 中心对齐（whisper center=True：两侧 pad n_fft/2） */
                float x = 0;
                if (idx >= 0 && idx < n) x = pcm[idx] / 32768.0f;
                frame[i] = x * m_win[i];
            }
            for (int i = 0; i < m_nfft; i++) buf[i] = std::complex<float>(frame[i], 0);
            fftN(buf);
            for (int k = 0; k < num_freqs; k++) spec[k] = std::norm(buf[k]);
            /* mel：filters[128,201] @ power[201]，先存原始 log，循环外统一 clamp */
            float *dst = &logmel[(size_t)f * m_cfg.feature_size];
            for (int j = 0; j < m_cfg.feature_size; j++) {
                float e = 0;
                const float *fr = m_filters[j].data();
                for (int k = 0; k < num_freqs; k++) e += fr[k] * spec[k];
                dst[j] = std::log10(std::max(e, 1e-10f));
            }
        }
        /* whisper 全段归一化：clamp 到 max-8 下限 → (x+4)/4 */
        float global_max = -1e30f;
        for (auto v : logmel)
            if (v > global_max) global_max = v;
        float lower = global_max - 8.0f;
        for (auto &v : logmel) {
            v = std::max(v, lower);
            v = (v + 4.0f) / 4.0f;
        }

        /* 重排为 [n_times, feature_size, audio_length]（chat.cpp forward_audio 期望布局） */
        out.resize((size_t)n_times * m_cfg.feature_size * m_cfg.audio_length);
        for (int t = 0; t < n_times; t++) {
            for (int j = 0; j < m_cfg.feature_size; j++) {
                for (int k = 0; k < m_cfg.audio_length; k++) {
                    int src_f = t * m_cfg.audio_length + k;    /* 帧序号 */
                    out[((size_t)t * m_cfg.feature_size + j) * m_cfg.audio_length + k] =
                        logmel[(size_t)src_f * m_cfg.feature_size + j];
                }
            }
        }
        return n_times;
    }

private:
    bool loadFiltersNpy(const std::string &path) {
        if (path.empty()) return false;
        std::vector<float> data;
        int rows, cols;
        if (!readNpyF32(path, data, rows, cols)) return false;
        /* 期望布局 [n_mels, n_freqs] 行主序；若为转置 [n_freqs, n_mels] 则按转置读取 */
        bool transposed = false;
        if (rows != m_cfg.feature_size && cols == m_cfg.feature_size) {
            std::swap(rows, cols);
            transposed = true;
        }
        if (rows != m_cfg.feature_size || cols != m_nfft / 2 + 1) return false;
        m_filters.assign(rows, std::vector<float>(cols));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m_filters[r][c] = transposed ? data[(size_t)c * rows + r]  /* 原始 [freq,mel] */
                                             : data[(size_t)r * cols + c]; /* 原始 [mel,freq] */
        return true;
    }

    /** librosa slaney 惯例 mel 滤波器组（whisper npy 缺席时的回退） */
    void buildSlaneyFilterbank(int n_mels, int sr, int nfft) {
        auto hzToMel = [](float f) {
            float m = f < 1000.0f ? 3.0f * f / 200.0f : 15.0f + 27.0f * std::log(f / 1000.0f) / std::log(6.4f);
            return m;
        };
        auto melToHz = [](float m) {
            float f = m < 15.0f ? 200.0f * m / 3.0f : 1000.0f * std::exp(std::log(6.4f) * (m - 15.0f) / 27.0f);
            return f;
        };
        int num_freqs = nfft / 2 + 1;
        float mel_min = hzToMel(0.0f), mel_max = hzToMel(8000.0f);
        std::vector<float> pts(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            pts[i] = melToHz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
        m_filters.assign(n_mels, std::vector<float>(num_freqs, 0.0f));
        for (int j = 0; j < n_mels; j++) {
            float low = pts[j], cen = pts[j + 1], high = pts[j + 2];
            float enorm = 2.0f / (high - low);   /* slaney norm */
            for (int k = 0; k < num_freqs; k++) {
                float hz = (float)k * sr / nfft;
                if (hz > low && hz < cen) m_filters[j][k] = (hz - low) / (cen - low) * enorm;
                else if (hz >= cen && hz < high) m_filters[j][k] = (high - hz) / (high - cen) * enorm;
            }
        }
    }

    void fft(std::vector<std::complex<float>> &a) {
        int n = (int)a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            float ang = -2.0f * (float)M_PI / len;
            std::complex<float> wlen(std::cos(ang), std::sin(ang));
            for (int i = 0; i < n; i += len) {
                std::complex<float> w(1, 0);
                for (int k = 0; k < len / 2; k++) {
                    std::complex<float> u = a[i + k], v = a[i + k + len / 2] * w;
                    a[i + k] = u + v;
                    a[i + k + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    /*================== 任意长度 FFT（Bluestein/chirp-z） ==================*/

    /** n_fft 非 2 的幂（=400）时预计算卷积核：chirp 相位表与 conj-chirp 频谱 */
    void initBluestein() {
        int N = m_nfft;
        int M = 1;
        while (M < 2 * N - 1) M <<= 1;   /* 线性卷积不环绕：N=400 → M=1024 */
        m_bluestein_M = M;
        m_chirp.resize(N);
        for (int i = 0; i < N; i++) {
            long long ii = (long long)i * i % (2LL * N);   /* e^{-iπn²/N} 以 2N 为周期 */
            double ang = -M_PI * (double)ii / (double)N;
            m_chirp[i] = std::complex<float>((float)std::cos(ang), (float)std::sin(ang));
        }
        std::vector<std::complex<float>> B(M, std::complex<float>(0, 0));
        for (int i = 0; i < N; i++) B[i] = std::conj(m_chirp[i]);
        for (int i = 1; i < N; i++) B[M - i] = B[i];   /* 负频率侧镜像（循环卷积） */
        fft(B);
        m_bluestein_b = std::move(B);
    }

    /** 任意长度 DFT：X[k] = w[k]·Σ (x[n]w[n])·conj(w[k-n])，w[n]=e^{-iπn²/N}；
     *  卷积经 M 点 radix-2 FFT 完成，频点与标准 DFT 严格一致（201 列对齐 mel） */
    void fftAny(std::vector<std::complex<float>> &a) {
        int N = (int)a.size();
        std::vector<std::complex<float>> A(m_bluestein_M, std::complex<float>(0, 0));
        for (int i = 0; i < N; i++) A[i] = a[i] * m_chirp[i];
        fft(A);
        for (int i = 0; i < m_bluestein_M; i++) A[i] *= m_bluestein_b[i];
        for (int i = 0; i < m_bluestein_M; i++) A[i] = std::conj(A[i]);   /* 逆 FFT 共轭法 */
        fft(A);
        for (int k = 0; k < N; k++)
            a[k] = std::conj(A[k]) / (float)m_bluestein_M * m_chirp[k];
    }

    /** FFT 入口：2 的幂走 radix-2，否则走 Bluestein */
    void fftN(std::vector<std::complex<float>> &a) {
        int n = (int)a.size();
        if (n > 0 && (n & (n - 1)) == 0) { fft(a); return; }
        fftAny(a);
    }

    Config m_cfg;
    int m_nfft = 400;   /* n_fft（非 2 的幂时走 Bluestein/chirp-z） */
    bool m_filters_loaded = false;   /* mel_filters.npy 加载结果（供调用方显式告警） */
    std::vector<float> m_win;
    std::vector<std::vector<float>> m_filters;
    /* Bluestein 预计算（n_fft 为 2 的幂时未初始化，fftN 不会用到） */
    int m_bluestein_M = 0;
    std::vector<std::complex<float>> m_chirp, m_bluestein_b;
};

} /* namespace al */
#endif /* AL_ASR_MEL_H */
