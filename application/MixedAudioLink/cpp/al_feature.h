//===----------------------------------------------------------------------===//
//
// al_feature.h - 共享前处理：kaldi 风格 fbank80 + mel 滤波器组 + npy/CMVN 读取
// vad 支路（fsmn_vad 输入）与 spk 支路（CAM++ 输入）共用。
// 参数（帧长/帧移/维数/预加重）走 JSON 配置，默认 funasr/kaldi 惯例值。
// 注意：纯 C++ 复现，与训练前端数值可能有微小差异（docs/dataflow.md 有说明）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_FEATURE_H
#define AL_FEATURE_H

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace al {

/*============================ npy 读取（float32 一/二维） ============================*/

inline bool readNpyF32(const std::string &path, std::vector<float> &out, int &rows, int &cols) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    unsigned char magic[6];
    if (fread(magic, 1, 6, fp) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(fp); return false; }
    unsigned char ver[2];
    if (fread(ver, 1, 2, fp) != 2) { fclose(fp); return false; }
    unsigned int hlen = 0;
    if (ver[0] == 1) { unsigned short h; if (fread(&h, 1, 2, fp) != 2) { fclose(fp); return false; } hlen = h; }
    else { unsigned int h; if (fread(&h, 1, 4, fp) != 4) { fclose(fp); return false; } hlen = h; }
    std::string header(hlen, '\0');
    if (fread(&header[0], 1, hlen, fp) != hlen) { fclose(fp); return false; }

    /* 解析 descr / shape（极简字典扫描） */
    auto findField = [&](const char *key) -> std::string {
        std::string k = "'" + std::string(key) + "'";
        size_t p = header.find(k);
        if (p == std::string::npos) return "";
        p = header.find(':', p + k.size());
        if (p == std::string::npos) return "";
        /* 值为 'xxx' 或 数字 */
        size_t q = header.find_first_of("'0123456789-", p + 1);
        if (q == std::string::npos) return "";
        if (header[q] == '\'') {
            size_t e = header.find('\'', q + 1);
            return header.substr(q + 1, e - q - 1);
        }
        size_t e = header.find_first_of(",})", q);
        return header.substr(q, e - q);
    };
    std::string descr = findField("descr");
    if (descr.find('f') == std::string::npos || descr.find("4") == std::string::npos) { fclose(fp); return false; }
    std::string shape = findField("shape");
    rows = 1; cols = 1;
    {   /* shape 形如 "(80,)" 或 "(80, 1)" */
        size_t comma = shape.find(',');
        try {
            if (comma == std::string::npos) cols = std::stoi(shape);
            else {
                rows = std::stoi(shape.substr(0, comma));
                std::string rest = shape.substr(comma + 1);
                size_t nz = rest.find_first_not_of(" \t");
                cols = (nz == std::string::npos) ? 1 : std::stoi(rest.substr(nz));
            }
        } catch (...) { fclose(fp); return false; }
    }
    size_t total = (size_t)rows * cols;
    out.resize(total);
    if (fread(out.data(), sizeof(float), total, fp) != total) { fclose(fp); return false; }
    fclose(fp);
    /* 转主机字节序（npy 为小端，x86/ARM 均为小端，无需转换） */
    return true;
}

/*============================ mel 滤波器组 ============================*/

/** kaldi mel 刻度：1127 * ln(1 + f/700) */
inline float melKaldi(float f) { return 1127.0f * std::log(1.0f + f / 700.0f); }
inline float melKaldiInv(float m) { return 700.0f * (std::exp(m / 1127.0f) - 1.0f); }

/** 三角滤波器组 [num_bins, num_fft_bins]，kaldi 风格（low→center→high 三点） */
inline void buildMelFilterbank(int num_bins, int sample_rate, int nfft,
                               std::vector<std::vector<float>> &fb) {
    int num_freqs = nfft / 2 + 1;
    fb.assign(num_bins, std::vector<float>(num_freqs, 0.0f));
    float mel_low = melKaldi(20.0f);                     /* kaldi low_freq=20 */
    float mel_high = melKaldi(sample_rate / 2.0f - 20.0f);
    std::vector<float> mel_pts(num_bins + 2);
    for (int i = 0; i < num_bins + 2; i++)
        mel_pts[i] = mel_low + (mel_high - mel_low) * i / (float)(num_bins + 1);
    std::vector<int> bin(num_bins + 2);
    for (int i = 0; i < num_bins + 2; i++) {
        float hz = melKaldiInv(mel_pts[i]);
        float b = hz * nfft / (float)sample_rate;
        bin[i] = (int)b;
        if (bin[i] >= num_freqs) bin[i] = num_freqs - 1;
    }
    for (int j = 0; j < num_bins; j++) {
        int b0 = bin[j], b1 = bin[j + 1], b2 = bin[j + 2];
        for (int k = b0; k < b1 && k < num_freqs; k++) /* 上升沿三角 */
            fb[j][k] = (b1 == b0) ? 1.0f : (float)(k - b0) / (float)(b1 - b0);
        for (int k = b1; k <= b2 && k < num_freqs; k++) /* 下降沿三角 */
            fb[j][k] = (b2 == b1) ? 1.0f : (float)(b2 - k) / (float)(b2 - b1);
    }
}

/*============================ FbankComputer ============================*/

struct FbankConfig {
    int num_mel_bins = 80;
    int sample_rate = 16000;
    float frame_ms = 25;     /* 帧长 */
    float shift_ms = 10;     /* 帧移 */
    float preemph = 0.97f;
    bool remove_dc = true;
    bool use_log = true;
    float dither = 0.0f;
};

class FbankComputer {
public:
    explicit FbankComputer(const FbankConfig &cfg) : m_cfg(cfg) {
        m_frame = (int)(cfg.sample_rate * cfg.frame_ms / 1000.0f);
        m_shift = (int)(cfg.sample_rate * cfg.shift_ms / 1000.0f);
        m_nfft = 1;
        while (m_nfft < m_frame) m_nfft <<= 1;   /* 512 @ 400 */
        buildMelFilterbank(cfg.num_mel_bins, cfg.sample_rate, m_nfft, m_fb);
        /* 汉明窗 */
        m_win.resize(m_frame);
        for (int i = 0; i < m_frame; i++)
            m_win[i] = 0.54f - 0.46f * std::cos(2.0f * (float)M_PI * i / (m_frame - 1));
    }

    int frameSize() const { return m_frame; }
    int frameShift() const { return m_shift; }
    int bins() const { return m_cfg.num_mel_bins; }

    /** pcm(s16) → fbank [T, num_mel_bins] 行优先展平 */
    void compute(const std::vector<int16_t> &pcm, std::vector<float> &feat, int &rows, int &cols) {
        cols = m_cfg.num_mel_bins;
        int n = (int)pcm.size();
        rows = (n >= m_frame) ? (n - m_frame) / m_shift + 1 : 0;
        feat.clear();
        if (rows <= 0) return;
        feat.reserve((size_t)rows * cols);

        std::vector<float> frame(m_frame), spec(m_nfft / 2 + 1);
        std::vector<std::complex<float>> buf(m_nfft);
        float prev = 0;
        for (int r = 0; r < rows; r++) {
            int off = r * m_shift;
            for (int i = 0; i < m_frame; i++) {
                float x = pcm[off + i] / 32768.0f;
                /* 预加重：帧内用前一样本，跨帧用上一帧末样本 */
                frame[i] = (x - prev * m_cfg.preemph) * m_win[i];
                prev = x;
            }

            /* FFT（radix-2） */
            for (int i = 0; i < m_frame; i++) buf[i] = std::complex<float>(frame[i], 0);
            for (int i = m_frame; i < m_nfft; i++) buf[i] = std::complex<float>(0, 0);
            fft(buf);
            for (int k = 0; k <= m_nfft / 2; k++) spec[k] = std::norm(buf[k]); /* 功率谱 */

            for (int j = 0; j < m_cfg.num_mel_bins; j++) {
                float e = 0;
                for (int k = 0; k <= m_nfft / 2; k++) e += m_fb[j][k] * spec[k];
                float v = m_cfg.use_log ? std::log(std::max(e, 1e-10f)) : e;
                feat.push_back(v);
            }
        }
    }

    /** 全局去直流：对整段 pcm 均值先扣（在 compute 前调用） */
    static void removeDc(std::vector<int16_t> &pcm) {
        if (pcm.empty()) return;
        double mean = 0;
        for (auto s : pcm) mean += s;
        mean /= pcm.size();
        for (auto &s : pcm) s = (int16_t)(s - mean);
    }

    /** CMVN：feat[rows,cols] 原地 (x-mean)/std */
    static bool applyCmvn(std::vector<float> &feat, int cols,
                          const std::vector<float> &mean, const std::vector<float> &stdv) {
        if ((int)mean.size() != cols || (int)stdv.size() != cols) return false;
        size_t total = feat.size();
        for (size_t i = 0; i < total; i++) {
            int c = (int)(i % cols);
            feat[i] = (feat[i] - mean[c]) / (stdv[c] > 1e-6f ? stdv[c] : 1.0f);
        }
        return true;
    }

    /** CAM++ 风格：按 utterance 均值减 */
    static void subtractFeatureMean(std::vector<float> &feat, int cols) {
        size_t total = feat.size();
        if (total == 0 || cols <= 0) return;
        int rows = (int)(total / cols);
        std::vector<double> mean(cols, 0);
        for (size_t i = 0; i < total; i++) mean[i % cols] += feat[i];
        for (int c = 0; c < cols; c++) mean[c] /= rows;
        for (size_t i = 0; i < total; i++) feat[i] -= (float)mean[i % cols];
    }

private:
    void fft(std::vector<std::complex<float>> &a) {
        int n = (int)a.size();
        for (int i = 1, j = 0; i < n; i++) { /* bit reversal */
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

    FbankConfig m_cfg;
    int m_frame = 400, m_shift = 160, m_nfft = 512;
    std::vector<std::vector<float>> m_fb;
    std::vector<float> m_win;
};

} /* namespace al */
#endif /* AL_FEATURE_H */
