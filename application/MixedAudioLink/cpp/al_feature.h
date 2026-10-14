//===----------------------------------------------------------------------===//
//
// al_feature.h - 共享前处理：kaldi 风格 fbank + mel 滤波器组 + npy/CMVN 读取
// vad 支路（fsmn_vad 输入）与 spk 支路（CAM++ 输入）共用。
// 参数（帧长/帧移/维数/预加重）走 JSON 配置，默认 funasr/kaldi 惯例值。
//
// 数值基准：与 kaldi / FunASR 官方前端逐帧对齐（tools/vad_ref.py 为参考实现，
// tools/feat_dump.cpp + tools/compare_feat.py 为对拍工具）。关键约定：
//   1) 输入样本按 kaldi 约定取 **int16 量纲**（不做 /32768 归一），
//      否则 log 域会整体偏移 2*ln(32768)≈20.79，CMVN 后残留 ~-3.2 的常数偏差。
//   2) DC 去除为 **逐帧**（kaldi remove_dc_offset），非整段均值。
//   3) 预加重在 DC 去除之后、加窗之前；frame[0] 按 kaldi 语义自减去。
//   4) mel 滤波器组：低/高截止按 kaldi low_freq=20 / high_freq=-400（相对 Nyquist），
//      权重用连续 mel 刻度三角（非整数 bin 近似）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_FEATURE_H
#define AL_FEATURE_H

#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
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
        /* 值为 'xxx' / 数字 / (元组) */
        size_t q = header.find_first_of("'0123456789-(", p + 1);
        if (q == std::string::npos) return "";
        if (header[q] == '\'') {
            size_t e = header.find('\'', q + 1);
            return header.substr(q + 1, e - q - 1);
        }
        if (header[q] == '(') {   /* 元组整段返回，各维由调用方解析 */
            size_t e = header.find(')', q);
            return (e == std::string::npos) ? header.substr(q) : header.substr(q, e - q + 1);
        }
        size_t e = header.find_first_of(",})", q);
        return header.substr(q, e - q);
    };
    std::string descr = findField("descr");
    if (descr.find('f') == std::string::npos || descr.find("4") == std::string::npos) { fclose(fp); return false; }
    std::string shape = findField("shape");
    rows = 1; cols = 1;
    size_t total = 0;
    {   /* shape 形如 "80" / "(80,)" / "(128, 201)" / "(1, 128, 201)"。
         * 注意：旧实现只取第一个数字，2-D 的 (128, 201) 被读成 rows=1, cols=128，
         *       导致 mel_filters.npy 静默走回退分支 —— 多维必须逐维解析。
         * 元素总数 = 各维之积（>2 维时不再是 rows*cols）；
         * rows/cols 取倒数第二/最后一维（2-D 即首/次维，与调用方约定一致）。 */
        std::string s = shape;
        size_t lp = s.find('('), rp = s.find(')');
        if (lp != std::string::npos)
            s = s.substr(lp + 1, (rp == std::string::npos ? std::string::npos : rp - lp - 1));
        std::vector<int> dims;
        for (size_t p = 0; p < s.size();) {
            size_t nz = s.find_first_of("0123456789", p);
            if (nz == std::string::npos) break;
            size_t e = s.find_first_not_of("0123456789", nz);
            if (e == std::string::npos) e = s.size();
            try { int v = std::stoi(s.substr(nz, e - nz)); dims.push_back(v); }
            catch (...) { fclose(fp); return false; }
            p = e;
        }
        if (dims.empty()) { fclose(fp); return false; }
        rows = (dims.size() >= 2) ? dims[dims.size() - 2] : 1;
        cols = dims[dims.size() - 1];
        total = 1;
        for (int v : dims) total *= (size_t)(v > 0 ? v : 0);
    }
    out.resize(total);
    if (total && fread(out.data(), sizeof(float), total, fp) != total) { fclose(fp); return false; }
    fclose(fp);
    /* 转主机字节序（npy 为小端，x86/ARM 均为小端，无需转换） */
    return true;
}

/*============================ kaldi CMVN（am.mvn）读取 ============================*/

/**
 * 读取 kaldi 文本 CMVN（如 funasr 的 am.mvn）：
 *   <AddShift> 400 400
 *   <LearnRateCoef> 0 [ v0 v1 ... v399 ]
 *   <Rescale>  400 400
 *   <LearnRateCoef> 0 [ r0 r1 ... r399 ]
 * 返回 shift(=AddShift) / scale(=Rescale)。应用公式：x' = (x + shift) * scale。
 */
inline bool readKaldiCmvn(const std::string &path, std::vector<float> &shift, std::vector<float> &scale) {
    std::ifstream ifs(path.c_str());
    if (!ifs) return false;
    std::vector<std::string> lines;
    std::string ln;
    while (std::getline(ifs, ln)) lines.push_back(ln);

    auto parseVecAfter = [&](const char *tag, std::vector<float> &out) -> bool {
        for (size_t i = 0; i + 1 < lines.size(); i++) {
            std::istringstream ss(lines[i]);
            std::string t0;
            if (!(ss >> t0) || t0 != tag) continue;
            const std::string &d = lines[i + 1];   /* 数值在标记的下一行 */
            size_t lb = d.find('[');
            if (lb == std::string::npos) continue;
            size_t rb = d.find(']', lb);
            if (rb == std::string::npos || rb <= lb) continue;
            std::istringstream vs(d.substr(lb + 1, rb - lb - 1));
            out.clear();
            float v;
            while (vs >> v) out.push_back(v);
            if (!out.empty()) return true;
        }
        return false;
    };
    if (!parseVecAfter("<AddShift>", shift)) return false;
    if (!parseVecAfter("<Rescale>", scale)) return false;
    return shift.size() == scale.size();
}

/*============================ mel 滤波器组 ============================*/

/** kaldi mel 刻度：1127 * ln(1 + f/700) */
inline float melKaldi(float f) { return 1127.0f * std::log(1.0f + f / 700.0f); }
inline float melKaldiInv(float m) { return 700.0f * (std::exp(m / 1127.0f) - 1.0f); }

/**
 * kaldi 风格三角滤波器组 [num_bins, nfft/2+1]。
 * low_freq/high_freq 按 kaldi 约定：>0 取绝对值(Hz)，<=0 取相对 Nyquist 的偏移
 * （默认 low_freq=20，high_freq=-400 → 上截止 = 8000-400 = 7600Hz）。
 * 权重用连续 mel 刻度线性插值（与 kaldi MelBanks::ComputeWeights 一致）。
 */
inline void buildMelFilterbank(int num_bins, int sample_rate, int nfft,
                               float low_freq, float high_freq,
                               std::vector<std::vector<float>> &fb) {
    int num_freqs = nfft / 2 + 1;
    fb.assign(num_bins, std::vector<float>(num_freqs, 0.0f));
    if (num_bins <= 0 || num_freqs <= 0) return;
    float nyquist = 0.5f * sample_rate;
    float low = (low_freq > 0.0f) ? low_freq : (nyquist + low_freq);
    float high = (high_freq > 0.0f) ? high_freq : (nyquist + high_freq);
    if (low < 0.0f) low = 0.0f;
    if (high > nyquist) high = nyquist;
    if (!(high > low)) high = nyquist;   /* 防御：退化为全带宽 */

    float fft_bin_width = (float)sample_rate / (float)nfft;
    float mel_low = melKaldi(low), mel_high = melKaldi(high);
    float delta = (mel_high - mel_low) / (float)(num_bins + 1);
    for (int j = 0; j < num_bins; j++) {
        float left = mel_low + j * delta;
        float center = mel_low + (j + 1) * delta;
        float right = mel_low + (j + 2) * delta;
        for (int i = 0; i < num_freqs; i++) {
            float freq = fft_bin_width * i;
            float mel = melKaldi(freq);
            if (mel > left && mel < right) {
                float w = (mel <= center) ? (mel - left) / (center - left)
                                          : (right - mel) / (right - center);
                fb[j][i] = w;
            }
        }
    }
}

/*============================ FbankComputer ============================*/

struct FbankConfig {
    int num_mel_bins = 80;
    int sample_rate = 16000;
    float frame_ms = 25;         /* 帧长 */
    float shift_ms = 10;         /* 帧移 */
    float preemph = 0.97f;
    float low_freq = 20.0f;      /* kaldi low_freq */
    float high_freq = -400.0f;   /* kaldi high_freq（<=0 相对 Nyquist；0 = 到 Nyquist） */
    bool remove_dc = true;       /* 逐帧去直流（kaldi remove_dc_offset） */
    bool use_log = true;
    float dither = 0.0f;
    /* 采样幅度缩放：funasr/kaldi 前端用 int16 量纲（=1.0）；
     * CAM++ 官方前端先把 int16 归一化到 [-1,1] 再算 log-mel（=1/32767），
     * 该归一化会改变 log 能量的绝对值，并影响 log 的 FLT_EPSILON 下限触发点，
     * 故须与官方一致才能逐点对齐（见 tools/spk_ref_dump.cpp / compare_spk.py）。 */
    float sample_scale = 1.0f;
    /* 窗型：0=hamming（kaldi/funasr 默认），1=povey（CAM++/3D-Speaker 官方前端）。
     * hamming: 0.54 - 0.46cos(2πi/(N-1))；povey: (0.5 - 0.5cos(2πi/(N-1)))^0.85 */
    int window = 0;
};

/* 窗型枚举（FbankConfig::window） */
enum { FBANK_WIN_HAMMING = 0, FBANK_WIN_POVEY = 1 };

class FbankComputer {
public:
    explicit FbankComputer(const FbankConfig &cfg) : m_cfg(cfg) {
        m_frame = (int)(cfg.sample_rate * cfg.frame_ms / 1000.0f);
        m_shift = (int)(cfg.sample_rate * cfg.shift_ms / 1000.0f);
        m_nfft = 1;
        while (m_nfft < m_frame) m_nfft <<= 1;   /* 512 @ 400 */
        buildMelFilterbank(cfg.num_mel_bins, cfg.sample_rate, m_nfft,
                           cfg.low_freq, cfg.high_freq, m_fb);
        /* 窗函数（kaldi FeatureWindowFunction）：
         *   hamming: 0.54 - 0.46cos(2pi i/(N-1))   —— funasr VAD 前端
         *   povey  : (0.5 - 0.5cos(2pi i/(N-1)))^0.85 —— CAM++/3D-Speaker 官方前端 */
        m_win.resize(m_frame);
        for (int i = 0; i < m_frame; i++) {
            double a = 2.0 * (double)M_PI / (m_frame - 1);
            if (FBANK_WIN_POVEY == cfg.window)
                m_win[i] = (float)std::pow(0.5 - 0.5 * std::cos(a * i), 0.85);
            else
                m_win[i] = 0.54f - 0.46f * std::cos((float)a * i);
        }
    }

    int frameSize() const { return m_frame; }
    int frameShift() const { return m_shift; }
    int bins() const { return m_cfg.num_mel_bins; }

    /**
     * pcm(s16, **int16 量纲**) → fbank [T, num_mel_bins] 行优先展平。
     * 逐帧流程严格对齐 kaldi ExtractWindow + PowerSpectrum + MelBanks：
     *   去直流 → 预加重(含 frame[0] 自减) → 加窗 → FFT → 功率谱 → mel 加权 → log
     * 帧数按 snip_edges=true：rows = (n - frame)/shift + 1
     */
    void compute(const std::vector<int16_t> &pcm, std::vector<float> &feat, int &rows, int &cols) {
        cols = m_cfg.num_mel_bins;
        int n = (int)pcm.size();
        rows = (n >= m_frame) ? (n - m_frame) / m_shift + 1 : 0;
        feat.clear();
        if (rows <= 0 || cols <= 0) { rows = 0; return; }
        feat.reserve((size_t)rows * cols);

        std::vector<float> frame(m_frame), spec(m_nfft / 2 + 1);
        std::vector<std::complex<float>> buf(m_nfft);
        for (int r = 0; r < rows; r++) {
            int off = r * m_shift;
            /* 1) 取帧（int16 量纲 × sample_scale）+ 去直流（逐帧） */
            double sum = 0.0;
            for (int i = 0; i < m_frame; i++) {
                float v = (float)pcm[off + i] * m_cfg.sample_scale;
                frame[i] = v; sum += v;
            }
            if (m_cfg.remove_dc) {
                float mean = (float)(sum / (double)m_frame);
                for (int i = 0; i < m_frame; i++) frame[i] -= mean;
            }
            /* 2) 预加重（kaldi Preemphasize：frame[0] 自减） */
            if (m_cfg.preemph != 0.0f) {
                for (int i = m_frame - 1; i > 0; i--) frame[i] -= m_cfg.preemph * frame[i - 1];
                frame[0] -= m_cfg.preemph * frame[0];
            }
            /* 3) 加窗 */
            for (int i = 0; i < m_frame; i++) frame[i] *= m_win[i];

            /* 4) FFT（radix-2）→ 功率谱 */
            for (int i = 0; i < m_frame; i++) buf[i] = std::complex<float>(frame[i], 0);
            for (int i = m_frame; i < m_nfft; i++) buf[i] = std::complex<float>(0, 0);
            fft(buf);
            for (int k = 0; k <= m_nfft / 2; k++) spec[k] = std::norm(buf[k]); /* 功率谱 */

            /* 5) mel 加权 + log（floor 取 kaldi 的 FLT_EPSILON） */
            for (int j = 0; j < cols; j++) {
                float e = 0;
                const std::vector<float> &w = m_fb[j];
                for (int k = 0; k < (int)w.size(); k++) e += w[k] * spec[k];
                float v = m_cfg.use_log ? std::log(std::max(e, FLT_EPSILON)) : e;
                feat.push_back(v);
            }
        }
    }

    /** 全局去直流（整段均值先扣）。⚠️ 与 kaldi 语义不同，仅供兼容保留；
     *  当前 kaldi 对齐路径改用 compute() 内的逐帧去直流，调用方不应再调本函数。 */
    static void removeDc(std::vector<int16_t> &pcm) {
        if (pcm.empty()) return;
        double mean = 0;
        for (auto s : pcm) mean += s;
        mean /= pcm.size();
        for (auto &s : pcm) s = (int16_t)(s - mean);
    }

    /** CMVN：feat[rows,cols] 原地 (x-mean)/std（旧 npy 路径，保留兼容） */
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

    /** CMVN（kaldi am.mvn 语义）：feat[rows,cols] 原地 (x + shift) * scale */
    static bool applyCmvnShiftScale(std::vector<float> &feat, int cols,
                                    const std::vector<float> &shift, const std::vector<float> &scale) {
        if ((int)shift.size() != cols || (int)scale.size() != cols) return false;
        size_t total = feat.size();
        for (size_t i = 0; i < total; i++) {
            int c = (int)(i % cols);
            feat[i] = (feat[i] + shift[c]) * scale[c];
        }
        return true;
    }

    /**
     * FunASR LFR（低帧率拼接）：fbank[T,cols] → [T', lfr_m*cols]
     *   T' = ceil(T / lfr_n)；首部补 (lfr_m-1)/2 帧首帧，尾部补末帧。
     * 与 funasr.frontends.apply_lfr 逐位等价（lfr_n=1 时 T'=T）。
     */
    static void applyLfr(const std::vector<float> &in, int rows, int cols,
                         int lfr_m, int lfr_n, std::vector<float> &out, int &out_rows, int &out_cols) {
        if (lfr_n <= 0) lfr_n = 1;
        out_cols = (lfr_m > 0) ? lfr_m * cols : cols;
        out_rows = (rows > 0) ? (int)std::ceil((double)rows / (double)lfr_n) : 0;
        out.clear();
        if (rows <= 0 || cols <= 0 || lfr_m <= 0) { out_rows = 0; out_cols = cols; return; }
        int pad_l = (lfr_m - 1) / 2;
        int Tp = rows + pad_l;               /* padded 帧数（左侧重复首帧） */
        out.resize((size_t)out_rows * out_cols);
        auto frameAt = [&](int t) -> const float * {
            if (t >= Tp) t = Tp - 1;         /* 右补：重复末帧 */
            int src = t - pad_l;
            if (src < 0) src = 0;            /* 左补：重复首帧 */
            return &in[(size_t)src * cols];
        };
        for (int i = 0; i < out_rows; i++) {
            for (int m = 0; m < lfr_m; m++) {
                std::memcpy(&out[(size_t)i * out_cols + (size_t)m * cols],
                            frameAt(i * lfr_n + m), sizeof(float) * cols);
            }
        }
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

/*============================ LfrStreamer（流式 LFR） ============================*/

/**
 * 流式 LFR 拼接器：等价 funasr `WavFrontendOnline.lfr_splice_cache` 语义。
 *
 * 为什么需要它：批处理 apply_lfr 对整段音频一次性拼接，尾部不足 lfr_m 的窗口
 * 用**末帧重复**补齐。流式下若每块各自补末帧，块尾 ~lfr_m/lfr_n 帧会取到重复帧
 * 而非真实相邻帧（200ms 块约 10% 帧特征失真）。本类做法：
 *   - 左补：首次 push 时重复首帧 (lfr_m-1)/2 次（一次性，与批处理一致）；
 *   - 非 final：只输出**窗口完整**的帧，不足的尾部帧保留在缓存，下块用真实
 *     相邻帧续接（不补、不丢）；
 *   - finish()：右补末帧把剩余窗口一次补全，此后清空。
 *
 * 不变式（已被 tools/lfr_stream_test.cpp 对拍验证）：
 *   任意分块方式下，各次 push 的输出 + finish 的输出，**逐位等于**一次性
 *   FbankComputer::applyLfr 的完整输出。
 *
 * 帧号：输入帧序 = 输出帧序的 1:1 映射（lfr_n=1）或 1:lfr_n；输出帧 k 的
 * 时间戳 = k * frame_shift 样本，与块切分无关。
 */
class LfrStreamer {
public:
    void init(int lfr_m, int lfr_n, int cols) {
        m_lfr_m = (lfr_m >= 1) ? lfr_m : 1;
        m_lfr_n = (lfr_n >= 1) ? lfr_n : 1;
        m_cols = (cols > 0) ? cols : 0;
        reset();
    }

    void reset() {
        m_cache.clear();
        m_padded_len = 0;
        m_base = 0;
        m_last.clear();
        m_frames_in = 0;
        m_frames_out = 0;
        m_seeded = false;
    }

    int outCols() const { return m_cols * m_lfr_m; }     /* 输出帧维数 = lfr_m * cols */
    int64_t framesIn() const { return m_frames_in; }     /* 累计送入 fbank 帧数 */
    int64_t framesOut() const { return m_frames_out; }   /* 累计输出 LFR 帧数（=绝对帧号） */
    int rows() const { return (m_cols > 0) ? (int)(m_cache.size() / (size_t)m_cols) : 0; }
    int pending() const { return rows(); }               /* 缓存中未构成完整窗口的帧数 */

    /** 推入 new_rows 帧（[new_rows, cols] 行优先展平），输出所有"窗口完整"的 LFR 帧 */
    void push(const std::vector<float> &in, int new_rows, std::vector<float> &out, int &out_rows) {
        out.clear();
        out_rows = 0;
        if (m_cols <= 0 || new_rows <= 0) return;
        if ((int)in.size() < new_rows * m_cols) return;

        /* 首次左补：重复首帧 (lfr_m-1)/2 次（lfr_m=1 时 half=0，即无左补） */
        if (!m_seeded) {
            int half = (m_lfr_m - 1) / 2;
            m_cache.assign((size_t)half * m_cols, 0.0f);
            for (int r = 0; r < half; r++)
                std::copy(in.begin(), in.begin() + m_cols, m_cache.begin() + (size_t)r * m_cols);
            m_padded_len = half;
            m_base = 0;
            m_seeded = true;
        }

        /* 追加本块帧：丢弃 padded 下标 < m_base 的（已被判定永不再被引用） */
        int64_t start = m_padded_len;
        int64_t from = (m_base > start) ? m_base : start;
        int skip = (int)(from - start);
        if (skip < new_rows)
            m_cache.insert(m_cache.end(), in.begin() + (size_t)skip * m_cols, in.end());
        m_padded_len = start + new_rows;
        m_frames_in += new_rows;
        m_last.assign(in.end() - m_cols, in.end());   /* 末帧（finish 右补用） */

        emitFull(out, out_rows);
    }

    /**
     * 收尾：把剩余窗口右补末帧后全部输出（等价批处理 apply_lfr 的尾部 num_padding）。
     * 调用后缓存清空；重复调用安全（无剩余窗口时输出为空）。
     */
    void finish(std::vector<float> &out, int &out_rows) {
        out.clear();
        out_rows = 0;
        if (m_cols <= 0 || m_frames_in <= 0) return;

        emitFull(out, out_rows);
        int64_t total = (m_frames_in + m_lfr_n - 1) / m_lfr_n;   /* 批处理总窗口数 */
        int64_t remaining = total - m_frames_out;
        if (remaining <= 0) return;

        /* 右补末帧使最后一个窗口完整（补帧 = 输入最后一帧，与批处理 inputs[-1] 一致） */
        int64_t need_rows = (remaining - 1) * m_lfr_n + m_lfr_m;
        if ((int64_t)rows() < need_rows) {
            if (m_last.empty()) return;   /* 无末帧可用：放弃补尾 */
            for (int64_t i = rows(); i < need_rows; i++)
                m_cache.insert(m_cache.end(), m_last.begin(), m_last.end());
        }

        const int ocols = outCols();
        size_t posted = out.size();
        out.resize(posted + (size_t)remaining * ocols);
        for (int64_t k = 0; k < remaining; k++) {
            const float *src = m_cache.data() + (size_t)(k * m_lfr_n) * m_cols;
            std::copy(src, src + (size_t)m_lfr_m * m_cols, out.data() + posted + (size_t)k * ocols);
        }
        m_frames_out += remaining;
        m_base += remaining * m_lfr_n;
        out_rows += (int)remaining;
        m_cache.clear();
    }

private:
    /**
     * 输出所有窗口完整的窗口（步长 lfr_n）。
     * 窗口 g 用 padded[g*lfr_n .. g*lfr_n+lfr_m)，缓存行 0 即 padded[m_base]，
     * 故窗口 g 用缓存行 [(g-frames_out)*lfr_n ...]。推进后回收 padded[m_base) 之前的帧
     * （lfr_m < lfr_n 时被"跨过"的帧一并丢弃，保证 base 始终对齐到窗口栅格）。
     */
    void emitFull(std::vector<float> &out, int &out_rows) {
        out.clear();
        out_rows = 0;
        int64_t n = (int64_t)m_padded_len - m_base;   /* 可用 padded 行数（=缓存行数） */
        if (n < m_lfr_m) return;
        int64_t count = (n - m_lfr_m) / m_lfr_n + 1;
        if (count <= 0) return;

        const int ocols = outCols();
        out.resize((size_t)count * ocols);
        for (int64_t k = 0; k < count; k++) {
            const float *src = m_cache.data() + (size_t)(k * m_lfr_n) * m_cols;
            std::copy(src, src + (size_t)m_lfr_m * m_cols, out.begin() + (size_t)k * ocols);
        }
        m_base += count * m_lfr_n;
        m_frames_out += count;
        out_rows = (int)count;

        /* 回收：只保留 padded[m_base .. m_padded_len) */
        int64_t keep = (int64_t)m_padded_len - m_base;
        if (keep < 0) keep = 0;
        int64_t drop = (int64_t)rows() - keep;
        if (drop > 0)
            m_cache.erase(m_cache.begin(), m_cache.begin() + (size_t)drop * m_cols);
    }

    int m_lfr_m = 1, m_lfr_n = 1, m_cols = 0;
    std::vector<float> m_cache;   /* 行优先 [rows, cols]，对应 padded[m_base .. m_padded_len) */
    int64_t m_padded_len = 0;     /* 已产生的 padded 帧数（含左补） */
    int64_t m_base = 0;           /* cache[0] 对应的 padded 下标（= 下一窗口起点栅格） */
    std::vector<float> m_last;    /* 最后一帧输入（finish 右补用） */
    int64_t m_frames_in = 0;
    int64_t m_frames_out = 0;
    bool m_seeded = false;
};


/*============================ StreamFrontend（流式前端） ============================*/

/**
 * 流式 fbank -> LFR -> CMVN 前端（VAD 支路）：
 *   - 采样级 carry-over（m_fbuf）：帧栅格自样本 0 起连续，块边界不再撕裂帧
 *     （等价 funasr WavFrontendOnline 的 input_cache）；
 *   - LfrStreamer：LFR 窗口跨块用真实相邻帧，尾部延迟到 finish 才补末帧
 *     （等价 lfr_splice_cache）；
 *   - CMVN 按 kaldi am.mvn 语义 x' = (x + shift) * scale。
 *
 * 等价性（tools/stream_frontend_dump.cpp 对拍验证）：任意分块方式下，
 * 各次 push 输出 + finish 输出按序拼接，**逐位等于**整段一次性
 * FbankComputer::compute + FbankComputer::applyLfr + applyCmvnShiftScale。
 *
 * 帧号：输出 LFR 帧 k 对应流内样本 k*frame_shift（时间戳 = k*shift/16k ms）。
 */
class StreamFrontend {
public:
    StreamFrontend() = default;

    /** 初始化：fc 为 fbank 参数（帧长/帧移/mel 维），lfr_m/lfr_n 为 LFR 拼接参数 */
    void init(const FbankConfig &fc, int lfr_m, int lfr_n) {
        m_fbank.reset(new FbankComputer(fc));
        m_lfr.init(lfr_m, lfr_n, fc.num_mel_bins);
        reset();
    }

    /** 设置 CMVN 参数（kaldi am.mvn），长度须 == outCols()（= lfr_m*num_mel_bins） */
    bool setCmvn(const std::vector<float> &shift, const std::vector<float> &scale) {
        if (shift.empty() || shift.size() != scale.size()) return false;
        if ((int)shift.size() != outCols()) return false;
        m_shift = shift;
        m_scale = scale;
        m_has_cmvn = true;
        return true;
    }

    void reset() {
        m_fbuf.clear();
        m_total_in = 0;
        m_lfr.reset();
    }

    bool hasCmvn() const { return m_has_cmvn; }
    int frameShift() const { return m_fbank ? m_fbank->frameShift() : 160; }
    int frameSize() const { return m_fbank ? m_fbank->frameSize() : 400; }
    int bins() const { return m_fbank ? m_fbank->bins() : 0; }
    int outCols() const { return m_lfr.outCols(); }             /* 输出帧维数 */
    int64_t framesOut() const { return m_lfr.framesOut(); }     /* 已输出 LFR 帧数（绝对帧号） */
    int64_t totalIn() const { return m_total_in; }              /* 累计推入样本数 */
    int pendingFrames() const { return m_lfr.pending(); }       /* 未构成完整窗口的缓存帧数 */

    /** 推入一块 PCM：内部完成 carry-over -> fbank -> LFR -> CMVN，输出可用 LFR 帧 */
    void push(const std::vector<int16_t> &pcm, std::vector<float> &feat, int &out_rows) {
        feat.clear();
        out_rows = 0;
        if (!m_fbank) return;
        m_total_in += (int64_t)pcm.size();

        std::vector<int16_t> buf;
        buf.reserve(m_fbuf.size() + pcm.size());
        buf.insert(buf.end(), m_fbuf.begin(), m_fbuf.end());
        buf.insert(buf.end(), pcm.begin(), pcm.end());

        std::vector<float> fbank;
        int frows = 0, fcols = 0;
        m_fbank->compute(buf, fbank, frows, fcols);
        /* 保留未成帧尾部样本（起始处即下一帧栅格边界）：下块续接 */
        int consumed = frows * m_fbank->frameShift();
        if (consumed < 0) consumed = 0;
        if (consumed > (int)buf.size()) consumed = (int)buf.size();
        m_fbuf.assign(buf.begin() + consumed, buf.end());

        m_lfr.push(fbank, frows, feat, out_rows);
        applyCmvn(feat);
    }

    /** 结束：补 LFR 尾帧并输出剩余 LFR 帧（随后前端回到空态，可承载下一路流） */
    void finish(std::vector<float> &feat, int &out_rows) {
        m_lfr.finish(feat, out_rows);
        applyCmvn(feat);
    }

private:
    void applyCmvn(std::vector<float> &feat) {
        if (!m_has_cmvn || feat.empty()) return;
        FbankComputer::applyCmvnShiftScale(feat, outCols(), m_shift, m_scale);
    }

    std::unique_ptr<FbankComputer> m_fbank;
    std::vector<int16_t> m_fbuf;   /* 尚未成帧的尾部样本（起始处即帧栅格边界） */
    int64_t m_total_in = 0;        /* 累计推入样本数（绝对样本下标基准） */
    LfrStreamer m_lfr;             /* 流式 LFR 拼接 */
    bool m_has_cmvn = false;
    std::vector<float> m_shift, m_scale;
};

} /* namespace al */
#endif /* AL_FEATURE_H */
