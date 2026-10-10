//===----------------------------------------------------------------------===//
//
// al_resample.h - 内置多相 FIR 重采样（header-only，仿 al_log.h：namespace al
// + inline，无全局可变状态单例；系数表为函数内 static 只读缓存，按需生成）。
//
// 能力：任意有理数比 in_rate → 16000 Hz（48k→16k = 1/3、44.1k→16k = 160/441、
// 8k→16k = 2/1 等），多声道均值混缩，int16 饱和输出（恒 16k mono）。
//
// 支持范围与边界：
//   - 系数表内存 ≈ L×TAPS×4B（L=up），建议 L ≤ 160（覆盖 44.1k→16k 的
//     160/441，表约 20KB）；更高比率（如与 16k 互质的 44101）表达 512K tap
//     ≈ 2MB，不建议使用；
//   - 输入长度非 in_ch 整数倍时，尾部不足一个声道的样本被静默截断。
//
// 多相分解原理：
//   有理数比 L/M（up/down，互质）重采样 = 升采样 L 倍 → 低通 → 降采样 M 倍。
//   直接实现需构造插入 L-1 个零的中间序列，计算浪费；多相分解把原型低通
//   h[n]（长度 NT = TAPS×L，TAPS = 每相位抽头数）按 h[kL+p] 拆成 L 个相位
//   子滤波器，输出可直接由输入序列计算：
//       y[n] = Σ_{k=0..TAPS-1} h[kL + p] · x[base - k]
//   其中 base = ⌊nM/L⌋，p = (nM) mod L。
//   推导：y[n] = v[nM] = Σ_j h[j]·w[nM - j]（w 为升采样序列），仅当
//   (nM - j) ≡ 0 (mod L) 时 w 项非零，令 j = (base - k)·L + p 代入即得；
//   且 kL + p ≤ (TAPS-1)L + (L-1) = NT - 1 恒成立，k 取满 0..TAPS-1。
//   原型低通截止 fc = 1/max(L,M)（Nyquist=1 归一化，同时覆盖镜像与混叠），
//   增益 ×L 补偿升采样能量损失，Blackman 窗抑制旁瓣（约 -58 dB）。
//
// 跨调用相位状态（T_ResampleState）：vecHist 保留最近 TAPS 个输入样本，
//   iPhase 记录"下一输出"相对 vecHist 末样本的位置（单位 1/L 输入样本）：
//       base_rel = (TAPS-1) + iPhase / L，p = iPhase mod L
//   每帧处理完后 vecHist 滚动为最近 TAPS 个样本、iPhase 更新为
//   (b - last)·L + p（b/last 为下一未处理输出位置与本帧末样本下标），
//   帧边界任意、跨调用无缝。同一 state 须始终配同一 in_rate。
//
//===----------------------------------------------------------------------===//
#ifndef AL_RESAMPLE_H
#define AL_RESAMPLE_H

#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace al {

/** 每相位抽头数（总抽头 = 32×L，44.1k→16k 时 5120 tap，一次生成常驻） */
#define AL_RS_TAPS_PER_PHASE 32

/** 重采样跨调用相位状态（vecHist 末尾为最新样本；iPhase 语义见文件头注释） */
struct T_ResampleState {
    std::vector<int16_t> vecHist;
    int iPhase = 0;
};

inline int AlRsGcd(int a, int b) {
    while (b) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a < 0 ? -a : a;
}

/**
 * 原型低通系数表（按 L/M 缓存，首次生成后只读）：h[kL+p] 布局与多相取法一致。
 * h[n] = L·fc·sinc(fc·(n-c))·blackman(n)，c = (NT-1)/2，fc = 1/max(L,M)。
 * 线程安全：map 插入持锁；节点式存储返回引用跨插入稳定，读侧无锁安全。
 */
inline const std::vector<float> &AlRsCoeffs(int L, int M) {
    static std::mutex s_mtx;
    static std::map<std::pair<int, int>, std::vector<float>> s_cache;
    std::lock_guard<std::mutex> lk(s_mtx);
    auto key = std::make_pair(L, M);
    auto it = s_cache.find(key);
    if (it != s_cache.end()) {
        return it->second;
    }

    const int taps = AL_RS_TAPS_PER_PHASE;
    int nt = taps * L;
    int fmax = (L > M) ? L : M;          /* 截止 = 1/fmax（Nyquist=1 归一化） */
    auto &tab = s_cache[key];
    tab.resize((size_t)nt);
    double c = (nt - 1) / 2.0;
    for (int n = 0; n < nt; n++) {
        double x = ((double)n - c) / fmax;             /* = fc·(n-c) */
        double s = (fabs(x) < 1e-12) ? 1.0 : sin(M_PI * x) / (M_PI * x);
        /* Blackman 窗：0.42 - 0.5cos(2πn/(NT-1)) + 0.08cos(4πn/(NT-1)) */
        double w = 0.42 - 0.5 * cos(2.0 * M_PI * n / (nt - 1)) +
                   0.08 * cos(4.0 * M_PI * n / (nt - 1));
        tab[(size_t)n] = (float)((double)L / fmax * s * w);
    }
    return tab;
}

/** int16 饱和（clamp -32768..32767） */
inline int16_t AlRsClamp16(double v) {
    if ((v >= 32767.0)) {
        return 32767;
    }
    if ((v <= -32768.0)) {
        return -32768;
    }
    return (int16_t)lrint(v);
}

/**
 * 单帧重采样入口：in（交错多声道 int16）→ 16k mono int16。
 * @param in       输入样本（交错排列，长度须为 in_ch 的整数倍）
 * @param in_rate  输入采样率（>0；同一 st 须始终配同一 in_rate）
 * @param in_ch    输入声道数（>1 时帧内均值混缩）
 * @param st       跨调用相位状态（首调用自动初始化）
 * @return 16k/mono 输出（输入过短暂无输出时返回空，状态已推进）
 */
inline std::vector<int16_t> AlResampleFrame(const std::vector<int16_t> &in,
                                            int in_rate, int in_ch,
                                            T_ResampleState &st) {
    std::vector<int16_t> out;
    if ((in.empty()) || (in_rate <= 0) || (in_ch <= 0)) {
        return out;
    }
    if ((16000 == in_rate) && (1 == in_ch)) {
        return in;   /* 目标格式直通 */
    }

    /*---- 多声道均值混缩（帧内平均 + 四舍五入 + 饱和）----*/
    std::vector<int16_t> mono;
    if (in_ch > 1) {
        size_t frames = in.size() / (size_t)in_ch;
        if (0 == frames) {
            return out;
        }
        mono.resize(frames);
        for (size_t f = 0; f < frames; f++) {
            double acc = 0;
            for (int c = 0; c < in_ch; c++) acc += in[f * (size_t)in_ch + (size_t)c];
            mono[f] = AlRsClamp16(acc / in_ch);
        }
    } else {
        mono = in;
    }
    if (16000 == in_rate) {
        return mono;
    }

    /*---- 多相 FIR：y[n] = Σ h[kL+p]·x[base-k]（推导见文件头）----*/
    int g = AlRsGcd(in_rate, 16000);
    int L = 16000 / g, M = in_rate / g;
    const int taps = AL_RS_TAPS_PER_PHASE;
    const std::vector<float> &h = AlRsCoeffs(L, M);

    /* 首调用初始化：hist 填 taps 个静音（起始输出含短暂瞬态，音频流场景无感），
     * phase = L 对应首输出 base=0、p=0（hist 末样本视作流下标 -1） */
    if ((taps != (int)st.vecHist.size())) {
        st.vecHist.assign((size_t)taps, 0);
        st.iPhase = L;
    }

    /* data = 上一帧尾部 hist + 本帧输入 */
    std::vector<int16_t> data;
    data.reserve(st.vecHist.size() + mono.size());
    data.insert(data.end(), st.vecHist.begin(), st.vecHist.end());
    data.insert(data.end(), mono.begin(), mono.end());

    int64_t last = (int64_t)data.size() - 1;
    int64_t b = (int64_t)(taps - 1) + st.iPhase / L;   /* 下一输出 base（data 下标） */
    int p = st.iPhase % L;

    out.reserve(mono.size() * (size_t)L / (size_t)M + 1);
    while ((b <= last)) {
        /* b ≥ taps 恒成立（phase ≥ L → b ≥ 31+1），x[base-31..base] 不越界 */
        const int16_t *xw = &data[(size_t)b];
        const float *hp = &h[(size_t)p];
        double acc = 0;
        for (int k = 0; k < taps; k++) {
            acc += (double)hp[(size_t)k * (size_t)L] * (double)xw[-k];
        }
        out.push_back(AlRsClamp16(acc));
        /* 推进 nM += M：b += (p+M)/L（整除进位），p = (p+M)%L */
        p += M;
        b += p / L;
        p %= L;
    }

    /* 保存状态：hist 滚动为本帧 data 尾部 taps 个样本；
     * phase = (b - last)·L + p，下一帧以 hist 末样本为下标 31 恢复同一位置 */
    st.vecHist.assign(data.end() - taps, data.end());
    st.iPhase = (int)((b - last) * L + p);
    return out;
}

} /* namespace al */
#endif /* AL_RESAMPLE_H */
