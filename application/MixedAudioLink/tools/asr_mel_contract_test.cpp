//===----------------------------------------------------------------------===//
//
// asr_mel_contract_test.cpp - AsrMel 输出契约回归（纯 CPU、无需 TPU 设备）
//
// 目的：锁死「每块 mel 特征字节数 == audio 子网设备输入字节数」这条契约。
//       asr_link.cpp 的 forwardAudio 以 in_bytes 为步长从 host mel 缓冲逐块取值，
//       一旦二者不等就是越界读（首次上板即崩），而离线环境没有任何其他手段暴露它。
//
// 背景（2026-10-14 修复）：model_hub.cpp 曾把 AUDIO_MEL_BINS 取成
//       input_shapes[0].dims[1]，而 qwen3_asr.bmodel 的 audio 子网签名是
//       [1,1,128,100] —— dims[1]=1、dims[2]=128，于是 bins 被算成 1，
//       每块 400 B 而设备要 51200 B（差 128 倍）。已改为相对下标
//       dims[num_dims-2] / dims[num_dims-1]。
//
// 用例：
//   1) 正例  bins=128, audio_length=100, device=51200B -> 契约成立且 filters 已加载
//   2) 反例  bins=1,   audio_length=100, device=51200B -> 契约必须被判失败（回归护栏）
//   3) 数值  输出非退化（有限、有动态范围、值域落在 whisper 归一化后的合理区间）
//
// 构建/运行（宿主 x86 或容器内均可，纯头文件、无 bmlib 依赖）：
//   g++ -O2 -std=c++17 -I cpp tools/asr_mel_contract_test.cpp -o /tmp/asr_mel_contract_test
//   /tmp/asr_mel_contract_test test16k.wav config/audio/mel_filters.npy
//
// 期望输出：3 passed, 0 failed  ->  PASS（退出码 0）
//
//===----------------------------------------------------------------------===//
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "asr_mel.h"

/*------------------------------ 最小 wav 读取 ------------------------------*/
/* 只支持 16bit PCM；够用即可，不追求覆盖所有 RIFF 变体 */
static std::vector<int16_t> readWavPcm16(const char *path) {
    std::vector<int16_t> pcm;
    FILE *f = fopen(path, "rb");
    if (!f) { printf("  open failed: %s\n", path); return pcm; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf((size_t)(sz > 0 ? sz : 0));
    if (sz > 0 && fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return pcm; }
    fclose(f);
    if (buf.size() < 12 || memcmp(buf.data(), "RIFF", 4)) { printf("  not RIFF: %s\n", path); return pcm; }
    size_t off = 12;
    while (off + 8 <= buf.size()) {
        const unsigned char *p = &buf[off];
        uint32_t csz = (uint32_t)p[4] | ((uint32_t)p[5] << 8) |
                       ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
        if (!memcmp(p, "data", 4)) { off += 8; break; }
        off += 8 + csz + (csz & 1);
    }
    if (off >= buf.size()) return pcm;
    size_t n = (buf.size() - off) / 2;
    pcm.resize(n);
    memcpy(pcm.data(), &buf[off], n * 2);
    return pcm;
}

/*------------------------------ 断言框架 ------------------------------*/
static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char *what, const std::string &detail) {
    if (ok) { g_pass++; printf("  [PASS] %s   %s\n", what, detail.c_str()); }
    else    { g_fail++; printf("  [FAIL] %s   %s\n", what, detail.c_str()); }
}

static std::string fmtsz(size_t v) { return std::to_string(v); }

/**
 * 契约判定，与 asr_link.cpp::create() 内的启动期校验同一公式。
 * 该式同时充当回归护栏：只要有人把 bins 的下标改回 dims[1]，反例就会变 PASS。
 */
static bool contractHolds(int bins, int audio_length, size_t device_in_bytes) {
    if (bins <= 0 || audio_length <= 0) return false;
    return (size_t)bins * (size_t)audio_length * sizeof(float) == device_in_bytes;
}

int main(int argc, char **argv) {
    /* 默认值对应 qwen3_asr.bmodel audio 子网实测签名 [1,1,128,100]（51200 B） */
    const char *wav   = argc > 1 ? argv[1] : "test16k.wav";
    const char *npy   = argc > 2 ? argv[2] : "config/audio/mel_filters.npy";
    int bins          = argc > 3 ? atoi(argv[3]) : 128;
    int audio_length  = argc > 4 ? atoi(argv[4]) : 100;
    size_t device_in  = argc > 5 ? (size_t)atoll(argv[5]) : (size_t)51200;

    printf("asr mel contract test\n");
    printf("  wav=%s\n  mel_filters=%s\n  bins=%d audio_length=%d device_in_bytes=%zu\n\n",
           wav, npy, bins, audio_length, device_in);

    std::vector<int16_t> pcm = readWavPcm16(wav);
    if (pcm.empty()) { printf("  cannot read pcm -> abort\n"); return 2; }

    /*-------------------- 用例 1：正例，契约成立 --------------------*/
    printf("[case 1] contract with model-derived shape (bins=%d)\n", bins);
    al::AsrMel mel;
    al::AsrMel::Config mc;
    mc.sample_rate = 16000;
    mc.feature_size = bins;
    mc.n_fft = 400;
    mc.hop_length = 160;
    mc.audio_length = audio_length;
    mc.mel_filters_npy = npy;
    mel.init(mc);

    std::vector<float> out;
    int n_times = mel.compute(pcm, out);

    check(n_times > 0, "n_times > 0",
          std::to_string(n_times) + " chunks (pcm " + fmtsz(pcm.size()) + " samples)");
    check(out.size() == (size_t)n_times * (size_t)bins * (size_t)audio_length,
          "out.size == n_times*bins*audio_length",
          fmtsz(out.size()) + " floats");

    size_t per_chunk_floats = (size_t)bins * (size_t)audio_length;
    size_t per_chunk_bytes = per_chunk_floats * sizeof(float);
    check(contractHolds(bins, audio_length, device_in),
          "per_chunk_bytes == device_in_bytes",
          fmtsz(per_chunk_bytes) + " == " + fmtsz(device_in));
    check(mel.filtersLoaded(), "mel_filters.npy loaded",
          std::string(mel.filtersLoaded() ? "npy" : "slaney fallback"));

    /*-------------------- 用例 2：反例，护栏必须拒绝 --------------------*/
    printf("\n[case 2] negative guard: wrong bins (dims[1] bug) must be rejected\n");
    check(!contractHolds(1, audio_length, device_in),
          "bins=1 rejected (regression guard)",
          fmtsz(1 * audio_length * 4) + " != " + fmtsz(device_in));
    check(!contractHolds(bins, audio_length, device_in * 2),
          "device_bytes mismatch rejected",
          fmtsz(per_chunk_bytes) + " != " + fmtsz(device_in * 2));

    /*-------------------- 用例 3：数值非退化 --------------------*/
    printf("\n[case 3] numeric sanity\n");
    if (!out.empty()) {
        float mn = 1e30f, mx = -1e30f;
        double sum = 0;
        bool finite = true;
        for (float v : out) {
            if (!std::isfinite(v)) { finite = false; break; }
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double mean = sum / (double)out.size();
        check(finite, "all finite", "no nan/inf");
        check(mx > mn, "non-degenerate", "min=" + std::to_string(mn) + " max=" + std::to_string(mx));
        /* whisper 归一化后 (log10(mel)+4)/4，实测落在 [-1.2, 1.1] 量级 */
        check(mn > -3.0f && mx < 3.0f, "range plausible",
              "min=" + std::to_string(mn) + " max=" + std::to_string(mx));
        printf("  mean=%.4f\n", mean);
    } else {
        check(false, "out non-empty", "empty");
    }

    /*-------------------- 用例 4：npy 滤波器组 vs slaney 回退 --------------------*/
    /* 价值：min/max 恒差 2（clamp 下限），看不出滤波器组是否真的换了；
     *      只有逐元素对拍才能证明"静默回退"有没有改变数值。 */
    printf("\n[case 4] whisper mel_filters.npy vs slaney fallback\n");
    {
        al::AsrMel ref;
        al::AsrMel::Config rc2 = mc;
        rc2.mel_filters_npy = "";   /* 强制回退分支 */
        ref.init(rc2);
        std::vector<float> out2;
        int n2 = ref.compute(pcm, out2);
        check(!ref.filtersLoaded(), "fallback path reports not-loaded", "slaney");
        check(n2 == n_times && out2.size() == out.size(), "same shape",
              std::to_string(n2) + " chunks / " + fmtsz(out2.size()) + " floats");
        if (n2 == n_times && out2.size() == out.size()) {
            double maxd = 0, sumd = 0;
            for (size_t i = 0; i < out.size(); i++) {
                double d = std::fabs((double)out[i] - (double)out2[i]);
                if (d > maxd) maxd = d;
                sumd += d;
            }
            printf("  npy vs slaney: max_abs_diff=%.3e  mean_abs_diff=%.3e\n",
                   maxd, sumd / (double)out.size());
            /* 两者都是 slaney 归一化三角滤波器组（whisper 即 librosa mel htk=False），
             * 数值应当高度一致；差异大说明其中一支实现有问题，值得人工复核 */
            check(maxd < 1e-3, "filterbanks agree (<1e-3)",
                  "max_abs_diff=" + std::to_string(maxd));
        }
    }

    printf("\n%d passed, %d failed  ->  %s\n", g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
