//===----------------------------------------------------------------------===//
//
// spk_ref_dump.cpp - CAM++ 官方前端的参考特征导出（对拍基准）
//
// 直接编译 **sophon-demo/sample/CAM++/cpp/dependencies** 里的 speakerlab 前端
// （feature_fbank / feature_common / feature_functions / feature_basic / wav_reader），
// 参数与 sample/CAM++/cpp/main.cpp 完全一致（num_bins=80, 16k, dither=0）。
//
// 构建（CAMPP 指向 sample/CAM++/cpp/dependencies/include）：
//   CAMPP=<sophon-demo>/sample/CAM++/cpp/dependencies/include
//   g++ -O2 -std=c++17 -I "$CAMPP" tools/spk_ref_dump.cpp \
//       "$CAMPP"/feature/feature_basic.cpp \
//       "$CAMPP"/feature/feature_common.cpp \
//       "$CAMPP"/feature/feature_fbank.cpp \
//       "$CAMPP"/feature/feature_functions.cpp \
//       "$CAMPP"/utils/wav_reader.cpp -o /tmp/spk_ref_dump
//   /tmp/spk_ref_dump <wav16k_mono> <out_prefix>
// 产物：<prefix>.fbank.bin(.dim)、<prefix>.mean.bin(.dim)（格式同 spk_feat_dump）
//
// 说明：官方前端把 int16 归一化到 [-1,1] 再算 log-mel；本仓用 int16 量纲。
//       两者 log-mel 相差常量 log(32767)，去整句均值后相互抵消，故对拍在看
//       <prefix>.mean 时应一致；<prefix>.fbank 允许常量偏移。
//
//===----------------------------------------------------------------------===//
#include "feature/feature_common.h"
#include "feature/feature_fbank.h"
#include "utils/wav_reader.h"

#include <cstdio>
#include <string>
#include <vector>

static void dumpBin(const std::string &prefix, const char *tag,
                    const std::vector<float> &v, int rows, int cols, bool with_dim_only) {
    (void)with_dim_only;
    std::string p = prefix + "." + tag;
    FILE *f = fopen((p + ".bin").c_str(), "wb");
    if (!f) { fprintf(stderr, "[spk_ref] open %s failed\n", p.c_str()); return; }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    f = fopen((p + ".bin.dim").c_str(), "w");
    fprintf(f, "%d %d\n", rows, cols);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: spk_ref_dump <wav16k_mono> <out_prefix>\n");
        return 2;
    }
    speakerlab::WavReader wav(argv[1]);

    /* 与 sample/CAM++/cpp/main.cpp 一致：num_bins=80 / 16k / dither=0 */
    speakerlab::FbankOptions opts;
    opts.mel_opts.num_bins = 80;
    opts.frame_opts.sample_freq = 16000.0;
    opts.frame_opts.dither = 0.0f;

    speakerlab::FbankComputer fc(opts);
    speakerlab::Feature feat = fc.compute_feature(wav);
    int rows = (int)feat.size();
    int cols = rows > 0 ? (int)feat[0].size() : 0;
    printf("[spk_ref] fbank rows=%d cols=%d\n", rows, cols);

    std::vector<float> raw;
    raw.reserve((size_t)rows * cols);
    for (const auto &r : feat) raw.insert(raw.end(), r.begin(), r.end());
    dumpBin(argv[2], "fbank", raw, rows, cols, true);

    speakerlab::subtract_feature_mean(feat);
    std::vector<float> meanv;
    meanv.reserve((size_t)rows * cols);
    for (const auto &r : feat) meanv.insert(meanv.end(), r.begin(), r.end());
    dumpBin(argv[2], "mean", meanv, rows, cols, true);

    printf("[spk_ref] wrote %s.fbank.bin + %s.mean.bin\n", argv[2], argv[2]);
    return 0;
}
