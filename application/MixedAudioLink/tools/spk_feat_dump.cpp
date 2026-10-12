//===----------------------------------------------------------------------===//
//
// spk_feat_dump.cpp - 声纹（CAM++）前端特征导出工具
//
// 用 cpp/al_feature.h 的 FbankComputer，按 **CAM++/3D-Speaker 官方前端** 参数
// （见 cpp/links/feat_link.h create()）导出 log-mel fbank 与"去整句均值"后的特征，
// 供 tools/compare_spk.py 与官方 C++ 实现（sample/CAM++/cpp/dependencies）对拍。
//
// 构建/运行（纯头文件，无 bmlib 依赖）：
//   g++ -O2 -std=c++17 -I cpp tools/spk_feat_dump.cpp -o /tmp/spk_feat_dump
//   /tmp/spk_feat_dump <wav16k_mono> <out_prefix>
// 产物：<prefix>.fbank.bin(.dim)   原始 log-mel [T,80]
//       <prefix>.mean.bin(.dim)    去均值后    [T,80]（spk_link 实际喂给模型的量）
//
//===----------------------------------------------------------------------===//
#include "../cpp/al_feature.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<int16_t> readWav16(const char *path, bool *ok) {
    std::vector<int16_t> pcm;
    *ok = false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return pcm;
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, fp) != 44) { fclose(fp); return pcm; }
    /* 从 offset 22 起按小端读 ch/rate/bits 做基本校验（假定标准 44 字节头） */
    int ch = hdr[22] | (hdr[23] << 8);
    unsigned rate = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | ((unsigned)hdr[27] << 24);
    int bits = hdr[34] | (hdr[35] << 8);
    if (ch != 1 || rate != 16000 || bits != 16) {
        fprintf(stderr, "[spk_dump] expect 16k/mono/16bit, got ch=%d rate=%u bits=%d\n",
                ch, rate, bits);
        fclose(fp);
        return pcm;
    }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp) - 44;
    fseek(fp, 44, SEEK_SET);
    pcm.resize((size_t)(n / 2));
    if (fread(pcm.data(), 2, pcm.size(), fp) != pcm.size()) { fclose(fp); pcm.clear(); return pcm; }
    fclose(fp);
    *ok = true;
    return pcm;
}

static void dumpBin(const std::string &prefix, const char *tag,
                    const std::vector<float> &v, int rows, int cols) {
    std::string p = prefix + "." + tag;
    FILE *f = fopen((p + ".bin").c_str(), "wb");
    if (!f) { fprintf(stderr, "[spk_dump] open %s failed\n", p.c_str()); return; }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    f = fopen((p + ".bin.dim").c_str(), "w");
    fprintf(f, "%d %d\n", rows, cols);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: spk_feat_dump <wav16k_mono> <out_prefix>\n");
        return 2;
    }
    bool ok = false;
    std::vector<int16_t> pcm = readWav16(argv[1], &ok);
    if (!ok) return 1;
    printf("[spk_dump] wav samples = %zu\n", pcm.size());

    /* 与 cpp/links/feat_link.h create() 保持一致（CAM++ 官方前端） */
    al::FbankConfig fc;
    fc.num_mel_bins = 80;
    fc.sample_rate   = 16000;
    fc.frame_ms      = 25;
    fc.shift_ms      = 10;
    fc.preemph       = 0.97f;
    fc.low_freq      = 20.0f;
    fc.high_freq     = 0.0f;              /* → Nyquist */
    fc.remove_dc     = true;
    fc.use_log       = true;
    fc.dither        = 0.0f;
    fc.sample_scale  = 1.0f / 32767.0f;
    fc.window        = al::FBANK_WIN_POVEY;

    al::FbankComputer c(fc);
    std::vector<float> raw;
    int rows = 0, cols = 0;
    c.compute(pcm, raw, rows, cols);
    if (rows <= 0) { fprintf(stderr, "[spk_dump] no frames\n"); return 1; }
    printf("[spk_dump] fbank rows=%d cols=%d\n", rows, cols);
    dumpBin(argv[2], "fbank", raw, rows, cols);

    al::FbankComputer::subtractFeatureMean(raw, cols);
    dumpBin(argv[2], "mean", raw, rows, cols);
    printf("[spk_dump] wrote %s.fbank.bin + %s.mean.bin\n", argv[2], argv[2]);
    return 0;
}
