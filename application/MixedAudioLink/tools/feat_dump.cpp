//===----------------------------------------------------------------------===//
// feat_dump.cpp - 前端特征导出工具（交叉验证用）
//
// 直接复用 cpp/al_feature.h 的 FbankComputer，把 fbank / LFR+CMVN 以裸 float32
// 落盘，与 tools/vad_ref.py 的 kaldi/FunASR 参考实现对拍。
//
// 用法: feat_dump <wav16k_mono> <am.mvn> <out_prefix> [lfr_m] [lfr_n]
// 产物: <prefix>.fbank.bin / <prefix>.chunk0.fbank.bin
//       <prefix>.lfr_cmvn.bin / <prefix>.chunk0.lfr_cmvn.bin  (+ .dim)
//
//===----------------------------------------------------------------------===//
#include "../cpp/al_feature.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<int16_t> readWav(const char *path, bool *ok) {
    *ok = false;
    FILE *fp = fopen(path, "rb");
    if (!fp) { printf("cannot open %s\n", path); return {}; }
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, fp) != 44) { fclose(fp); return {}; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 44, SEEK_SET);
    long n = (sz - 44) / 2;
    std::vector<int16_t> pcm(n);
    if (fread(pcm.data(), 2, n, fp) != (size_t)n) { fclose(fp); return {}; }
    fclose(fp);
    *ok = true;
    return pcm;
}

static void dumpBin(const std::string &path, const std::vector<float> &v, int rows, int cols) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (fp) { fwrite(v.data(), sizeof(float), v.size(), fp); fclose(fp); }
    FILE *fd = fopen((path + ".dim").c_str(), "w");
    if (fd) { fprintf(fd, "%d %d\n", rows, cols); fclose(fd); }
}

static void dumpFbank(const std::string &prefix, al::FbankComputer &fb,
                      const std::vector<int16_t> &pcm, const char *tag) {
    std::vector<float> feat;
    int rows, cols;
    fb.compute(pcm, feat, rows, cols);
    printf("[dump] %-14s fbank rows=%d cols=%d\n", tag, rows, cols);
    dumpBin(prefix + ".fbank.bin", feat, rows, cols);
}

static void dumpLfrCmvn(const std::string &path, al::FbankComputer &fb,
                        const std::vector<int16_t> &pcm, int lfr_m, int lfr_n,
                        const std::vector<float> &shift, const std::vector<float> &scale,
                        const char *tag) {
    std::vector<float> feat;
    int rows, cols;
    fb.compute(pcm, feat, rows, cols);
    std::vector<float> lfr;
    int lrows, lcols;
    al::FbankComputer::applyLfr(feat, rows, cols, lfr_m, lfr_n, lfr, lrows, lcols);
    bool ok = al::FbankComputer::applyCmvnShiftScale(lfr, lcols, shift, scale);
    printf("[dump] %-14s lfr_cmvn rows=%d cols=%d cmvn=%s\n", tag, lrows, lcols, ok ? "ok" : "FAIL");
    dumpBin(path, lfr, lrows, lcols);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: feat_dump <wav> <am.mvn> <out_prefix> [lfr_m] [lfr_n]\n");
        return 1;
    }
    int lfr_m = (argc > 4) ? atoi(argv[4]) : 5;
    int lfr_n = (argc > 5) ? atoi(argv[5]) : 1;

    bool ok = false;
    std::vector<int16_t> pcm = readWav(argv[1], &ok);
    if (!ok) return 1;
    printf("[dump] wav samples = %zu (%.2fs)\n", pcm.size(), pcm.size() / 16000.0);

    std::vector<float> shift, scale;
    if (!al::readKaldiCmvn(argv[2], shift, scale)) { printf("cmvn read FAIL: %s\n", argv[2]); return 1; }
    printf("[dump] cmvn dim=%zu shift[0:3]=%.4f,%.4f,%.4f scale[0:3]=%.4f,%.4f,%.4f\n",
           shift.size(), shift[0], shift[1], shift[2], scale[0], scale[1], scale[2]);
    dumpBin(std::string(argv[3]) + ".cmvn.bin", shift, 1, (int)shift.size());

    al::FbankConfig fc;
    al::FbankComputer fb(fc);
    std::string prefix(argv[3]);

    /* 整段 */
    {
        std::vector<float> feat;
        int rows, cols;
        fb.compute(pcm, feat, rows, cols);
        printf("[dump] full          fbank rows=%d cols=%d\n", rows, cols);
        dumpBin(prefix + ".fbank.bin", feat, rows, cols);
        std::vector<float> lfr;
        int lr, lc;
        al::FbankComputer::applyLfr(feat, rows, cols, lfr_m, lfr_n, lfr, lr, lc);
        al::FbankComputer::applyCmvnShiftScale(lfr, lc, shift, scale);
        printf("[dump] full          lfr_cmvn rows=%d cols=%d\n", lr, lc);
        dumpBin(prefix + ".lfr_cmvn.bin", lfr, lr, lc);
    }
    /* 第 0 块（3200 样本 = 200ms） */
    {
        std::vector<int16_t> a(pcm.begin(), pcm.begin() + std::min<size_t>(3200, pcm.size()));
        dumpFbank(prefix + ".chunk0", fb, a, "chunk0");
        dumpLfrCmvn(prefix + ".chunk0.lfr_cmvn.bin", fb, a, lfr_m, lfr_n, shift, scale, "chunk0");
    }
    (void)dumpLfrCmvn;
    return 0;
}
