//===----------------------------------------------------------------------===//
//
// stream_frontend_dump.cpp - 流式前端（StreamFrontend）端到端导出/对拍工具
//
// 目的：证明 vad_link 实际使用的**流式**前处理链路
//         StreamFrontend（采样级 carry-over fbank → LfrStreamer → CMVN）
//       在任意分块下，输出按序拼接后逐位等于整段一次性
//         FbankComputer::compute + applyLfr + applyCmvnShiftScale
//       （即 tools/vad_ref.py 的 kaldi/FunASR 基准，见 ref_*_lfr_cmvn.npy）。
//
// 构建/运行（宿主 x86 或容器内，纯头文件、无 bmlib 依赖）：
//   g++ -O2 -std=c++17 -I cpp tools/stream_frontend_dump.cpp -o /tmp/stream_dump
//   /tmp/stream_dump <wav16k> <am.mvn> <out_prefix> [chunk_ms=200] [lfr_m=5] [lfr_n=1]
// 产物：<prefix>.stream.bin / <prefix>.stream.bin.dim
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "al_feature.h"

using namespace al;

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
    std::vector<int16_t> pcm((size_t)(n > 0 ? n : 0));
    if (!pcm.empty() && fread(pcm.data(), 2, pcm.size(), fp) != pcm.size()) { fclose(fp); return {}; }
    fclose(fp);
    *ok = true;
    return pcm;
}

static void writeBin(const std::string &path, const std::vector<float> &v, int rows, int cols) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (fp) { if (!v.empty()) fwrite(v.data(), sizeof(float), v.size(), fp); fclose(fp); }
    FILE *fd = fopen((path + ".dim").c_str(), "w");
    if (fd) { fprintf(fd, "%d %d\n", rows, cols); fclose(fd); }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: %s <wav16k> <am.mvn> <out_prefix> [chunk_ms=200] [lfr_m=5] [lfr_n=1]\n",
               argv[0]);
        return 2;
    }
    const char *wav = argv[1];
    const char *mvn = argv[2];
    std::string prefix = argv[3];
    int chunk_ms = (argc > 4) ? atoi(argv[4]) : 200;
    int lfr_m = (argc > 5) ? atoi(argv[5]) : 5;
    int lfr_n = (argc > 6) ? atoi(argv[6]) : 1;
    if (chunk_ms < 1) chunk_ms = 200;

    bool ok = false;
    std::vector<int16_t> pcm = readWav(wav, &ok);
    if (!ok || pcm.empty()) { printf("bad wav: %s\n", wav); return 1; }
    printf("[stream] wav samples = %zu (%.2fs), chunk=%dms, lfr=%d/%d\n",
           pcm.size(), pcm.size() / 16000.0, chunk_ms, lfr_m, lfr_n);

    /* CMVN：kaldi am.mvn（(x+shift)*scale） */
    std::vector<float> shift, scale;
    if (!readKaldiCmvn(mvn, shift, scale)) { printf("bad cmvn: %s\n", mvn); return 1; }

    FbankConfig fc;   /* 默认：80 mel / 25ms / 10ms / high_freq=-400，与 vad_ref.py 一致 */
    StreamFrontend front;
    front.init(fc, lfr_m, lfr_n);
    if (!front.setCmvn(shift, scale)) {
        printf("[stream] cmvn dim %zu != outCols %d\n", shift.size(), front.outCols());
        return 1;
    }
    printf("[stream] fbank bins=%d outCols=%d cmvn=%d\n",
           front.bins(), front.outCols(), (int)front.hasCmvn());

    /* 按 chunk_ms 分块喂入（与 CPcmLink::getFrame 的 200ms 拉块一致） */
    const int chunk_samples = 16000 * chunk_ms / 1000;
    std::vector<float> all;
    std::vector<int> counts;
    for (size_t off = 0; off < pcm.size(); off += (size_t)chunk_samples) {
        size_t n = std::min<size_t>((size_t)chunk_samples, pcm.size() - off);
        std::vector<int16_t> blk(pcm.begin() + off, pcm.begin() + off + n);
        std::vector<float> feat;
        int rows = 0;
        front.push(blk, feat, rows);
        if (rows > 0) all.insert(all.end(), feat.begin(), feat.end());
        counts.push_back(rows);
    }
    std::vector<float> tail;
    int trows = 0;
    front.finish(tail, trows);
    if (trows > 0) all.insert(all.end(), tail.begin(), tail.end());
    counts.push_back(trows);

    const int cols = front.outCols();
    const int rows = (cols > 0) ? (int)(all.size() / (size_t)cols) : 0;
    printf("[stream] per-chunk out rows =");
    for (size_t i = 0; i < counts.size() && i < 12; i++) printf(" %d", counts[i]);
    if (counts.size() > 12) printf(" ...");
    printf("\n[stream] total rows = %d x %d (framesOut=%lld, samplesIn=%lld)\n",
           rows, cols, (long long)front.framesOut(), (long long)front.totalIn());

    /* 同进程整段基准（compute + applyLfr + applyCmvnShiftScale）：
     * 只有同二进制内比较才可要求逐位相等——跨二进制会引入 ~1e-6 的浮点代码生成差异 */
    FbankComputer fb(fc);
    std::vector<float> fb_full;
    int fbr = 0, fbc = 0;
    fb.compute(pcm, fb_full, fbr, fbc);
    std::vector<float> lfr_full;
    int lbr = 0, lbc = 0;
    FbankComputer::applyLfr(fb_full, fbr, fbc, lfr_m, lfr_n, lfr_full, lbr, lbc);
    FbankComputer::applyCmvnShiftScale(lfr_full, lbc, shift, scale);
    writeBin(prefix + ".batch.bin", lfr_full, lbr, lbc);
    printf("[stream] (same-binary) batch rows = %d x %d\n", lbr, lbc);

    writeBin(prefix + ".stream.bin", all, rows, cols);
    printf("[stream] wrote %s.stream.bin(.dim) + %s.batch.bin(.dim)\n",
           prefix.c_str(), prefix.c_str());
    return 0;
}
