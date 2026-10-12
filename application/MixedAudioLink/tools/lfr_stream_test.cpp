//===----------------------------------------------------------------------===//
//
// lfr_stream_test.cpp - LfrStreamer（流式 LFR）与批处理 applyLfr 对拍
//
// 目的：证明"任意分块方式下，各次 LfrStreamer::push 的输出 + finish 的输出，
//       逐位等于一次性 FbankComputer::applyLfr 的完整输出"。
//       这是 VAD 流式前端正确性的核心不变式（块边界不得因补帧而失真）。
//
// 构建/运行（宿主 x86 或容器内均可，纯头文件、无 bmlib 依赖）：
//   g++ -O2 -std=c++17 -I cpp tools/lfr_stream_test.cpp -o /tmp/lfr_stream_test && /tmp/lfr_stream_test
//
//===----------------------------------------------------------------------===//
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "al_feature.h"

using namespace al;

static int g_fail = 0;
static int g_total = 0;

/* 单例：随机 fbank[T,cols]，按给定分块序列流式喂入，与批处理对拍 */
static void run_case(int T, int cols, int lfr_m, int lfr_n, const std::vector<int> &chunks) {
    g_total++;

    std::mt19937 rng(1234u + (unsigned)(T * 131 + cols * 17 + lfr_m * 7 + lfr_n * 3));
    std::uniform_real_distribution<float> dist(-12.0f, 12.0f);
    std::vector<float> in((size_t)T * cols);
    for (auto &v : in) v = dist(rng);

    /* --- 基准：批处理 applyLfr（整段一次性拼接） --- */
    std::vector<float> batch;
    int brow = 0, bcol = 0;
    FbankComputer::applyLfr(in, T, cols, lfr_m, lfr_n, batch, brow, bcol);

    /* --- 流式：LfrStreamer，按 chunks 循环切块 --- */
    LfrStreamer ls;
    ls.init(lfr_m, lfr_n, cols);
    std::vector<float> stream;
    int off = 0;
    size_t ci = 0;
    while (off < T) {
        int n = chunks.empty() ? T : chunks[ci++ % chunks.size()];
        if (n > T - off) n = T - off;
        if (n <= 0) break;
        std::vector<float> part(in.begin() + (size_t)off * cols,
                                in.begin() + (size_t)(off + n) * cols);
        std::vector<float> o;
        int orow = 0;
#ifdef LFR_DBG
        fprintf(stderr, "dbg T=%d lfr=%d/%d off=%d n=%d pending=%d fout=%lld\n",
                T, lfr_m, lfr_n, off, n, ls.pending(), (long long)ls.framesOut());
#endif
        ls.push(part, n, o, orow);
        if (orow > 0 && (int)o.size() != orow * ls.outCols()) {
            printf("  [BAD SHAPE] push orow=%d size=%zu outCols=%d\n", orow, o.size(), ls.outCols());
            g_fail++;
            return;
        }
        stream.insert(stream.end(), o.begin(), o.end());
        off += n;
    }
    std::vector<float> ft;
    int frow = 0;
    ls.finish(ft, frow);
    if (frow > 0 && (int)ft.size() != frow * ls.outCols()) {
        printf("  [BAD SHAPE] finish frow=%d size=%zu outCols=%d\n", frow, ft.size(), ls.outCols());
        g_fail++;
        return;
    }
    stream.insert(stream.end(), ft.begin(), ft.end());

    /* --- 对比 --- */
    const int scol = ls.outCols();
    int srow = (scol > 0) ? (int)(stream.size() / (size_t)scol) : 0;
    if (T == 0) {
        /* 批处理对 rows<=0 会把 out_cols 回退为 cols，语义不同；仅要求双方均无输出 */
        if (!stream.empty() || !batch.empty() || ls.framesOut() != 0) {
            printf("  [T0] stream=%zu batch=%zu fout=%lld\n", stream.size(), batch.size(),
                   (long long)ls.framesOut());
            g_fail++;
        }
        return;
    }
    if (stream.size() != batch.size() || srow != brow || scol != bcol) {
        printf("  [SIZE] T=%d cols=%d lfr=%d/%d chunk0=%d : stream %zuB(%d x %d) vs batch %zuB(%d x %d)\n",
               T, cols, lfr_m, lfr_n, chunks.empty() ? -1 : chunks[0],
               stream.size(), srow, scol, batch.size(), brow, bcol);
        g_fail++;
        return;
    }
    float maxd = 0.0f;
    size_t bad = 0;
    for (size_t i = 0; i < stream.size(); i++) {
        float d = std::fabs(stream[i] - batch[i]);
        if (d > maxd) maxd = d;
        if (stream[i] != batch[i]) bad++;   /* 要求逐位相等（同一运算序列，应无舍入差异） */
    }
    if (bad || (int64_t)srow != ls.framesOut()) {
        printf("  [DIFF] T=%d cols=%d lfr=%d/%d chunk0=%d : rows=%d/%d maxdiff=%.3g mism=%zu\n",
               T, cols, lfr_m, lfr_n, chunks.empty() ? -1 : chunks[0], srow, brow, maxd, bad);
        g_fail++;
    }
}

int main() {
    const int COLS = 80;
    const std::vector<std::vector<int>> chunkSets = {
        {20}, {1}, {3}, {7}, {32}, {160}, {1, 2, 3, 5, 8, 13}, {17, 1, 40}};
    const std::vector<std::pair<int, int>> lfrs = {
        {5, 1},   /* fsmn_vad 实际配置 */
        {1, 1},   /* 禁用 LFR */
        {5, 2}, {7, 1}, {3, 3}, {5, 3}, {2, 2}, {7, 3}, {5, 5}, {1, 2}, {2, 3}, {4, 3}};
    const int Ts[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 21, 32, 33,
                      50, 51, 100, 101, 200, 417, 418, 419, 1000};

    printf("LfrStreamer vs FbankComputer::applyLfr 对拍\n");
    printf("cols=%d, lfr 组合=%zu, 帧数=%zu, 分块方式=%zu\n\n",
           COLS, lfrs.size(), sizeof(Ts) / sizeof(Ts[0]), chunkSets.size());

    for (auto &lf : lfrs) {
        printf("=== lfr_m=%d lfr_n=%d ===\n", lf.first, lf.second);
        int before = g_fail;
        for (int T : Ts)
            for (auto &cs : chunkSets) run_case(T, COLS, lf.first, lf.second, cs);
        printf("  %s\n", (g_fail == before) ? "OK" : "!! FAILED");
    }

    printf("\n%d/%d cases failed\n", g_fail, g_total);
    return g_fail ? 1 : 0;
}
