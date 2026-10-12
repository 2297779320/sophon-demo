//===----------------------------------------------------------------------===//
// vad_seg_test.cpp - 驱动 FsmnVadSegmenter，按 fixture 逐块推入，输出段 JSON
//
// 编译（宿主机即可，段状态机不依赖 bmrt）：
//   g++ -O2 -std=c++17 -I cpp tools/vad_seg_test.cpp -o /tmp/vad_seg_test
//   # ASan 版：-fsanitize=address,undefined（容器内需 setarch $(uname -m) -R 关 ASLR）
//
// 用法：vad_seg_test <fixture>
// fixture 为 key value 行文本，见 tools/vad_seg_ref.py 的 gen_fixture()
//===----------------------------------------------------------------------===//
#include "vad_segmenter.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace al;

static uint64_t fnv1a(const int16_t *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n * 2; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <fixture>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string k;
        is >> k;
        std::string rest;
        std::getline(is, rest);
        if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
        kv[k] = rest;
    }
    auto num = [&](const char *k, int d) {
        auto it = kv.find(k);
        return it == kv.end() ? d : atoi(it->second.c_str());
    };

    FsmnVadSegmenter::Config c;
    c.frame_ms = num("frame_ms", 10);
    c.frame_samples = num("frame_samples", 160);
    c.extra_samples = num("extra_samples", 240);
    c.window_size_ms = num("window_ms", 200);
    c.sil_to_speech_ms = num("sil2sp_ms", 150);
    c.speech_to_sil_ms = num("sp2sil_ms", 150);
    c.max_end_silence_ms = num("max_end_sil_ms", 800);
    c.max_start_silence_ms = num("max_start_sil_ms", 3000);
    c.max_single_seg_ms = num("max_single_seg_ms", 20000);
    c.do_extend = num("do_extend", 1) != 0;
    c.lookback_start_ms = num("lookback_ms", 200);
    c.lookahead_end_ms = num("lookahead_ms", 100);
    c.detect_mode = num("detect_mode", 1);

    const int nframes = num("nframes", 0);
    std::string dec = kv.count("decision") ? kv["decision"] : std::string();
    std::vector<int> chunks;
    {
        std::istringstream is(kv.count("chunks") ? kv["chunks"] : std::string());
        int v;
        while (is >> v) chunks.push_back(v);
    }
    if (chunks.empty() && nframes > 0) chunks.push_back(nframes);

    FsmnVadSegmenter seg;
    seg.init(c);

    const int FS = c.frame_samples;
    std::vector<int16_t> pcm((size_t)nframes * (size_t)FS);
    for (size_t i = 0; i < pcm.size(); i++) pcm[i] = (int16_t)((int)(i % 2000) - 1000);

    std::vector<VadSegment> all;
    int f0 = 0;
    for (size_t ci = 0; ci < chunks.size() && f0 < nframes; ci++) {
        int n = chunks[ci];
        if (n <= 0) continue;
        if (f0 + n > nframes) n = nframes - f0;
        if (n <= 0) break;
        std::vector<int16_t> cp(pcm.begin() + (size_t)f0 * FS, pcm.begin() + (size_t)(f0 + n) * FS);
        std::vector<uint8_t> sp((size_t)n, 0);
        for (int i = 0; i < n; i++) {
            size_t fi = (size_t)f0 + (size_t)i;
            sp[(size_t)i] = (fi < dec.size() && dec[fi] == '1') ? 1u : 0u;
        }
        seg.push(cp, (int64_t)f0 * FS, sp, f0, (ci + 1 == chunks.size()));
        for (VadSegment &s : seg.pending()) all.push_back(std::move(s));
        seg.pending().clear();
        f0 += n;
    }

    printf("{\"segments\":[");
    for (size_t i = 0; i < all.size(); i++) {
        VadSegment &s = all[i];
        uint64_t h = fnv1a(s.pcm.data(), s.pcm.size());
        printf("%s{\"beg_ms\":%lld,\"end_ms\":%lld,\"final\":%d,\"n\":%zu,\"hash\":%llu}",
               i ? "," : "", (long long)s.beg_ms, (long long)s.end_ms, s.is_final ? 1 : 0,
               s.pcm.size(), (unsigned long long)h);
    }
    printf("]}\n");
    return 0;
}
