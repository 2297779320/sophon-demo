//===----------------------------------------------------------------------===//
//
// main.cpp - MixedAudioLink demo
// 用法：./audiolink_demo -c config/audio/audio_link.json -d 0 -i test16k.wav
// 模拟实时流：按 100ms 步长将 wav 切块推入 AudioLinkPushPcm，
// 结果经 AL_RESULT_CB 打印（文本 + 声纹向量维度/首元素）。
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include "audio_link.h"
#include "meeting_summary.h"

/*---- 极简 wav(PCM16) 读取 ----*/
static bool readWav16kMono(const char *path, std::vector<int16_t> &pcm, int &sample_rate) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, fp) != 44) { fclose(fp); return false; }
    if (memcmp(hdr + 0, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4) ||
        memcmp(hdr + 12, "fmt ", 4) || memcmp(hdr + 36, "data", 4)) {
        fclose(fp);
        return false;
    }
    sample_rate = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | ((int)hdr[27] << 24);
    int channels = hdr[22] | (hdr[23] << 8);
    int bits = hdr[34] | (hdr[35] << 8);
    if (bits != 16) { fclose(fp); return false; }
    long data_pos = 44;
    /* 跳过 fmt 扩展（若 header 长度 > 16） */
    int fmt_len = hdr[16] | (hdr[17] << 8) | (hdr[18] << 16) | ((int)hdr[19] << 24);
    if (fmt_len > 16) {
        fseek(fp, fmt_len - 16, SEEK_CUR);
        char id[4];
        while (fread(id, 1, 4, fp) == 4) {
            int sz = id[0] | (id[1] << 8) | (id[2] << 16) | ((int)id[3] << 24);
            if (memcmp(id, "data", 4) == 0) { data_pos = ftell(fp); break; }
            fseek(fp, sz, SEEK_CUR);
        }
    }
    fseek(fp, 0, SEEK_END);
    long total = ftell(fp) - data_pos;
    fseek(fp, data_pos, SEEK_SET);
    int frames = (int)(total / 2 / channels);
    pcm.resize(frames * channels);
    if (fread(pcm.data(), 2, pcm.size(), fp) != pcm.size()) { fclose(fp); return false; }
    fclose(fp);
    if (channels > 1) { /* 多声道取第一声道（demo 简化） */
        for (int i = 0; i < frames; i++) pcm[i] = pcm[(size_t)i * channels];
        pcm.resize(frames);
    }
    return true;
}

/*---- 结果回调：打印 "讲话人N：xxx" 归属行 ----*/
static void onResult(const AL_RESULT *r, void *user) {
    (void)user;
    printf("[%d-%d ms] %s\n", r->start_ms, r->end_ms,
           r->line ? r->line : (r->text ? r->text : "(no text)"));
    if (r->spk_emb && r->spk_dim > 0 && r->speaker_id > 0)
        printf("        (spk_id=%d dim=%d emb[0]=%.4f)\n", r->speaker_id, r->spk_dim,
               r->spk_emb[0]);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    const char *cfg = "config/audio/audio_link.json";
    const char *wav = nullptr;
    int devid = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i + 1 < argc) cfg = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) devid = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) wav = argv[++i];
        else { fprintf(stderr, "usage: %s -c cfg.json -d devid -i test16k.wav\n", argv[0]); return 1; }
    }

    printf("version: %s\n", AudioLinkGetVersion());

    AL_HANDLE mgr = AudioLinkInit(cfg, devid);
    if (!mgr) { fprintf(stderr, "AudioLinkInit failed\n"); return 1; }

    /* 查看模型信息与子网列表（静态 shape 推导结果） */
    char buf[4096] = {0};
    AL_PARAM p{(void *)buf, sizeof(buf)};
    AudioLinkCtrl(mgr, nullptr, "list_nets", p);
    printf("nets: %s\n", buf);

    /* 注册结果回调（等价于 sink 的 create 参数） */
    struct { AL_RESULT_CB cb; void *user; } cbp = {onResult, nullptr};
    p.data = &cbp; p.size = sizeof(cbp);
    AudioLinkCtrl(mgr, "sink", "set_result_cb", p);

    if (AudioLinkStart(mgr) != E_AL_OK) { fprintf(stderr, "start failed\n"); return 1; }

    if (wav) {
        std::vector<int16_t> pcm;
        int sr = 0;
        if (!readWav16kMono(wav, pcm, sr) || sr != 16000) {
            fprintf(stderr, "wav must be 16k mono pcm16: %s\n", wav);
            return 1;
        }
        printf("pushing %zu samples (%.1f s)...\n", pcm.size(), pcm.size() / 16000.0);
        /* 100ms 块模拟实时流 */
        const int chunk = 1600;
        for (size_t off = 0; off < pcm.size(); off += chunk) {
            int n = (int)std::min<size_t>(chunk, pcm.size() - off);
            AudioLinkPushPcm(mgr, pcm.data() + off, n * 2);
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); /* >2x 实时加速演示 */
        }
        /* 冲刷尾段 */
        AudioLinkCtrl(mgr, nullptr, "flush", AL_PARAM{nullptr, 0});
        std::this_thread::sleep_for(std::chrono::seconds(2));

        /* 等待任务线程空闲（在途段处理完）再取转写/卸载模型 */
        for (int i = 0; i < 100; i++) {
            char stats[256] = {0};
            AL_PARAM sp{stats, sizeof(stats)};
            AudioLinkCtrl(mgr, nullptr, "get_stats", sp);
            if (strstr(stats, "\"task_busy\":false")) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        /*---- 会议总结时序（显存受限：先卸 asr，总结后重载） ----*/
        char transcript[16384] = {0};
        p.data = transcript;
        p.size = sizeof(transcript);
        AudioLinkCtrl(mgr, "diarize", "get_transcript", p);
        printf("\n==== transcript ====\n%s\n", transcript);

        printf("unload asr model...\n");
        AudioLinkCtrl(mgr, "asr", "unload_model", AL_PARAM{nullptr, 0});

        AL_HANDLE sm = MeetingSummaryInit("config/summary/meeting_summary.json", devid);
        if (sm) {
            char summary[8192] = {0};
            p.data = summary;
            p.size = sizeof(summary);
            if (MeetingSummaryRun(sm, transcript, p) == E_AL_OK)
                printf("==== summary ====\n%s\n", summary);
            else
                fprintf(stderr, "summary run failed\n");
            MeetingSummaryDeinit(sm);
        } else {
            fprintf(stderr, "MeetingSummaryInit failed (可选功能，跳过)\n");
        }

        printf("reload asr model...\n");
        AudioLinkCtrl(mgr, "asr", "load_model", AL_PARAM{nullptr, 0});
        AudioLinkCtrl(mgr, nullptr, "new_meeting", AL_PARAM{nullptr, 0}); /* 开新会 */
    } else {
        printf("no -i given, running idle; Ctrl+C to quit (push PCM via AudioLinkPushPcm)\n");
        while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    AudioLinkStop(mgr);
    AudioLinkDeinit(mgr);
    return 0;
}
