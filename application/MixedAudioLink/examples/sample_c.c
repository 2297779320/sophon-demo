/*==============================================================================
 *
 * sample_c.c - 纯 C 程序调用 MixedAudioLink 库的最小示例
 *
 * 演示：静态库 libaudiolink.a 的 C 程序接入方式（音频流 + 会议总结完整时序）
 *
 * 编译（板端 SoC，SDK 为 sophon-sdk 路径；在库根目录先构建静态库）：
 *
 *   export SDK=/path/to/sophon-sdk
 *   cmake -B build -DSDK="$SDK" && cmake --build build
 *   gcc -std=c99 -O2 \
 *       -Icpp -I$SDK/include \
 *       examples/sample_c.c \
 *       -Lbuild -L$SDK/lib \
 *       -laudiolink -lbmlib -lbmruntime -lpthread \
 *       -o sample_c
 *
 * （或直接使用构建产物 build/sample_c，与下方手动 gcc 等价）
 *
 * 运行：
 *   音频流模式（默认全常驻：asr/spk/vad 与总结模型同时驻留，总结零加载延迟）：
 *   ./sample_c -c ../config/audio_link.json -s ../config/meeting_summary.json \
 *              -d 0 -i ../test16k.wav
 *   小显存设备加 --swap 走换入换出时序（总结前卸 asr，总结后重载）：
 *   ./sample_c --swap -c ../config/audio_link.json -i ../test16k.wav
 *   独立总结模式（对齐官方 meeting_summary 例程三件套用法，不依赖音频流）：
 *   ./sample_c -s ../config/meeting_summary.json -d 0 -t meeting.txt
 *   （音频流/独立总结两种模式总结结果均落盘 summary_output.txt）
 *
 *============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "audio_link.h"
#include "meeting_summary.h"

/*============================================================================
 * 极简 WAV(PCM16) 读取 —— 仅演示用，生产环境请用成熟解码库
 *==========================================================================*/
static int read_wav_mono16k(const char *path, short *buf, int max_frames)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, fp) != 44) { fclose(fp); return -1; }
    if (memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) { fclose(fp); return -1; }

    int sample_rate = hdr[24] | (hdr[25] << 8);
    int channels    = hdr[22] | (hdr[23] << 8);
    int bits        = hdr[34] | (hdr[35] << 8);
    if (sample_rate != 16000 || channels != 1 || bits != 16) {
        fprintf(stderr, "need 16k/mono/pcm16 wav\n");
        fclose(fp);
        return -1;
    }
    fseek(fp, 44, SEEK_END);
    long bytes = ftell(fp) - 44;
    fseek(fp, 44, SEEK_SET);
    int frames = (int)(bytes / 2);
    if (frames > max_frames) frames = max_frames;
    if (fread(buf, 2, frames, fp) != (size_t)frames) { fclose(fp); return -1; }
    fclose(fp);
    return frames;
}

/*============================================================================
 * 结果回调：音频流每识别完一段触发一次（diarize 已归属说话人）
 *==========================================================================*/
static void on_result(const AL_RESULT *r, void *user)
{
    (void)user;
    printf("[%6d-%6d ms] %s\n", r->start_ms, r->end_ms,
           r->line ? r->line : (r->text ? r->text : "(no text)"));
    fflush(stdout);
}

/*============================================================================
 * 便捷封装：读一个 Ctrl 字符串出参
 *==========================================================================*/
static void ctrl_str(AL_HANDLE h, const char *name, const char *cmd,
                     char *buf, int size)
{
    AL_PARAM p;
    p.data = buf;
    p.size = size;
    if (AudioLinkCtrl(h, name, cmd, p) != E_AL_OK)
        buf[0] = '\0';
}

/*============================================================================
 * 便捷封装：总结结果落盘 summary_output.txt（对齐官方例程 output.txt）
 *==========================================================================*/
static void write_summary_file(const char *summary)
{
    FILE *fp = fopen("summary_output.txt", "w");
    if (!fp) {
        fprintf(stderr, "write summary_output.txt failed\n");
        return;
    }
    fwrite(summary, 1, strlen(summary), fp);
    fclose(fp);
    printf("summary written to summary_output.txt\n");
}

int main(int argc, char *argv[])
{
    const char *audio_cfg  = "../config/audio_link.json";
    const char *summary_cfg = "../config/meeting_summary.json";
    const char *wav_path   = NULL;
    const char *text_path  = NULL;
    int dev_id = 0;
    int swap_mode = 0;   /* 1 = 换入换出时序（小显存设备）；默认全常驻 */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i + 1 < argc)       audio_cfg = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc)  summary_cfg = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc)  dev_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc)  wav_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)  text_path = argv[++i];
        else if (!strcmp(argv[i], "--swap"))              swap_mode = 1;
        else { printf("usage: %s -c cfg -s summary_cfg -d devid "
                      "[-i file.wav | -t meeting.txt] [--swap]\n", argv[0]); return 1; }
    }

    /*---- 独立总结模式：-t 给定文本文件时，不初始化音频流模块， ----
     *---- 直接 MeetingSummaryInit → MeetingSummaryRunFile → 落盘退出 ----*/
    if (text_path) {
        AL_HANDLE sm = MeetingSummaryInit(summary_cfg, dev_id);
        if (!sm) {
            /* Init 失败句柄为 NULL，错误已由 al_log(ERROR) 打印到 stderr */
            fprintf(stderr, "MeetingSummaryInit failed (see log)\n");
            return 1;
        }
        static char summary[8192];
        memset(summary, 0, sizeof(summary));
        AL_PARAM p;
        p.data = summary;
        p.size = sizeof(summary);
        if (MeetingSummaryRunFile(sm, text_path, p) == E_AL_OK) {
            printf("==== summary ====\n%s\n", summary);
            write_summary_file(summary);
        } else {
            char err[256] = {0};
            MeetingSummaryCtrl(sm, "get_last_error", (AL_PARAM){err, sizeof(err)});
            fprintf(stderr, "summary failed: %s\n", err);
            MeetingSummaryDeinit(sm);
            return 1;
        }
        MeetingSummaryDeinit(sm);
        return 0;
    }

    /*---- 1. 初始化音频流模块（加载 asr/spk/vad 三组模型，创建默认链路） ----*/
    AL_HANDLE mgr = AudioLinkInit(audio_cfg, dev_id);
    if (!mgr) {
        char err[256] = {0};
        AudioLinkCtrl(NULL, NULL, "get_last_error", (AL_PARAM){err, sizeof(err)});
        fprintf(stderr, "AudioLinkInit failed: %s\n", err);
        return 1;
    }

    /*---- 1.5 常驻模式（默认）：启动即加载总结模型，与音频流模型同时驻留 ----
     *---- （64TOPS 级设备推荐：总结零加载延迟，MeetingSummary 句柄复用）；
     *---- --swap 模式此处不加载，改在总结前换入换出（小显存设备） ----*/
    AL_HANDLE sm = NULL;
    if (!swap_mode) {
        sm = MeetingSummaryInit(summary_cfg, dev_id);
        if (!sm)
            fprintf(stderr, "MeetingSummaryInit failed, summary disabled "
                            "(显存不足时可加 --swap 走换入换出时序)\n");
    }

    /*---- 2. 注册结果回调（"讲话人N：xxx" 归属行）与退出钩子/状态快照 ----*/
    AL_RESULT_CB cb = on_result;
    AL_PARAM p;
    p.data = &cb;
    p.size = sizeof(cb);
    AudioLinkCtrl(mgr, "sink", "set_result_cb", p);

    /* 退出前自动写入运行状态快照（版本/时长/段数/丢弃/转写统计） */
    AudioLinkCtrl(mgr, NULL, "set_status_file",
                  (AL_PARAM){"runtime_status.md", 18});

    /*---- 3. 启动链路（内部：VAD 线程 + 任务线程） ----*/
    if (AudioLinkStart(mgr) != E_AL_OK) {
        fprintf(stderr, "AudioLinkStart failed\n");
        AudioLinkDeinit(mgr);
        return 1;
    }

    /*---- 4. 推入 PCM（生产环境由采集线程循环调用） ----*/
    if (wav_path) {
        static short pcm[16000 * 300];   /* 最大 300s */
        int frames = read_wav_mono16k(wav_path, pcm, 16000 * 300);
        if (frames < 0) {
            fprintf(stderr, "read wav failed: %s\n", wav_path);
            AudioLinkDeinit(mgr);
            return 1;
        }
        printf("pushing %d frames (%.1f s)...\n", frames, frames / 16000.0);

        const int chunk = 1600;   /* 100ms 步长，模拟实时流 */
        for (int off = 0; off < frames; off += chunk) {
            int n = frames - off < chunk ? frames - off : chunk;
            AudioLinkPushPcm(mgr, pcm + off, n * 2);   /* 非阻塞，永不丢调用 */
            usleep(50 * 1000);                          /* 2x 实时加速演示 */
        }

        /*---- 5. 冲刷尾段，等待任务线程空闲 ----*/
        AudioLinkCtrl(mgr, NULL, "flush", (AL_PARAM){NULL, 0});
        {
            char stats[256];
            int busy = 1, wait_ms = 0;
            while (busy && wait_ms < 10000) {
                usleep(100 * 1000);
                wait_ms += 100;
                stats[0] = '\0';
                ctrl_str(mgr, NULL, "get_stats", stats, sizeof(stats));
                busy = strstr(stats, "\"task_busy\":true") != NULL;
            }
        }
        usleep(500 * 1000);   /* 回调落盘余量 */
    } else {
        printf("no -i given, running idle (Ctrl+C quit)\n");
    }

    /*---- 6. 取转写文本集合（"讲话人N：xxx" 行） ----*/
    static char transcript[16384];
    memset(transcript, 0, sizeof(transcript));
    ctrl_str(mgr, "diarize", "get_transcript", transcript, sizeof(transcript));
    printf("\n==== transcript ====\n%s\n", transcript);

    /*---- 7. 会议总结 ----
     * 常驻模式（默认）：所有模型同时驻留，直接复用 MeetingSummary 句柄；
     * --swap 模式（小显存设备）：卸 asr → Init → Run → Deinit → 重载 asr ----*/
    if (wav_path && transcript[0]) {
        static char summary[8192];
        memset(summary, 0, sizeof(summary));
        p.data = summary;
        p.size = sizeof(summary);

        if (swap_mode) {
            printf("unload asr model...\n");
            AudioLinkCtrl(mgr, "asr", "unload_model", (AL_PARAM){NULL, 0});
            sm = MeetingSummaryInit(summary_cfg, dev_id);
        }

        if (sm) {
            if (MeetingSummaryRun(sm, transcript, p) == E_AL_OK) {
                printf("==== summary ====\n%s\n", summary);
                write_summary_file(summary);
            } else {
                char err[256] = {0};
                MeetingSummaryCtrl(sm, "get_last_error", (AL_PARAM){err, sizeof(err)});
                fprintf(stderr, "summary failed: %s\n", err);
            }
        } else {
            fprintf(stderr, "MeetingSummary unavailable (summary skipped)\n");
        }

        if (swap_mode) {
            if (sm) MeetingSummaryDeinit(sm);   /* 释放总结模型显存 */
            sm = NULL;
            printf("reload asr model...\n");
            AudioLinkCtrl(mgr, "asr", "load_model", (AL_PARAM){NULL, 0});
        }
        AudioLinkCtrl(mgr, NULL, "new_meeting", (AL_PARAM){NULL, 0});  /* 开新会 */
    }

    /*---- 8. 停止并释放（常驻模式：总结模型在此统一释放） ----*/
    AudioLinkStop(mgr);
    if (sm) MeetingSummaryDeinit(sm);
    AudioLinkDeinit(mgr);
    printf("done\n");
    return 0;
}
