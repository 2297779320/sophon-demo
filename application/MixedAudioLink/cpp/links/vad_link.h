//===----------------------------------------------------------------------===//
//
// vad_link.h - FSMN VAD 流式推理 + 切段状态机
// 协议参照 application/Audio_assistant/python/whisper_minicpm_llama3_vits.py
// （200ms 块 + cache + [beg,-1]/[-1,end] 事件语义）。
// 子网约定（均可 JSON 覆盖）：
//   inputs[0]        = fbank 特征 [1, T, 80]（静态 T）
//   inputs[1..n-1]   = in_cache（块间状态，静态 shape）
//   outputs[0]       = 每帧语音概率（[T] 或 [T,2]，后者取 [:,1]）
//   outputs[1..n-1]  = out_cache（d2d 回写 inputs[1..n-1]）
//
//===----------------------------------------------------------------------===//
#ifndef AL_VAD_LINK_H
#define AL_VAD_LINK_H

#include <atomic>
#include <deque>

#include "../al_feature.h"
#include "../link_mng.h"

namespace al {

class CVadLink : public CBaseLink {
public:
    CVadLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_VAD) {}
    ~CVadLink() override { deinitMem(); }

    int create(const JsonValue *cfg, AL_PARAM param) override;
    int stop() override;
    int ctrl(const std::string &cmd, AL_PARAM param) override;

    /** VAD 线程节奏入口：拉 PCM 块 → fbank → launch → 状态机 → 出段 */
    int tick();

    /** 冲刷：把正在进行的语音段按 is_final 出队 */
    int flush();

private:
    void deinitMem();
    int launch(const std::vector<float> &feat, int frames, std::vector<float> &probs);
    void resetCache();
    /** 段状态机软复位（launch 失败自愈用）：不动 cache 与时间轴 m_stream_ms */
    void softReset();
    void feedState(const std::vector<float> &probs, int frames, int64_t chunk_start_ms,
                   const std::vector<int16_t> &chunk_pcm);
    void emitSegment(int64_t beg_ms, int64_t end_ms, bool final_seg);

    /* 子网 */
    std::string m_model, m_net;
    const bm_net_info_t *m_info = nullptr;
    int m_feat_in_idx = 0;        /* 特征输入下标 */
    int m_prob_out_idx = 0;       /* 概率输出下标 */
    std::vector<int> m_cache_in;  /* cache 输入下标（即其余输入） */
    std::vector<int> m_cache_out; /* cache 输出下标（其余输出） */
    int m_max_frames = 0;         /* 静态 T */
    int m_bins = 80;

    /* 前处理 */
    std::unique_ptr<FbankComputer> m_fbank;
    std::vector<float> m_cmvn_mean, m_cmvn_std;
    bool m_has_cmvn = false;

    /* 状态机参数（JSON vad 节） */
    int m_chunk_ms = 200;
    int m_padding_ms = 300;
    int m_min_sil_ms = 500;
    int m_min_speech_ms = 250;
    int m_max_seg_ms = 30000;     /* 强制切段上限（防连续噪音下内存无界） */
    float m_noise_thres = 0.5f;   /* speech_noise_thres：p = p_speech - thres*p_sil */

    /* 能量前置门（JSON vad.energy_gate，默认关闭）：
     * 非语音 + 低 RMS 块跳过 VAD 推理与 fbank（省 TPU/功耗）；
     * hold_ms 挂起期保证语音结束后状态机正常收尾 */
    std::atomic<bool> bGateEnable{false};   /* ctrl 线程写 / vad 线程读，原子消除数据竞争 */
    std::atomic<int> iGateRms{300};
    std::atomic<int> iGateHoldMs{600};
    std::atomic<int> iGateHoldLeftMs{0};

    /* 运行态 */
    bool m_in_speech = false;
    int64_t m_speech_beg_ms = 0;
    int64_t m_last_voice_ms = -1;
    int64_t m_stream_ms = 0;          /* 已消费音频流时间 */
    int m_sil_frames = 0;
    int m_speech_frames = 0;
    int m_onset_frame = -1;           /* 首个语音帧号（确认前记录，音频从此截入） */
    int64_t m_onset_chunk_beg = 0;    /* onset 帧所在块的时间起点 */
    std::vector<int16_t> m_preroll;   /* 语音前 padding 预卷缓冲 */
    std::vector<int16_t> m_seg_pcm;   /* 当前段累积音频 */
    int iFailStreak = 0;            /* launch 连续失败计数（成功清零，达 3 升级 logErr） */    std::atomic<bool> m_flush_req{false};
};

} /* namespace al */
#endif /* AL_VAD_LINK_H */
