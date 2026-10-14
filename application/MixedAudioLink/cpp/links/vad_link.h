//===----------------------------------------------------------------------===//
//
// vad_link.h - FSMN VAD 流式推理 + 切段
// 协议参照 FunASR fsmn_vad_streaming（WavFrontendOnline + FsmnVADStreaming）：
//   fbank80 → LFR(lfr_m=5) → CMVN → 逐帧 248 类 PDF 判决 → 端点状态机切段。
//
// 模型基准：iic/speech_fsmn_vad_zh-cn-16k-common-onnx（同 iic/...-pytorch）
//   frontend: WavFrontendOnline  fs=16000 hamming n_mels=80 25/10 dither=0 lfr=5/1
//   encoder : input_dim=400  output_dim=248
//   model   : FsmnVADStreaming   detect_mode=1(多句)  sil_pdf_ids=[0]
//             speech_noise_thres=0.6  speech_2_noise_ratio=1.0
//             window_size_ms=200  sil_to_speech=150  speech_to_sil=150
//             max_end_silence_time=800  max_start_silence_time=3000
//             max_single_segment_time=20000
//             do_extend=1  lookback_start=200  lookahead_end=100
//   （snr_thres / decibel_thres 均为 -100.0、fe_prior_thres=1e-4，实测量纲下恒真→不起作用）
//
// 子网约定（**全流程静态**，均可 JSON 覆盖）：
//   inputs[0]        = 特征 [1, T, 400]（**固定 T**，每次 launch 恰好 T 行，T 由 bmodel 决定）
//   inputs[1..n-1]   = in_cache（块间状态，静态 shape，如 fsmn 的 128x19x1 x4）
//   outputs[0]       = 每帧 PDF 概率 [1, T, C]（C=248，网络已含 Softmax）
//   outputs[1..n-1]  = out_cache（d2d 回写 inputs[1..n-1]）
//
// **静态批处理约定（v2）**：
//   VAD 已放弃 dynamic shape，改回「单网静态 shape」——T 固定，必须与 vad.chunk_ms 匹配
//   （200ms → 20 帧，见 docs/README.md）。前端每块产出的 LFR 帧数并非固定
//   （首块 14、后续 20），故 vad_link 内维护**特征 FIFO**：攒满 T 行才 launch 一次，
//   一次 launch 的 T 行判决全部喂状态机；流末不足 T 行时**补零到 T** 跑最后一组，
//   只取前 rem 行判决（因果 FSMN：补零行不影响前 rem 行输出）。
//   PCM 与判决解耦：PCM 随块即时推入状态机（保证滚动缓冲连续），判决按 T 组补齐后推。
//
// 与官方前端对齐的要点（数值基准见 tools/vad_ref.py）：
//   1) fbank 为 kaldi 语义（int16 量纲、逐帧去直流、high_freq=-400）
//   2) LFR：首部补 (lfr_m-1)/2 帧首帧；流式下尾部**延迟**到 flush 才补末帧
//      （等价 funasr WavFrontendOnline 的 lfr_splice_cache：非 final 块不右补）
//   3) CMVN：(x + AddShift) * Rescale
//   4) 判决：speech iff (1 - P_sil) >= P_sil^ratio + speech_noise_thres
//      （ratio=1.0、thres=0.6 等价 P_sil <= 0.2）
//   5) 切段：funasr WindowDetector（20 帧环形窗投票）+ 端点状态机 → 见 vad_segmenter.h
//
//===----------------------------------------------------------------------===//
#ifndef AL_VAD_LINK_H
#define AL_VAD_LINK_H

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "../al_feature.h"
#include "../link_mng.h"
#include "../vad_segmenter.h"

namespace al {

class CVadLink : public CBaseLink {
public:
    CVadLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_VAD) {}
    ~CVadLink() override { deinitMem(); }

    int create(const JsonValue *cfg, AL_PARAM param) override;
    int stop() override;
    int ctrl(const std::string &cmd, AL_PARAM param) override;

    /** VAD 线程节奏入口：拉 PCM 块 → fbank → LFR → CMVN → launch → 状态机 → 出段 */
    int tick();

    /** 冲刷：把正在进行的语音段按 is_final 出队 */
    int flush();

private:
    void deinitMem();
    /** 静态 T 启动：以固定 T 行特征跑一次网络，取 T 行 [C] 概率 */
    int launchT(const std::vector<float> &feat, std::vector<float> &probs);
    void resetCache();
    /** 段状态机软复位：复位端点状态机与滚动 PCM 缓冲，不动设备 cache、不动时间轴
     *  m_stream_ms，也不动前处理（fbank/LFR）连续态——前处理跨块累积，复位会撕裂帧栅格 */
    void softReset();
    /** 处理一个 PCM 块：PCM 入状态机 → fbank/LFR/CMVN → 追加 FIFO → 凑满 T 组推理 */
    int processChunk(const FramePtr &chunk);
    /** flush 收尾：补 LFR 尾帧入 FIFO → 跑完整组 → 尾组补零跑完（末行置 final） */
    int finishFlush();

    /*---- 固定 T 静态批处理 ----*/
    /** PCM 即时推入状态机（连续），与判决解耦（判决按 T 组补齐后再推） */
    void pushPcmToSeg(const std::vector<int16_t> &pcm, int64_t abs_beg);
    /** 把 rows 行特征追加到 FIFO（frame0 为这批行的绝对首帧号；gated=整块被能量门跳过） */
    void appendFifo(const std::vector<float> &feat, int rows, int64_t frame0, bool gated);
    /** 消费 FIFO：凑满 T 行即成组推理；flush_tail=true 时把不足 T 的尾组补零跑完，
     *  返回**是否已给整条流末帧置过 is_final**（false → 调用方需 closeInProgress 兜底） */
    bool drainFifo(bool flush_tail);
    /** 跑一个 T 行组：real_rows ≤ T（尾部补零行不喂判决），is_final 置该组末个**真实**行 */
    void runGroup(int real_rows, int64_t frame0, bool is_final);
    /** 概率 [frames, C] → 逐帧判决（funasr GetFrameState 口径） */
    std::vector<uint8_t> probsToDecisions(const std::vector<float> &probs, int frames);
    /** 判决喂状态机（PCM 已由 pushPcmToSeg 推入，此处只推判决） */
    void pushDecisions(const std::vector<uint8_t> &is_speech, int64_t frame0, bool is_final);
    void clearFifo();

    /** 把 FsmnVadSegmenter 产出的段下发到下游（并清空 pending） */
    void drainSegments();
    /** 收尾兜底：把在途段按 final 下发（尾帧无推理 / 自愈路径用） */
    void closeInProgress();

    bool inSpeech() const { return m_seg.inSpeech(); }

    /* 子网 */
    std::string m_model, m_net;
    const bm_net_info_t *m_info = nullptr;
    uint32_t m_gen = 0;                /* create 时模型的加载代次（见 SModel::gen） */
    int m_feat_in_idx = 0;        /* 特征输入下标 */
    int m_prob_out_idx = 0;       /* 概率输出下标 */
    std::vector<int> m_cache_in;  /* cache 输入下标（即其余输入） */
    std::vector<int> m_cache_out; /* cache 输出下标（即其余输出） */
    int m_T = 0;                  /* 静态 bmodel 固定帧数（= 输入 shape dim[1]），每次 launch 恰好 T 行 */
    int m_feat_dim = 400;         /* 特征输入维数（= fbank_bins * lfr_m） */
    int m_num_pdf = 1;            /* 概率输出类数（fsmn_vad 为 248） */
    bool m_is_dynamic = false;    /* 子网是否 dynamic（本工程要求静态；create 时拒绝 dynamic 子网） */

    /* 固定 T 静态批处理：特征 FIFO（攒满 T 行才 launch） */
    std::vector<float> m_fifo;          /* 待 launch 的行（rows*feat_dim，行优先） */
    std::vector<uint8_t> m_fifo_gated;  /* 每行是否被能量门跳过（整组全跳过才不发网络） */
    int m_fifo_rows = 0;                /* FIFO 行数 */
    int64_t m_fifo_frame0 = 0;          /* FIFO 首行的绝对帧号 */

    /* 前处理：流式 fbank→LFR→CMVN（采样级 carry-over + LFR 拼接，见 al_feature.h::StreamFrontend） */
    StreamFrontend m_front;
    int m_lfr_m = 5;              /* LFR 拼接帧数（funasr lfr_m；create 已注入 m_front） */
    int m_lfr_n = 1;              /* LFR 步长（funasr lfr_n） */
    bool m_has_cmvn = false;      /* 仅用于日志（CMVN 应用在 m_front 内部） */

    /* 判决参数（JSON vad 节） */
    float m_noise_thres = 0.6f;          /* speech_noise_thres（funasr 加性阈值） */
    float m_speech_2_noise_ratio = 1.0f; /* speech_2_noise_ratio（噪声项指数） */
    int m_sil_pdf_id = 0;                /* 静音 PDF 类下标（funasr sil_pdf_ids[0]） */
    int m_chunk_ms = 200;

    /* 端点状态机（funasr WindowDetector + FsmnVADStreaming，见 vad_segmenter.h） */
    FsmnVadSegmenter m_seg;

    /* 能量前置门（JSON vad.energy_gate，默认关闭）：
     * 非语音 + 低 RMS 块跳过**网络推理**（省 TPU/功耗），fbank/LFR 前处理照常推进
     * （保证帧栅格跨块连续），低能量块合成全静音判决喂状态机；
     * hold_ms 挂起期保证语音结束后状态机正常收尾 */
    std::atomic<bool> bGateEnable{false};   /* ctrl 线程写 / vad 线程读，原子消除数据竞争 */
    std::atomic<int> iGateRms{300};
    std::atomic<int> iGateHoldMs{600};
    std::atomic<int> iGateHoldLeftMs{0};

    /* 运行态 */
    int64_t m_stream_ms = 0;          /* 已消费音频流时间 */
    int iFailStreak = 0;            /* launch 连续失败计数（成功清零，达 3 升级 logErr） */
    std::atomic<bool> m_flush_req{false};
};

} /* namespace al */
#endif /* AL_VAD_LINK_H */
