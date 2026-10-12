//===----------------------------------------------------------------------===//
//
// vad_segmenter.h - funasr FsmnVADStreaming 端点状态机（与 bmrt 解耦，可离线逐帧对拍）
//
// 逐行对应 funasr/models/fsmn_vad_streaming/model.py：
//   WindowDetector.DetectOneFrame    → detectOneFrame 前半段（20 帧环形窗投票）
//   FsmnVADStreaming.DetectOneFrame  → detectOneFrame 四分支状态机
//   OnVoiceStart/Detected/Silence/End→ 同名 onXxx
//   PopDataBufTillFrame              → dropBeforeFrame
//   PopDataToOutputBuf               → popOut /（流末）sent-end 分支
//   ResetDetection → DropCachedFrames→ resetDetection
//
// 音频口径与 funasr 完全一致（逐帧 pop 累加，见 popOut）：
//   - 正常判停（end_point_is_sent_end=False）：每帧 pop frame_shift 个样本，
//     段长 = (end-start+1)*shift，**不加** extra（funasr 的 extra_sample 只用于
//     长度校验与 sent_end 分支，这点极易读错）
//   - 流末收尾（sent_end=True）：data_to_pop = max(shift+extra, 缓冲剩余)，
//     把缓冲剩余音频全部并入段尾
//   - 因此当 end_frame 落后于缓冲头时（do_extend=0 的回看判停），末尾 pop 会多吐
//     1 帧（PCM 比 end_ms 长 10ms）——该行为与 funasr 一致，不做"修正"
//
// 时间口径：帧 k ↔ 流内样本 k*frame_samples ↔ 时间 k*frame_ms（与切块无关，
//   分钟级无累计漂移）；段起点含 latency 回看（win_size + lookback）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_VAD_SEGMENTER_H
#define AL_VAD_SEGMENTER_H

#include <algorithm>
#include <cstdint>
#include <vector>

namespace al {

/** 一段 VAD 切段结果 */
struct VadSegment {
    int64_t beg_ms = 0;              /* confirmed_start_frame * frame_ms */
    int64_t end_ms = 0;              /* (confirmed_end_frame + 1) * frame_ms */
    bool is_final = false;           /* 末帧收尾（end_point_is_sent_end / 流结束） */
    std::vector<int16_t> pcm;        /* 段音频（逐帧 pop 累加的结果） */
};

/** FSMN VAD 端点状态机。无外部依赖，可独立驱动做逐帧对拍。 */
class FsmnVadSegmenter {
public:
    struct Config {
        int window_size_ms = 200;        /* window_size_ms */
        int sil_to_speech_ms = 150;      /* sil_to_speech_time_thres */
        int speech_to_sil_ms = 150;      /* speech_to_sil_time_thres */
        int max_end_silence_ms = 800;    /* max_end_silence_time */
        int max_start_silence_ms = 3000; /* max_start_silence_time */
        int max_single_seg_ms = 20000;   /* max_single_segment_time */
        bool do_extend = true;           /* do_extend */
        int lookback_start_ms = 200;     /* lookback_time_start_point */
        int lookahead_end_ms = 100;      /* lookahead_time_end_point */
        int detect_mode = 1;             /* 1=多句（连续流）, 0=单句 */
        int frame_ms = 10;               /* frame_in_ms */
        int frame_samples = 160;         /* frame_shift * fs / 1000 */
        int extra_samples = 240;         /* (frame_length - frame_shift) * fs / 1000 */
    };

    void init(const Config &c) {
        m_cfg = c;
        recompute();
        m_win_state.assign((size_t)m_win_size_frame, 0);
        resetAll();
    }

    /*---- 运行时热调（派生常量一并重算）----*/
    void setMaxEndSilenceMs(int v) {
        m_cfg.max_end_silence_ms = v;
        m_max_end_sil_cnt = std::max(0, m_cfg.max_end_silence_ms - m_cfg.speech_to_sil_ms);
    }
    void setSilToSpeechMs(int v) {
        m_cfg.sil_to_speech_ms = v;
        m_sil2sp_frmcnt = frameDiv(m_cfg.sil_to_speech_ms);
    }
    void setMaxSingleSegMs(int v) {
        m_cfg.max_single_seg_ms = v;
        m_max_single_frames = frameDiv(m_cfg.max_single_seg_ms);
    }

    /*---- 查询（日志 / get_params）----*/
    const Config &cfg() const { return m_cfg; }
    int winSizeFrame() const { return m_win_size_frame; }
    int sil2spFrmcnt() const { return m_sil2sp_frmcnt; }
    int sp2silFrmcnt() const { return m_sp2sil_frmcnt; }
    int maxEndSilCnt() const { return m_max_end_sil_cnt; }
    int latencyFrames() const { return m_latency_frames; }
    int maxSingleFrames() const { return m_max_single_frames; }

    bool inSpeech() const { return m_vad_state == 1; }
    int vadState() const { return m_vad_state; }

    /* 全量复位：清端点状态机 + 滑动窗 + 缓冲（流重启 / create） */
    void resetAll() {
        m_vad_state = 0;                 /* kVadInStateStartPointNotDetected */
        m_confirmed_start_frame = -1;
        m_confirmed_end_frame = -1;
        m_latest_conf_speech_frame = 0;  /* funasr Stats 口径：初值为 0 */
        m_lastest_conf_silence_frame = -1;
        m_continous_silence_frames = 0;
        m_number_end_detected = 0;
        m_last_frame = -1;
        m_last_end_drop_frame = -1;
        std::fill(m_win_state.begin(), m_win_state.end(), 0);
        m_win_sum = 0;
        m_win_pos = 0;
        m_pre_frame_state = 0;
        m_buf.clear();
        m_buf_beg_sample = 0;
        m_out_pcm.clear();
        m_pending.clear();
    }

    /* 推入一块 PCM（绝对样本下标 abs_beg）+ 本块逐帧判决；产出的段追加到 pending()。
     * is_final_batch=true 表示本批末帧为整条流末帧（对应 funasr is_final_frame）。 */
    void push(const std::vector<int16_t> &pcm, int64_t abs_beg,
              const std::vector<uint8_t> &is_speech, int64_t frame0, bool is_final_batch) {
        appendPcm(pcm, abs_beg);
        const int frames = (int)is_speech.size();
        for (int f = 0; f < frames; f++)
            detectOneFrame(0 != is_speech[(size_t)f], frame0 + (int64_t)f,
                           is_final_batch && (f == frames - 1));
    }

    /* 流尾兜底：把在途段按 final 收尾（尾帧无推理 / 自愈路径用） */
    void closeInProgress() {
        if (m_vad_state == 1 && m_confirmed_start_frame >= 0) {
            int64_t end_frame = m_last_frame;
            if (end_frame < m_confirmed_start_frame) end_frame = m_confirmed_start_frame;
            onVoiceEnd(end_frame, true);
        }
        if (m_vad_state == 1) m_vad_state = 2;
        if (m_cfg.detect_mode == 1) resetDetection();
    }

    /* 本批产出的段（调用方消费后应 clear） */
    std::vector<VadSegment> &pending() { return m_pending; }

private:
    int frameDiv(int ms) const {
        const int S = (m_cfg.frame_ms > 0) ? m_cfg.frame_ms : 10;
        return ms / S;
    }

    void recompute() {
        m_win_size_frame = std::max(1, frameDiv(m_cfg.window_size_ms));
        m_sil2sp_frmcnt = frameDiv(m_cfg.sil_to_speech_ms);
        m_sp2sil_frmcnt = frameDiv(m_cfg.speech_to_sil_ms);
        m_max_end_sil_cnt = std::max(0, m_cfg.max_end_silence_ms - m_cfg.speech_to_sil_ms);
        m_latency_frames =
            m_win_size_frame + (m_cfg.do_extend ? frameDiv(m_cfg.lookback_start_ms) : 0);
        m_max_single_frames = frameDiv(m_cfg.max_single_seg_ms);
        if (m_cfg.frame_samples <= 0) m_cfg.frame_samples = 160;
        if (m_cfg.frame_ms <= 0) m_cfg.frame_ms = 10;
        if (m_cfg.extra_samples < 0) m_cfg.extra_samples = 0;
    }

    /*---- 滚动 PCM 缓冲（等价 funasr stats.data_buf，按绝对样本下标寻址）----*/
    void appendPcm(const std::vector<int16_t> &pcm, int64_t abs_beg) {
        if (pcm.empty()) return;
        if (m_buf.empty()) {
            m_buf_beg_sample = abs_beg;
        } else {
            int64_t expect = m_buf_beg_sample + (int64_t)m_buf.size();
            if (abs_beg > expect) {
                /* 缺口：补零对齐帧栅格（正常不出现；上游丢块时保时间轴正确） */
                m_buf.resize(m_buf.size() + (size_t)(abs_beg - expect), 0);
            } else if (abs_beg < expect) {
                size_t overlap = (size_t)(expect - abs_beg);
                if (overlap >= m_buf.size()) return;          /* 完全回退：丢弃本块 */
                m_buf.erase(m_buf.begin(), m_buf.begin() + overlap);
                m_buf_beg_sample = abs_beg;
            }
        }
        m_buf.insert(m_buf.end(), pcm.begin(), pcm.end());
    }

    /* 丢弃帧号 F 之前的音频（funasr PopDataBufTillFrame / DropCachedFrames） */
    void dropBeforeFrame(int64_t frame) {
        int64_t want_beg = frame * m_cfg.frame_samples;
        if (want_beg <= m_buf_beg_sample) return;
        int64_t drop = want_beg - m_buf_beg_sample;
        if (drop >= (int64_t)m_buf.size()) {
            m_buf.clear();
            m_buf_beg_sample = want_beg;
            return;
        }
        m_buf.erase(m_buf.begin(), m_buf.begin() + (size_t)drop);
        m_buf_beg_sample = want_beg;
    }

    /* 取 n_samples 个样本进当前段输出缓冲（等价 funasr data_buf → output_data_buf） */
    void takeToOutput(int64_t n_samples) {
        int64_t take = n_samples;
        if (take > (int64_t)m_buf.size()) take = (int64_t)m_buf.size();
        if (take <= 0) return;
        m_out_pcm.insert(m_out_pcm.end(), m_buf.begin(), m_buf.begin() + (size_t)take);
        m_buf.erase(m_buf.begin(), m_buf.begin() + (size_t)take);
        m_buf_beg_sample += take;
    }

    /* 等价 funasr PopDataToOutputBuf(..., first/last_is_end_point=False)：
     *   ① PopDataBufTillFrame(s_frame) ② 取 frame_shift 个样本 */
    void popOut(int64_t s_frame) {
        dropBeforeFrame(s_frame);
        takeToOutput(m_cfg.frame_samples);
    }

    int64_t bufBegFrame() const {
        return (m_cfg.frame_samples > 0) ? (m_buf_beg_sample / m_cfg.frame_samples) : 0;
    }

    /*---- funasr OnVoiceStart / OnVoiceDetected / OnSilenceDetected / OnVoiceEnd ----*/
    void onVoiceDetected(int64_t valid_frame) {
        m_latest_conf_speech_frame = valid_frame;
        popOut(valid_frame);
    }

    void onSilenceDetected(int64_t valid_frame) {
        m_lastest_conf_silence_frame = valid_frame;
        /* 起点未确认时丢弃该帧之前的音频（缓冲即 data_buf，仅保留 latency 窗） */
        if (m_vad_state == 0) dropBeforeFrame(valid_frame);
    }

    void onVoiceStart(int64_t start_frame) {
        m_confirmed_start_frame = start_frame;
        if (m_vad_state == 0) {
            m_out_pcm.clear();          /* 新段：清洗输出缓冲（funasr 新开 output entry） */
            popOut(start_frame);
        }
    }

    void onVoiceEnd(int64_t end_frame, bool is_last) {
        for (int64_t t = m_latest_conf_speech_frame + 1; t < end_frame; t++) onVoiceDetected(t);
        if (end_frame < m_confirmed_start_frame) end_frame = m_confirmed_start_frame;
        m_confirmed_end_frame = end_frame;

        if (is_last) {
            /* end_point_is_sent_end：expected = frm_cnt*shift + extra，再与缓冲剩余取大 */
            dropBeforeFrame(end_frame);
            int64_t expected = (int64_t)m_cfg.frame_samples + m_cfg.extra_samples;
            if ((int64_t)m_buf.size() > expected) expected = (int64_t)m_buf.size();
            takeToOutput(expected);
        } else {
            popOut(end_frame);
        }
        m_number_end_detected++;

        VadSegment seg;
        seg.beg_ms = m_confirmed_start_frame * m_cfg.frame_ms;
        seg.end_ms = (end_frame + 1) * m_cfg.frame_ms;
        seg.is_final = is_last;
        seg.pcm = std::move(m_out_pcm);
        m_out_pcm.clear();
        m_pending.push_back(std::move(seg));

        /* 丢弃时机与 funasr 一致：由 ResetDetection → DropCachedFrames(end_ms/frame_in_ms)
         * 完成（多句模式出段后即刻执行；此处已因 pop 前进，drop 通常是 no-op） */
        m_last_end_drop_frame = end_frame + 1;
        m_vad_state = 2;                  /* kVadInStateEndPointDetected */
    }

    void resetDetection() {
        m_continous_silence_frames = 0;
        m_latest_conf_speech_frame = 0;   /* funasr Stats 口径：复位为 0 */
        m_lastest_conf_silence_frame = -1;
        m_confirmed_start_frame = -1;
        m_confirmed_end_frame = -1;
        m_vad_state = 0;
        std::fill(m_win_state.begin(), m_win_state.end(), 0);
        m_win_sum = 0;
        m_win_pos = 0;
        m_pre_frame_state = 0;
        if (m_last_end_drop_frame >= 0) {
            dropBeforeFrame(m_last_end_drop_frame);
            m_last_end_drop_frame = -1;
        }
    }

    void detectOneFrame(bool is_speech, int64_t cur, bool is_final) {
        m_last_frame = cur;

        /*---- funasr WindowDetector.DetectOneFrame：20 帧环形窗投票 ----*/
        int v = is_speech ? 1 : 0;
        m_win_sum -= m_win_state[(size_t)m_win_pos];
        m_win_sum += v;
        m_win_state[(size_t)m_win_pos] = v;
        m_win_pos = (m_win_pos + 1) % m_win_size_frame;

        const int pre = m_pre_frame_state;
        int change;   /* 0=Sil2Sil, 1=Sil2Speech, 2=Speech2Sil, 3=Speech2Speech */
        if (pre == 0 && m_win_sum >= m_sil2sp_frmcnt) { m_pre_frame_state = 1; change = 1; }
        else if (pre == 1 && m_win_sum <= m_sp2sil_frmcnt) { m_pre_frame_state = 0; change = 2; }
        else change = (pre == 0) ? 0 : 3;

        const int shift = m_cfg.frame_ms;

        if (change == 1) {                /* Sil2Speech */
            m_continous_silence_frames = 0;
            if (m_vad_state == 0) {       /* kVadInStateStartPointNotDetected */
                int64_t start_frame = std::max<int64_t>(bufBegFrame(), cur - m_latency_frames);
                onVoiceStart(start_frame);
                m_vad_state = 1;          /* kVadInStateInSpeechSegment */
                for (int64_t t = start_frame + 1; t <= cur; t++) onVoiceDetected(t);
            } else if (m_vad_state == 1) {
                for (int64_t t = m_latest_conf_speech_frame + 1; t < cur; t++) onVoiceDetected(t);
                if (cur - m_confirmed_start_frame + 1 > m_max_single_frames) {
                    onVoiceEnd(cur, false);
                } else if (!is_final) {
                    onVoiceDetected(cur);
                } else {
                    onVoiceEnd(cur, true);   /* MaybeOnVoiceEndIfLastFrame */
                }
            }
        } else if (change == 2) {         /* Speech2Sil */
            m_continous_silence_frames = 0;
            if (m_vad_state == 1) {
                if (cur - m_confirmed_start_frame + 1 > m_max_single_frames) onVoiceEnd(cur, false);
                else if (!is_final) onVoiceDetected(cur);
                else onVoiceEnd(cur, true);
            }
        } else if (change == 3) {         /* Speech2Speech */
            m_continous_silence_frames = 0;
            if (m_vad_state == 1) {
                if (cur - m_confirmed_start_frame + 1 > m_max_single_frames) onVoiceEnd(cur, false);
                else if (!is_final) onVoiceDetected(cur);
                else onVoiceEnd(cur, true);
            }
        } else {                          /* Sil2Sil */
            m_continous_silence_frames++;
            if (m_vad_state == 0) {
                /* 单句模式纯静音超时 / 末帧仍未检出任何端点 → 假起点+假终点（不出段） */
                if ((m_cfg.detect_mode == 0 &&
                     m_continous_silence_frames * shift > m_cfg.max_start_silence_ms) ||
                    (is_final && m_number_end_detected == 0)) {
                    for (int64_t t = m_lastest_conf_silence_frame + 1; t < cur; t++)
                        onSilenceDetected(t);
                    m_confirmed_start_frame = 0;   /* OnVoiceStart(0, fake_result=True) */
                    m_confirmed_end_frame = 0;     /* OnVoiceEnd(0, fake_result=True) */
                    m_vad_state = 2;
                } else if (cur >= m_latency_frames) {
                    onSilenceDetected(cur - m_latency_frames);
                }
            } else if (m_vad_state == 1) {
                if (m_continous_silence_frames * shift >= m_max_end_sil_cnt) {
                    /* 判停 + do_extend 前瞻补偿（funasr lookback_frame 推导） */
                    int64_t lookback = m_max_end_sil_cnt / shift;
                    if (m_cfg.do_extend) {
                        lookback -= m_cfg.lookahead_end_ms / shift;
                        lookback -= 1;
                        if (lookback < 0) lookback = 0;
                    }
                    onVoiceEnd(cur - lookback, false);
                } else if (cur - m_confirmed_start_frame + 1 > m_max_single_frames) {
                    onVoiceEnd(cur, false);
                } else if (m_cfg.do_extend && !is_final) {
                    if (m_continous_silence_frames <= m_cfg.lookahead_end_ms / shift)
                        onVoiceDetected(cur);
                } else if (is_final) {
                    onVoiceEnd(cur, true);
                }
            }
        }

        if (m_vad_state == 2 && m_cfg.detect_mode == 1) resetDetection();
    }

    Config m_cfg;
    /* 派生常量 */
    int m_win_size_frame = 20;
    int m_sil2sp_frmcnt = 15;
    int m_sp2sil_frmcnt = 15;
    int m_max_end_sil_cnt = 650;
    int m_latency_frames = 40;
    int m_max_single_frames = 2000;

    /* WindowDetector 运行态 */
    std::vector<int> m_win_state;
    int m_win_sum = 0;
    int m_win_pos = 0;
    int m_pre_frame_state = 0;        /* 0=Sil, 1=Speech */

    /* 端点状态机运行态 */
    int m_vad_state = 0;              /* 0=StartNotDetected, 1=InSpeechSegment, 2=EndPointDetected */
    int64_t m_confirmed_start_frame = -1;
    int64_t m_confirmed_end_frame = -1;
    int64_t m_latest_conf_speech_frame = 0;
    int64_t m_lastest_conf_silence_frame = -1;
    int m_continous_silence_frames = 0;
    int m_number_end_detected = 0;
    int64_t m_last_frame = -1;
    int64_t m_last_end_drop_frame = -1;   /* 最近一次出段的 end+1（ResetDetection 丢帧用） */

    /* 滚动 PCM 缓冲：[m_buf_beg_sample, m_buf_beg_sample + m_buf.size())（data_buf） */
    std::vector<int16_t> m_buf;
    int64_t m_buf_beg_sample = 0;
    /* 当前段已 pop 出的音频（output_data_buf 的单段等价体） */
    std::vector<int16_t> m_out_pcm;

    std::vector<VadSegment> m_pending;
};

} /* namespace al */
#endif /* AL_VAD_SEGMENTER_H */
