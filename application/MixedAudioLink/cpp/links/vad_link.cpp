//===----------------------------------------------------------------------===//
//
// vad_link.cpp - FSMN VAD 实现
//
//===----------------------------------------------------------------------===//
#include "vad_link.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "../al_json.h"   /* JsonValue 完整定义（link_mng.h 仅前置声明） */

namespace al {

int CVadLink::create(const JsonValue *cfg, AL_PARAM param) {
    (void)param;
    const JsonValue *vad_cfg = mng()->section("vad");
    if (vad_cfg) {
        m_chunk_ms = (int)vad_cfg->getNum("chunk_ms", 200);
        m_padding_ms = (int)vad_cfg->getNum("speech_padding_ms", 300);
        m_min_sil_ms = (int)vad_cfg->getNum("min_silence_ms", 500);
        m_min_speech_ms = (int)vad_cfg->getNum("min_speech_ms", 250);
        m_max_seg_ms = (int)(vad_cfg->getNum("max_seg_s", 30) * 1000);
        if (m_max_seg_ms < 1000) {
            mng()->logWarn("vad.max_seg_s < 1, fallback 30");
            m_max_seg_ms = 30000;   /* 下界钳制 */
        }
        if (m_max_seg_ms > 2147000) {
            /* 上界钳制：防 (int)(val*1000) 溢出（与 set_max_seg_s 的 [1,2147] 同源） */
            mng()->logWarn("vad.max_seg_s > 2147, fallback 30");
            m_max_seg_ms = 30000;
        }
        if (vad_cfg->isObj()) {
            auto nt = vad_cfg->get("speech_noise_thres");
            if (nt && nt->isNum()) m_noise_thres = (float)nt->num;
        }
        /* 能量前置门（默认关闭）；enable 兼容 BOOL 与 NUM 0/1
         * （专用命令写回经 UpdateCfgNum 为数值，保证 save_params 往返一致） */
        auto eg = vad_cfg->get("energy_gate");
        if (eg && eg->isObj()) {
            auto ev = eg->get("enable");
            if (ev) bGateEnable = ev->isNum() ? (0 != ev->num) : (JsonValue::BOOL == ev->type && ev->b);
            iGateRms = (int)eg->getNum("rms_thres", 300);
            iGateHoldMs = (int)eg->getNum("hold_ms", 600);
            if (iGateHoldMs < 0) iGateHoldMs = 600;
        }
    }

    /* 定位模型与子网 */
    CModelHub &hub = mng()->hub();
    m_model = ""; m_net = "";
    /* 管理器约定 VAD 模型名为 JSON models 下的 "vad" 键 */
    if (hub.hasModel("vad")) m_model = "vad";
    else { mng()->log("vad model must be named 'vad' in JSON models"); return E_AL_NET_NOT_MATCH; }
    m_net = hub.netByRole(m_model, "vad");
    if (m_net.empty()) { mng()->log("no net with role 'vad'"); return E_AL_NET_NOT_MATCH; }
    m_info = hub.netInfo(m_model, m_net);
    if (!m_info || m_info->input_num < 1 || m_info->output_num < 1)
        return E_AL_NET_NOT_MATCH;

    m_feat_in_idx = 0;
    m_prob_out_idx = 0;
    for (int i = 1; i < m_info->input_num; i++) m_cache_in.push_back(i);
    for (int i = 1; i < m_info->output_num; i++) m_cache_out.push_back(i);
    /* 静态 shape：特征输入 [1, T, bins] 或 [1, bins, T] */
    auto &sh = m_info->stages[0].input_shapes[0];
    if (sh.num_dims >= 3) {
        m_max_frames = (int)sh.dims[1];
        m_bins = (int)sh.dims[2];
    } else if (sh.num_dims == 2) {
        m_max_frames = (int)sh.dims[0];
        m_bins = (int)sh.dims[1];
    }
    if (m_max_frames <= 0) return E_AL_NET_NOT_MATCH;

    /* fbank 前处理 */
    FbankConfig fc;
    if (vad_cfg) {
        auto fb = vad_cfg->get("fbank");
        if (fb && fb->isObj()) {
            fc.num_mel_bins = (int)fb->getNum("num_mel_bins", 80);
            fc.frame_ms = (float)fb->getNum("frame_ms", 25);
            fc.shift_ms = (float)fb->getNum("shift_ms", 10);
        }
    }
    if (fc.num_mel_bins != m_bins)
        mng()->log("warn: vad fbank bins " + std::to_string(fc.num_mel_bins) + " != net input " + std::to_string(m_bins));
    m_fbank.reset(new FbankComputer(fc));

    /* CMVN（npy，路径可配可缺省） */
    if (vad_cfg) {
        std::string mp = vad_cfg->getStr("cmvn_mean"), sp = vad_cfg->getStr("cmvn_std");
        int r, c;
        if (!mp.empty() && readNpyF32(mp, m_cmvn_mean, r, c)) {
            if (!sp.empty()) readNpyF32(sp, m_cmvn_std, r, c);
            if ((int)m_cmvn_std.size() != (int)m_cmvn_mean.size()) m_cmvn_std.assign(m_cmvn_mean.size(), 1.0f);
            m_has_cmvn = ((int)m_cmvn_mean.size() == fc.num_mel_bins);
            if (!m_has_cmvn) mng()->log("warn: cmvn dim mismatch, disabled");
        }
    }

    resetCache();
    softReset();
    mng()->log("vad ready: net=" + m_net + " frames=" + std::to_string(m_max_frames) +
               " bins=" + std::to_string(m_bins) + " chunk_ms=" + std::to_string(m_chunk_ms));
    return E_AL_OK;
}

void CVadLink::deinitMem() {
    /* stages mems 由 runtime 管理，无需手动释放 */
}

void CVadLink::resetCache() {
    if (!m_info) return;
    bm_handle_t h = (bm_handle_t)bmrt_get_bm_handle(mng()->hub().model(m_model)->bmrt);
    int v = 0;
    for (int idx : m_cache_in)
        bm_memset_device_ext(h, &v, 1, m_info->stages[0].input_mems[idx]);
}

void CVadLink::softReset() {
    /* 段状态机软复位：复位确认/计时/累积状态，不动设备 cache（resetCache 单独调）
     * 与时间轴 m_stream_ms（转写时间戳连续性不受推理失败影响） */
    m_in_speech = false;
    m_speech_beg_ms = -1;
    m_last_voice_ms = -1;
    m_sil_frames = 0;
    m_speech_frames = 0;
    m_onset_frame = -1;
    m_onset_chunk_beg = 0;
    m_seg_pcm.clear();
    m_preroll.clear();
}

int CVadLink::stop() {
    flush();
    return E_AL_OK;
}

int CVadLink::ctrl(const std::string &cmd, AL_PARAM param) {
    if (cmd == "flush") return flush();

    /*---- 专用配置命令（数值统一 double 协议：AL_PARAM{&dVal, sizeof(double)}）----
     * 校验失败 setLastError（含期望范围与实际值）+ E_AL_PARAM_ERR，成员不变；
     * 生效后写回内存配置树（失败仅告警，运行期修改仍有效，只是不随 save_params 持久化） */
    if (cmd == "set_min_silence_ms" || cmd == "set_min_speech_ms") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("vad." + cmd + ": expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        if ((!std::isfinite(dVal)) || (dVal < 1) || (dVal != std::floor(dVal)))
            return mng()->setLastError("vad." + cmd + ": expect int >= 1, got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        if (cmd == "set_min_silence_ms") m_min_sil_ms = (int)dVal;
        else m_min_speech_ms = (int)dVal;
        bool saved = mng()->UpdateCfgNum("vad",
                                         cmd == "set_min_silence_ms" ? "min_silence_ms" : "min_speech_ms",
                                         dVal);
        if (!saved) mng()->logWarn("vad." + cmd + ": cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "set_noise_thres") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("vad.set_noise_thres: expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        if ((!std::isfinite(dVal)) || (dVal < 0) || (dVal > 1))
            return mng()->setLastError("vad.set_noise_thres: expect [0,1], got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        m_noise_thres = (float)dVal;
        if (!mng()->UpdateCfgNum("vad", "speech_noise_thres", dVal))
            mng()->logWarn("vad.set_noise_thres: cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "set_max_seg_s") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("vad.set_max_seg_s: expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        /* 上界 2147s：防 (int)(dVal*1000) 溢出（2^31/1000） */
        if ((!std::isfinite(dVal)) || (dVal < 1) || (dVal > 2147) || (dVal != std::floor(dVal)))
            return mng()->setLastError("vad.set_max_seg_s: expect int in [1,2147], got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        m_max_seg_ms = (int)(dVal * 1000);
        if (!mng()->UpdateCfgNum("vad", "max_seg_s", dVal))
            mng()->logWarn("vad.set_max_seg_s: cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "set_gate_enable") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("vad.set_gate_enable: expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        if ((!std::isfinite(dVal)) || ((0 != dVal) && (1 != dVal)))
            return mng()->setLastError("vad.set_gate_enable: expect 0/1, got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        bGateEnable = (1 == dVal);
        if (bGateEnable) iGateHoldLeftMs = 0;   /* 重新武装：清挂起期 */
        if (!mng()->UpdateCfgNum("vad", "energy_gate", "enable", dVal))
            mng()->logWarn("vad.set_gate_enable: cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "set_gate_rms" || cmd == "set_gate_hold_ms") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("vad." + cmd + ": expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        if ((!std::isfinite(dVal)) || (dVal < 1) || (dVal != std::floor(dVal)))
            return mng()->setLastError("vad." + cmd + ": expect int > 0, got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        if (cmd == "set_gate_rms") iGateRms = (int)dVal;
        else iGateHoldMs = (int)dVal;
        bool saved = (cmd == "set_gate_rms")
                         ? mng()->UpdateCfgNum("vad", "energy_gate", "rms_thres", dVal)
                         : mng()->UpdateCfgNum("vad", "energy_gate", "hold_ms", dVal);
        if (!saved) mng()->logWarn("vad." + cmd + ": cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "get_params") {
        if (!param.data || param.size <= 0)
            return mng()->setLastError("vad.get_params: empty param"), E_AL_PARAM_ERR;
        snprintf((char *)param.data, param.size,
                 "{\"min_silence_ms\":%d,\"min_speech_ms\":%d,\"speech_noise_thres\":%.4f,"
                 "\"max_seg_s\":%d,\"gate_enable\":%d,\"gate_rms_thres\":%d,\"gate_hold_ms\":%d}",
                 m_min_sil_ms, m_min_speech_ms, (double)m_noise_thres,
                 m_max_seg_ms / 1000, bGateEnable ? 1 : 0, iGateRms.load(), iGateHoldMs.load());
        return E_AL_OK;
    }

    (void)param;
    return E_AL_PARAM_ERR;
}

int CVadLink::flush() {
    m_flush_req = true;
    return E_AL_OK;
}

int CVadLink::tick() {
    /* flush 优先处理：PCM 断流后 Ctrl "flush" 也能收尾 */
    if (m_flush_req.exchange(false) && m_in_speech) {
        int64_t end_ms = m_last_voice_ms > 0 ? m_last_voice_ms + m_padding_ms : m_stream_ms;
        emitSegment(m_speech_beg_ms, end_ms, true);
        m_in_speech = false;
        m_speech_beg_ms = -1;     /* 复位段起点，避免下段沿用旧值 */
        m_speech_frames = 0;      /* 复位累计，避免下段提前确认 */
        m_onset_frame = -1;
    }

    FramePtr chunk;
    int rc = pullSource(chunk);
    if (rc != E_AL_OK) return rc;

    int64_t chunk_ms = (int64_t)chunk->pcm.size() * 1000 / 16000;
    int64_t chunk_beg = m_stream_ms;

    /*---- 能量前置门（默认关闭，JSON vad.energy_gate.enable 开启）----
     * 非语音 + 低 RMS 块跳过 VAD 推理与 fbank（省 TPU/功耗）；块级判定，
     * 不丢音频：一旦本块 RMS 超阈立即恢复全速（preroll 已维护近 300ms 上下文）。
     * 说话中不门控；恢复语音后 hold_ms 内仍全速（让状态机正常收尾上段）。 */
    bool gate_low = false;
    if (bGateEnable && !m_in_speech) {
        double rms = 0;
        for (int s : chunk->pcm) rms += (double)s * s;
        rms = chunk->pcm.empty() ? 0 : sqrt(rms / chunk->pcm.size());
        gate_low = rms < (double)iGateRms;
        if (gate_low && iGateHoldLeftMs <= 0) {
            /* 低能量静音块：跳过推理，仅维护预卷缓冲与时间轴 */
            m_preroll.insert(m_preroll.end(), chunk->pcm.begin(), chunk->pcm.end());
            int preroll_samples = 16000 * m_padding_ms / 1000;
            if ((int)m_preroll.size() > preroll_samples)
                m_preroll.erase(m_preroll.begin(), m_preroll.end() - preroll_samples);
            m_stream_ms += chunk_ms;
            return E_AL_OK;
        }
        if (!gate_low) iGateHoldLeftMs.store(iGateHoldMs.load());   /* 高能量块：武装挂起期 */
    }

    /* 前处理：去直流 → fbank → CMVN */
    std::vector<int16_t> pcm = chunk->pcm;
    FbankComputer::removeDc(pcm);
    std::vector<float> feat;
    int rows, cols;
    m_fbank->compute(pcm, feat, rows, cols);
    if (rows <= 0) { m_stream_ms += chunk_ms; return E_AL_OK; }
    if (m_has_cmvn) FbankComputer::applyCmvn(feat, cols, m_cmvn_mean, m_cmvn_std);

    /* 推理（launch 内部校验概率输出 shape，shape 不符返回 E_AL_NET_NOT_MATCH 跳过本块）。
     * 失败自愈（P1②）：E_AL_LAUNCH_FAIL 时复位设备 cache 与段状态后重试一次
     * （cache 错位是 launch 失败常见源）；E_AL_NET_NOT_MATCH 为 shape 类错误，
     * 重试无意义，直接计入失败连击。连续失败仅计数上报（log），连续 3 次升级
     * logErr；任何一次成功（含自愈重试）清零连击。 */
    std::vector<float> probs;
    int lrc = launch(feat, rows, probs);
    if (E_AL_OK != lrc) {
        if (E_AL_LAUNCH_FAIL == lrc) {
            /* 说话中自愈：先把已累积段按 final 下发（保全已收音频），再复位重试 */
            if (m_in_speech) {
                int64_t cut_ms = m_last_voice_ms > 0 ? m_last_voice_ms + m_padding_ms : m_stream_ms;
                emitSegment(m_speech_beg_ms, cut_ms, true);
                m_in_speech = false;
                m_speech_beg_ms = -1;
                m_speech_frames = 0;
                m_onset_frame = -1;
            }
            resetCache();
            softReset();
            lrc = launch(feat, rows, probs);
            if (E_AL_OK == lrc) {
                mng()->log("vad self-heal: launch recovered after cache/seg reset");
                mng()->IncSelfHeal();
            }
        }
        if (E_AL_OK != lrc) {
            iFailStreak++;
            std::string msg = "vad launch failed rc=" + std::to_string(lrc) +
                              ", fail_streak=" + std::to_string(iFailStreak);
            /* 变沿日志：streak==1 首败 WARN、==3 升级 logErr，其后静默防刷屏 */
            if (1 == iFailStreak) mng()->logWarn(msg);
            else if (3 == iFailStreak) mng()->logErr(msg + " (persistent)");
            m_stream_ms += chunk_ms;
            return lrc;
        }
    }
    iFailStreak = 0;

    /* 帧级状态机 + 本 tick 音频维护（feedState 内部按块处理，见函数注释） */
    feedState(probs, rows, chunk_beg, chunk->pcm);

    /* 挂起期倒计时：低能量块（hold>0 未跳过）逐渐放行至门控生效 */
    if (bGateEnable && !m_in_speech && gate_low && iGateHoldLeftMs > 0)
        iGateHoldLeftMs -= (int)chunk_ms;

    m_stream_ms += chunk_ms;
    return E_AL_OK;
}

int CVadLink::launch(const std::vector<float> &feat, int frames, std::vector<float> &probs) {
    bm_handle_t h = (bm_handle_t)bmrt_get_bm_handle(mng()->hub().model(m_model)->bmrt);
    void *bmrt = mng()->hub().model(m_model)->bmrt;

    /* 概率输出 shape 校验（launch 前判，shape 不符本块跳过推理）：
     * 按 output_shapes[m_prob_out_idx] 推 T_out 与通道数，
     * 约定 [T] / [T,2] / [1,T] / [1,T,2]（末维为 1/2 视作通道）。
     * fbank rows != T_out 时拒绝，避免按错误 stride 读垃圾概率 */
    auto &psh = m_info->stages[0].output_shapes[m_prob_out_idx];
    int t_out = 1, channels = 1;
    {
        int last = (psh.num_dims >= 1) ? (int)psh.dims[psh.num_dims - 1] : 0;
        bool last_is_ch = (psh.num_dims >= 2) && (last == 1 || last == 2);
        channels = last_is_ch ? last : 1;
        int nd = psh.num_dims - (last_is_ch ? 1 : 0);
        for (int d = 0; d < nd; d++) t_out *= (int)psh.dims[d];
    }
    if (frames != t_out) {
        mng()->log("vad fbank rows " + std::to_string(frames) + " != prob output T " +
                   std::to_string(t_out) + ", skip chunk (net=" + m_net +
                   ", check vad.chunk_ms vs static T)");
        return E_AL_NET_NOT_MATCH;
    }
    (void)channels;   /* 通道数由 feedState 按 probs.size()/frames 自适应 */

    /* 特征写入 inputs[0]（静态 shape，整块清零后写实际长度） */
    auto &in0 = m_info->stages[0].input_mems[m_feat_in_idx];
    size_t in_bytes = bm_mem_get_device_size(in0);
    size_t real_bytes = (size_t)frames * m_bins * sizeof(float);
    if (real_bytes > in_bytes) return E_AL_NET_NOT_MATCH;
    int zero = 0;
    bm_memset_device_ext(h, &zero, 1, in0);
    if (bm_memcpy_s2d_partial(h, in0, (void *)feat.data(), real_bytes) != BM_SUCCESS)
        return E_AL_LAUNCH_FAIL;

    /* launch（stages mems 直用，chat.cpp 惯例） */
    std::vector<bm_tensor_t> ins(m_info->input_num), outs(m_info->output_num);
    for (int i = 0; i < m_info->input_num; i++)
        bmrt_tensor_with_device(&ins[i], m_info->stages[0].input_mems[i], m_info->input_dtypes[i],
                                m_info->stages[0].input_shapes[i]);
    for (int i = 0; i < m_info->output_num; i++)
        bmrt_tensor_with_device(&outs[i], m_info->stages[0].output_mems[i], m_info->output_dtypes[i],
                                m_info->stages[0].output_shapes[i]);
    if (!bmrt_launch_tensor_ex(bmrt, m_net.c_str(), ins.data(), m_info->input_num,
                               outs.data(), m_info->output_num, true, false))
        return E_AL_LAUNCH_FAIL;
    bm_thread_sync(h);

    /* cache 回写：out_cache → in_cache（d2d，LLM block_cache 模式） */
    for (size_t i = 0; i < m_cache_out.size() && i < m_cache_in.size(); i++) {
        auto &src = m_info->stages[0].output_mems[m_cache_out[i]];
        auto &dst = m_info->stages[0].input_mems[m_cache_in[i]];
        size_t bytes = std::min(bm_mem_get_device_size(src), bm_mem_get_device_size(dst));
        bm_memcpy_d2d_byte(h, dst, 0, src, 0, bytes);
    }

    /* 读概率 outputs[0]（须为 float32，否则记日志并降级返回） */
    auto &pout = m_info->stages[0].output_mems[m_prob_out_idx];
    if (m_info->output_dtypes[m_prob_out_idx] != BM_FLOAT32) {
        mng()->log("vad prob output dtype not float32, skip (net=" + m_net + ")");
        return E_AL_NET_NOT_MATCH;
    }
    size_t pbytes = bm_mem_get_device_size(pout);
    probs.resize(pbytes / sizeof(float));
    if (bm_memcpy_d2s(h, (void *)probs.data(), pout) != BM_SUCCESS) return E_AL_LAUNCH_FAIL;
    return E_AL_OK;
}

void CVadLink::feedState(const std::vector<float> &probs, int frames, int64_t chunk_beg,
                         const std::vector<int16_t> &chunk_pcm) {
    int per_frame = (int)probs.size() / (frames > 0 ? frames : 1);
    if (per_frame <= 0) per_frame = 1;
    /* 真实帧间隔 = fbank 帧移（不是 chunk 均分）：
     * frame_samples = 帧移样本数（默认 160），frame_ms = 帧移毫秒（默认 10ms）。
     * 帧号 × 帧移即帧中心相对块起点的样本/时间偏移，分钟级无累计漂移 */
    int frame_samples = m_fbank->frameShift();
    int frame_ms = frame_samples * 1000 / 16000;
    if (frame_ms <= 0) frame_ms = 1;

    int preroll_samples = 16000 * m_padding_ms / 1000;
    int chunk_samples = (int)chunk_pcm.size();

    /* 本 tick 是否确认进入语音（onset 记录原始语音起点帧，音频维护在帧循环后按块做一次） */
    bool entered_speech = false;

    for (int f = 0; f < frames; f++) {
        float p_speech;
        if (per_frame >= 2) p_speech = probs[(size_t)f * per_frame + 1]; /* [T,2] 取 [:,1] */
        else p_speech = probs[f];                                        /* [T] 直读 */
        float p_sil = (per_frame >= 2) ? probs[(size_t)f * per_frame] : 1.0f - p_speech;
        float p = p_speech - m_noise_thres * p_sil; /* funasr speech_noise_thres 语义 */
        int64_t t_ms = chunk_beg + (int64_t)f * frame_ms;

        if (!m_in_speech) {
            if (p > 0) {
                m_speech_frames += frame_ms;
                if (m_speech_beg_ms < 0) {
                    m_speech_beg_ms = t_ms;
                    m_onset_frame = f;              /* 记录原始语音起点（非确认点） */
                    m_onset_chunk_beg = chunk_beg;
                }
                if (m_speech_frames >= m_min_speech_ms) {
                    m_in_speech = true;
                    entered_speech = true;
                    m_last_voice_ms = t_ms;
                    m_sil_frames = 0;
                    m_speech_beg_ms = std::max<int64_t>(0, m_speech_beg_ms - m_padding_ms); /* 段起点含前 padding */
                }
            } else {
                m_speech_frames = 0;
                m_speech_beg_ms = -1;
                m_onset_frame = -1;
            }
        } else {
            /* 强制切段上限（vad.max_seg_s）：连续噪音/音乐下防 m_seg_pcm 无界增长。
             * 截断段按 is_final 下发（log 标记），cache 由 emitSegment 复位 */
            if (t_ms - m_speech_beg_ms >= m_max_seg_ms) {
                mng()->log("vad max_seg cut at " + std::to_string(t_ms) + " ms");
                emitSegment(m_speech_beg_ms, t_ms, true);
                m_in_speech = false;
                m_speech_beg_ms = -1;
                m_speech_frames = 0;
                m_onset_frame = -1;
                continue;
            }
            if (p > 0) { m_last_voice_ms = t_ms; m_sil_frames = 0; }
            else {
                m_sil_frames += frame_ms;
                if (m_sil_frames >= m_min_sil_ms) {
                    int64_t end_ms = m_last_voice_ms + m_padding_ms;
                    emitSegment(m_speech_beg_ms, end_ms, false);
                    m_in_speech = false;
                    m_speech_beg_ms = -1;
                    m_speech_frames = 0;
                    m_seg_pcm.clear();
                }
            }
        }
    }

    /* ---- 本 tick 音频维护（每块一次，避免帧循环内重复整块插入） ---- */
    if (m_in_speech) {
        if (entered_speech) {
            /* 确认进入语音：先并入预卷缓冲，再从"原始语音起点"截入当前块。
             * onset 可能在本块（m_onset_chunk_beg==chunk_beg，从 onset 帧偏移截入）
             * 或更早的块（off=0，先前块剩余已由 preroll→seg_pcm 补齐） */
            m_seg_pcm.insert(m_seg_pcm.end(), m_preroll.begin(), m_preroll.end());
            m_preroll.clear();
            int off = (m_onset_chunk_beg == chunk_beg && m_onset_frame > 0)
                          ? m_onset_frame * frame_samples
                          : 0;
            if (off < 0) off = 0;
            if (off > chunk_samples) off = chunk_samples;
            m_seg_pcm.insert(m_seg_pcm.end(), chunk_pcm.begin() + off, chunk_pcm.end());
            m_onset_frame = -1;
        } else {
            m_seg_pcm.insert(m_seg_pcm.end(), chunk_pcm.begin(), chunk_pcm.end());
        }
    } else {
        /* 非语音：预卷缓冲每 tick 追加/裁剪一次，维持 padding 长度 */
        m_preroll.insert(m_preroll.end(), chunk_pcm.begin(), chunk_pcm.end());
        if ((int)m_preroll.size() > preroll_samples)
            m_preroll.erase(m_preroll.begin(), m_preroll.end() - preroll_samples);
    }
}

void CVadLink::emitSegment(int64_t beg_ms, int64_t end_ms, bool final_seg) {
    FramePtr seg = acquireFrame();
    if (!seg) { mng()->log("segment dropped: pool full"); return; }
    seg->start_ms = (int)beg_ms;
    seg->end_ms = (int)end_ms;
    seg->is_final = final_seg;
    seg->pcm = m_seg_pcm;
    size_t n = m_seg_pcm.size();
    m_seg_pcm.clear();
    mng()->pushSegment(std::move(seg));
    /* 段结束：cache 归零（funasr 语义 cache={}） */
    resetCache();
    mng()->log("vad segment [" + std::to_string(beg_ms) + "," + std::to_string(end_ms) +
               "] ms, " + std::to_string(n) + " samples" + (final_seg ? " (final)" : ""));
}

} /* namespace al */
