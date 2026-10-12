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
    /* 端点状态机参数：与官方 config.yaml 的 model_conf 同名同默认
     * （见 vad_link.h 顶部注释的模型基准） */
    FsmnVadSegmenter::Config sc;
    if (vad_cfg) {
        m_chunk_ms = (int)vad_cfg->getNum("chunk_ms", 200);
        sc.window_size_ms = (int)vad_cfg->getNum("window_size_ms", 200);
        sc.sil_to_speech_ms = (int)vad_cfg->getNum("sil_to_speech_time_ms", 150);
        sc.speech_to_sil_ms = (int)vad_cfg->getNum("speech_to_sil_time_ms", 150);
        sc.max_end_silence_ms = (int)vad_cfg->getNum("max_end_silence_ms", 800);
        sc.max_start_silence_ms = (int)vad_cfg->getNum("max_start_silence_ms", 3000);
        sc.do_extend = (0 != (int)vad_cfg->getNum("do_extend", 1));
        sc.lookback_start_ms = (int)vad_cfg->getNum("lookback_time_start_point_ms", 200);
        sc.lookahead_end_ms = (int)vad_cfg->getNum("lookahead_time_end_point_ms", 100);
        sc.detect_mode = (int)vad_cfg->getNum("detect_mode", 1);
        /* 强制切段上限：沿用 max_seg_s 键（秒），默认 20s 对齐官方 max_single_segment_time */
        double max_seg_s = vad_cfg->getNum("max_seg_s", 20.0);
        if (!(max_seg_s >= 1.0)) {
            mng()->logWarn("vad.max_seg_s < 1, fallback 20");
            max_seg_s = 20.0;   /* 下界钳制 */
        }
        if (max_seg_s > 2147.0) {
            /* 上界钳制：防 (int)(val*1000) 溢出（与 set_max_seg_s 的 [1,2147] 同源） */
            mng()->logWarn("vad.max_seg_s > 2147, fallback 20");
            max_seg_s = 20.0;
        }
        sc.max_single_seg_ms = (int)(max_seg_s * 1000.0);
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

    m_is_dynamic = m_info->is_dynamic;

    /* 全流程静态：本工程已放弃 dynamic shape。dynamic 子网（[max:...] 维）会在运行时按
     * 真实 T 启动，与"固定 T 分块 + 特征 FIFO"的设计不兼容，直接拒绝并给出重转指引。 */
    if (m_is_dynamic) {
        mng()->logErr("vad bmodel is DYNAMIC-compiled; this build requires a STATIC bmodel. "
                      "Re-convert with model_transform.py --input_shapes [[1,T,400],"
                      "[1,128,19,1]x4] (NO --dynamic), see "
                      "modelSource/speech_fsmn_vad_zh-cn-16k-common-onnx/test.sh");
        return E_AL_NET_NOT_MATCH;
    }

    /* 特征输入 shape：[1, T, D]（静态；T 即固定帧数）；兼容 [1, D, T] / [T, D] */
    auto &sh = m_info->stages[0].input_shapes[0];
    if (sh.num_dims >= 3) {
        m_T = (int)sh.dims[1];
        m_feat_dim = (int)sh.dims[2];
    } else if (sh.num_dims == 2) {
        m_T = (int)sh.dims[0];
        m_feat_dim = (int)sh.dims[1];
    }
    if (m_T <= 0 || m_feat_dim <= 0) return E_AL_NET_NOT_MATCH;

    /* 概率输出 shape：[1, T, C] / [T, C] / [T]；末维即 PDF 类数（fsmn_vad 为 248） */
    {
        auto &psh = m_info->stages[0].output_shapes[m_prob_out_idx];
        if (psh.num_dims >= 2) m_num_pdf = (int)psh.dims[psh.num_dims - 1];
        if (m_num_pdf <= 0) m_num_pdf = 1;
    }

    /* 前处理参数：fbank(bins) → LFR(→ m_feat_dim) → CMVN */
    FbankConfig fc;
    if (vad_cfg) {
        auto fb = vad_cfg->get("fbank");
        if (fb && fb->isObj()) {
            fc.num_mel_bins = (int)fb->getNum("num_mel_bins", 80);
            fc.frame_ms = (float)fb->getNum("frame_ms", 25);
            fc.shift_ms = (float)fb->getNum("shift_ms", 10);
        }
        m_lfr_m = (int)vad_cfg->getNum("lfr_m", 5);
        m_lfr_n = (int)vad_cfg->getNum("lfr_n", 1);
        m_sil_pdf_id = (int)vad_cfg->getNum("sil_pdf_id", 0);
        m_speech_2_noise_ratio = (float)vad_cfg->getNum("speech_2_noise_ratio", 1.0);
    }
    if (m_lfr_m < 1) m_lfr_m = 1;
    if (m_lfr_n < 1) m_lfr_n = 1;
    if (m_sil_pdf_id < 0 || m_sil_pdf_id >= m_num_pdf) {
        mng()->logWarn("vad sil_pdf_id " + std::to_string(m_sil_pdf_id) +
                       " out of range [0," + std::to_string(m_num_pdf) + "), fallback 0");
        m_sil_pdf_id = 0;
    }
    if (m_lfr_m * fc.num_mel_bins != m_feat_dim) {
        /* 兼容无 LFR 的 80 维子网；否则拒绝（避免按错误 stride 喂数据） */
        if (fc.num_mel_bins == m_feat_dim) {
            mng()->log("vad net input " + std::to_string(m_feat_dim) +
                       " == fbank bins, disable LFR (lfr_m=1)");
            m_lfr_m = 1;
        } else {
            mng()->log("vad lfr dim " + std::to_string(m_lfr_m) + "*" +
                       std::to_string(fc.num_mel_bins) + " != net input " +
                       std::to_string(m_feat_dim));
            return E_AL_NET_NOT_MATCH;
        }
    }
    /* 流式前端：fbank(kaldi 语义) → LFR(80→400，左补首帧 + 尾部延迟) → CMVN */
    m_front.init(fc, m_lfr_m, m_lfr_n);

    /* 端点状态机初始化：帧移/帧长由前端与 fbank 配置推出后交给 FsmnVadSegmenter
     * （ms 参数 → 帧数的折算全部在 segmenter 内完成，基准 frame_in_ms=10） */
    sc.frame_samples = m_front.frameShift() > 0 ? m_front.frameShift() : 160;
    sc.frame_ms = sc.frame_samples * 1000 / 16000;
    if (sc.frame_ms <= 0) sc.frame_ms = 10;
    {
        int frame_len_samples = (int)(fc.frame_ms * 16000.0f / 1000.0f + 0.5f);
        sc.extra_samples = frame_len_samples - sc.frame_samples;  /* 段尾补 frame_length-frame_shift */
        if (sc.extra_samples < 0) sc.extra_samples = 0;
    }
    m_seg.init(sc);

    /* CMVN：优先 am.mvn（kaldi 文本 CMVN，(x+shift)*scale）；缺省回退 npy mean/std
     * 资产路径（cmvn_file/cmvn_mean/cmvn_std）相对启动 JSON 所在目录解析（resolveAsset） */
    if (vad_cfg) {
        std::vector<float> shift, scale;
        std::string mvn = mng()->resolveAsset(vad_cfg->getStr("cmvn_file"));
        bool ok = false;
        if (!mvn.empty()) {
            ok = readKaldiCmvn(mvn, shift, scale) && ((int)shift.size() == m_feat_dim);
            if (!ok) mng()->logWarn("vad cmvn_file '" + mvn + "' parse/dim failed, try npy");
        }
        if (!ok) {
            std::string mp = mng()->resolveAsset(vad_cfg->getStr("cmvn_mean"));
            std::string sp = mng()->resolveAsset(vad_cfg->getStr("cmvn_std"));
            std::vector<float> mean, stdv;
            int r, c;
            if (!mp.empty() && readNpyF32(mp, mean, r, c)) {
                if (!sp.empty()) readNpyF32(sp, stdv, r, c);
                if ((int)stdv.size() != (int)mean.size()) stdv.assign(mean.size(), 1.0f);
                if ((int)mean.size() == m_feat_dim) {
                    shift.resize(mean.size());
                    scale.resize(mean.size());
                    for (size_t i = 0; i < mean.size(); i++) {
                        shift[i] = -mean[i];
                        scale[i] = (stdv[i] > 1e-6f) ? (1.0f / stdv[i]) : 1.0f;
                    }
                    ok = true;
                }
            }
        }
        m_has_cmvn = ok && m_front.setCmvn(shift, scale);
        if (!m_has_cmvn) mng()->logWarn("vad cmvn unavailable, running WITHOUT cmvn");
    }

    resetCache();
    softReset();
    /* T 与 chunk_ms 的耦合（docs/README.md 约定 T = chunk_ms/10，200ms → 20 帧）：
     * 不一致不会出错（FIFO 会跨块凑组），但稳态会多攒一段延迟，故告警提示。 */
    {
        int expect_T = m_chunk_ms / 10;
        if (expect_T > 0 && m_T != expect_T)
            mng()->logWarn("vad static T=" + std::to_string(m_T) + " != chunk_ms/10=" +
                           std::to_string(expect_T) + "; FIFO bridges chunks (extra latency). "
                           "Re-convert bmodel with --input_shapes [[1," +
                           std::to_string(expect_T) + ",400],...] to match.");
    }
    mng()->log("vad ready: net=" + m_net + " T=" + std::to_string(m_T) +
               " static=1 dim=" + std::to_string(m_feat_dim) + " pdf=" + std::to_string(m_num_pdf) +
               " lfr=" + std::to_string(m_lfr_m) + "/" + std::to_string(m_lfr_n) +
               " cmvn=" + std::to_string(m_has_cmvn ? 1 : 0) +
               " chunk_ms=" + std::to_string(m_chunk_ms) +
               " seg[win=" + std::to_string(m_seg.cfg().window_size_ms) +
               " sil2sp=" + std::to_string(m_seg.cfg().sil_to_speech_ms) +
               " sp2sil=" + std::to_string(m_seg.cfg().speech_to_sil_ms) +
               " maxEndSil=" + std::to_string(m_seg.cfg().max_end_silence_ms) +
               " maxSeg=" + std::to_string(m_seg.cfg().max_single_seg_ms) +
               " extend=" + std::to_string(m_seg.cfg().do_extend ? 1 : 0) +
               " mode=" + std::to_string(m_seg.cfg().detect_mode) + "]");
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
    /* 段状态机软复位：复位 funasr 端点状态机与滚动 PCM 缓冲；
     * 不动设备 cache（resetCache 单独调）、不动时间轴 m_stream_ms、
     * 也不动前处理（fbank/LFR）连续态——前处理跨块累积，复位会撕裂帧栅格。
     * 特征 FIFO 一并清空（未 launch 的行随流重置作废） */
    m_seg.resetAll();
    clearFifo();
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
    if (cmd == "set_max_end_silence_ms" || cmd == "set_sil_to_speech_time_ms") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("vad." + cmd + ": expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        if ((!std::isfinite(dVal)) || (dVal < 1) || (dVal != std::floor(dVal)))
            return mng()->setLastError("vad." + cmd + ": expect int >= 1, got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        const char *key = NULL;
        if (cmd == "set_max_end_silence_ms") {
            /* 判停阈值 = max_end_silence_time - speech_to_sil_time（funasr Stats 口径） */
            m_seg.setMaxEndSilenceMs((int)dVal);
            key = "max_end_silence_ms";
        } else {
            m_seg.setSilToSpeechMs((int)dVal);
            key = "sil_to_speech_time_ms";
        }
        if (!mng()->UpdateCfgNum("vad", key, dVal))
            mng()->logWarn("vad." + cmd + ": cfg writeback failed, not persisted");
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
        m_seg.setMaxSingleSegMs((int)(dVal * 1000.0));
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
                 "{\"max_end_silence_ms\":%d,\"sil_to_speech_time_ms\":%d,"
                 "\"speech_to_sil_time_ms\":%d,\"max_single_segment_ms\":%d,"
                 "\"max_seg_s\":%d,\"speech_noise_thres\":%.4f,"
                 "\"gate_enable\":%d,\"gate_rms_thres\":%d,\"gate_hold_ms\":%d}",
                 m_seg.cfg().max_end_silence_ms, m_seg.cfg().sil_to_speech_ms,
                 m_seg.cfg().speech_to_sil_ms, m_seg.cfg().max_single_seg_ms,
                 m_seg.cfg().max_single_seg_ms / 1000, (double)m_noise_thres,
                 bGateEnable ? 1 : 0, iGateRms.load(), iGateHoldMs.load());
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
    /* flush 优先处理：① 先把环内已凑满的块抽干（否则尾段音频随 tick 退出被丢弃）；
     * ② 再补 LFR 尾帧（补末帧）跑最后一次判决；③ 强制下发在途段（is_final） */
    if (m_flush_req.exchange(false)) {
        FramePtr c;
        while (pullSource(c) == E_AL_OK) {
            if (processChunk(c) != E_AL_OK) break;   /* 上游异常/背压：停止抽干 */
        }
        finishFlush();
        return E_AL_OK;
    }

    FramePtr chunk;
    int rc = pullSource(chunk);
    if (rc != E_AL_OK) return rc;
    return processChunk(chunk);
}

int CVadLink::processChunk(const FramePtr &chunk) {
    int64_t chunk_ms = (int64_t)chunk->pcm.size() * 1000 / 16000;
    int64_t chunk_abs_beg = m_front.totalIn();        /* 本块首样本的流内绝对下标 */
    m_stream_ms += chunk_ms;                          /* 时间轴：无论门控/空帧均前进 */

    /* PCM 即时推入状态机（滚动缓冲保持连续）。判决按 T 组补齐后再推——二者用绝对
     * 帧/样本号关联，判决滞后于 PCM 是安全的（状态机取 PCM 时数据必然已就位）。 */
    pushPcmToSeg(chunk->pcm, chunk_abs_beg);

    /*---- 能量前置门（默认关闭，JSON vad.energy_gate.enable 开启）----
     * 静态 T 语义：门控**只标注**本块各行（进入 FIFO 的 gated 标记），成组推理时
     * **整组全被跳过**才不发网络；混合组仍照常推理（此时门控退化为无操作）。
     * fbank/LFR 前处理照常推进——保证帧栅格跨块连续、时间戳不漂移。
     * 高能量块重新武装 hold_ms 挂起期，让状态机正常收尾上一段。 */
    bool gate_skip = false;
    if (bGateEnable && !inSpeech()) {
        double rms = 0;
        for (int s : chunk->pcm) rms += (double)s * s;
        rms = chunk->pcm.empty() ? 0.0 : sqrt(rms / (double)chunk->pcm.size());
        bool gate_low = rms < (double)iGateRms;
        if (gate_low && iGateHoldLeftMs > 0) {
            iGateHoldLeftMs -= (int)chunk_ms;         /* 挂起期内低能量块：仍全速推理并倒计时 */
        } else if (gate_low) {
            gate_skip = true;                         /* 低能量且挂起期已过：本块标为可跳过 */
        } else {
            iGateHoldLeftMs.store(iGateHoldMs.load());/* 高能量块：重新武装挂起期 */
        }
    }

    /* 前处理（StreamFrontend）：采样级 carry-over 的 fbank → 流式 LFR → CMVN */
    std::vector<float> feat;
    int lrows = 0;
    int64_t frame0 = m_front.framesOut();             /* 本批首帧绝对帧号（时间戳基准） */
    m_front.push(chunk->pcm, feat, lrows);

    /* 尚无完整 LFR 窗口（流起始前几帧 / 极短块）：无帧可判（PCM 已入滚动缓冲），FIFO 不动 */
    if (lrows <= 0) return E_AL_OK;

    /* 行入 FIFO；攒满 T 行即组成组推理（可能一次补齐多组，也可能跨块才凑满） */
    appendFifo(feat, lrows, frame0, gate_skip);
    drainFifo(false);
    return E_AL_OK;
}

int CVadLink::finishFlush() {
    /* ① 补 LFR 尾帧（右补末帧）入 FIFO；② drainFifo(true) 跑完整组 + 把不足 T 的尾组
     *    补零跑完，并让**最后一个真实帧**拿到 is_final —— 状态机据此按 funasr 语义
     *    （MaybeOnVoiceEndIfLastFrame）收段。
     * ③ 兜底：整条流一帧未喂（无尾帧且 FIFO 空）时末帧 final 从未置位 → 仍需按 final 收尾。 */
    std::vector<float> feat;
    int frows = 0;
    m_front.finish(feat, frows);
    if (frows > 0)
        appendFifo(feat, frows, m_front.framesOut() - frows, false);
    if (!drainFifo(true)) closeInProgress();
    return E_AL_OK;
}

int CVadLink::launchT(const std::vector<float> &feat, std::vector<float> &probs) {
    bm_handle_t h = (bm_handle_t)bmrt_get_bm_handle(mng()->hub().model(m_model)->bmrt);
    void *bmrt = mng()->hub().model(m_model)->bmrt;

    const int T = m_T;
    if (T <= 0) return E_AL_NET_NOT_MATCH;
    if ((int)feat.size() < T * m_feat_dim) {
        mng()->log("vad launchT short input: " + std::to_string(feat.size()) + " < T*dim=" +
                   std::to_string(T * m_feat_dim));
        return E_AL_NET_NOT_MATCH;
    }

    /* 特征写入 inputs[0]：**整块覆盖 T*dim 个 float**（调用方保证 FIFO 已凑满/补零到 T 行）。
     * 静态子网每次 launch 都是固定 T 行，无"尾部残留旧帧"问题。 */
    auto &in0 = m_info->stages[0].input_mems[m_feat_in_idx];
    size_t in_bytes = bm_mem_get_device_size(in0);
    size_t real_bytes = (size_t)T * m_feat_dim * sizeof(float);
    if (real_bytes > in_bytes) return E_AL_NET_NOT_MATCH;
    if (bm_memcpy_s2d_partial(h, in0, (void *)feat.data(), real_bytes) != BM_SUCCESS)
        return E_AL_LAUNCH_FAIL;

    /* 静态启动：tensor mem/shape 直接复用 stages 预分配，**不修改任何维度**
     * （dynamic 子网已在 create 阶段被拒绝；全流程固定 T，见 vad_link.h 顶部注释） */
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

    /* 读概率 outputs[0]：[1, T, C] 行优先，取全部 T*C 个 float（fsmn_vad: C=248） */
    if (m_info->output_dtypes[m_prob_out_idx] != BM_FLOAT32) {
        mng()->log("vad prob output dtype not float32, skip (net=" + m_net + ")");
        return E_AL_NET_NOT_MATCH;
    }
    auto &pout = m_info->stages[0].output_mems[m_prob_out_idx];
    size_t want = (size_t)T * m_num_pdf;
    size_t avail = bm_mem_get_device_size(pout) / sizeof(float);
    if (want > avail) return E_AL_NET_NOT_MATCH;
    probs.resize(want);
    if (bm_memcpy_d2s_partial(h, (void *)probs.data(), pout, (unsigned int)(want * sizeof(float))) !=
        BM_SUCCESS)
        return E_AL_LAUNCH_FAIL;
    return E_AL_OK;
}

/*==================== 固定 T 静态批处理：FIFO → 成组推理 ====================*/

void CVadLink::pushPcmToSeg(const std::vector<int16_t> &pcm, int64_t abs_beg) {
    /* 空判决 → m_seg.appendPcm 只把 PCM 收进滚动缓冲（等价 funasr stats.data_buf） */
    m_seg.push(pcm, abs_beg, std::vector<uint8_t>(), 0, false);
}

void CVadLink::clearFifo() {
    m_fifo.clear();
    m_fifo_gated.clear();
    m_fifo_rows = 0;
    m_fifo_frame0 = 0;
}

void CVadLink::appendFifo(const std::vector<float> &feat, int rows, int64_t frame0, bool gated) {
    const int D = m_feat_dim;
    if (rows <= 0 || D <= 0) return;
    if ((int)feat.size() < rows * D) {
        mng()->logWarn("vad fifo append short: " + std::to_string(feat.size()) + " < " +
                       std::to_string(rows * D));
        rows = (int)feat.size() / D;
        if (rows <= 0) return;
    }
    /* 帧连续性自检：FIFO 必须按绝对帧号紧邻追加，否则网络会看到"跳跃"的帧序列
     * （跨块 carry-over / 前处理异常的信号） */
    if (m_fifo_rows > 0 && frame0 != m_fifo_frame0 + m_fifo_rows)
        mng()->logWarn("vad fifo frame discontinuity: expect " +
                       std::to_string(m_fifo_frame0 + m_fifo_rows) + ", got " +
                       std::to_string(frame0));
    if (m_fifo_rows == 0) m_fifo_frame0 = frame0;
    m_fifo.insert(m_fifo.end(), feat.begin(), feat.begin() + (size_t)rows * D);
    m_fifo_gated.insert(m_fifo_gated.end(), (size_t)rows, gated ? 1u : 0u);
    m_fifo_rows += rows;
}

bool CVadLink::drainFifo(bool flush_tail) {
    const int T = m_T;
    const int D = m_feat_dim;
    if (T <= 0 || D <= 0) return false;
    bool fed_final = false;
    while (m_fifo_rows >= T) {
        /* flush 时若正好剩最后一整组，其末行即为整条流末帧 → is_final */
        const bool last_group = flush_tail && (m_fifo_rows == T);
        runGroup(T, m_fifo_frame0, last_group);
        if (last_group) fed_final = true;
        m_fifo.erase(m_fifo.begin(), m_fifo.begin() + (size_t)T * D);
        m_fifo_gated.erase(m_fifo_gated.begin(), m_fifo_gated.begin() + T);
        m_fifo_rows -= T;
        m_fifo_frame0 += T;
    }
    if (flush_tail && m_fifo_rows > 0) {
        const int rem = m_fifo_rows;
        /* 尾部补零到 T（零特征行）。因果 FSMN：第 rem..T-1 行不影响第 0..rem-1 行输出，
         * 故只取前 rem 行判决即与"按真实长度启动"等价；补零行仅用于推进一次网络。 */
        m_fifo.resize((size_t)T * D, 0.0f);
        m_fifo_gated.resize((size_t)T, 0u);
        runGroup(rem, m_fifo_frame0, /*is_final=*/true);
        fed_final = true;
        clearFifo();
    }
    return fed_final;
}

void CVadLink::runGroup(int real_rows, int64_t frame0, bool is_final) {
    const int T = m_T;
    if (real_rows < 1) real_rows = 1;
    if (real_rows > T) real_rows = T;

    /* 整组被能量门跳过 → 不发网络，直接合成"全静音"判决（帧栅格保持连续）；
     * 混合组（部分 gated）仍照常推理，此时门控退化为无操作 */
    bool all_gated = true;
    for (int i = 0; i < real_rows; i++)
        if (!m_fifo_gated[(size_t)i]) { all_gated = false; break; }
    if (all_gated) {
        pushDecisions(std::vector<uint8_t>((size_t)real_rows, 0u), frame0, is_final);
        return;
    }

    /* 推理（launchT 内部校验输出 shape）。失败自愈：E_AL_LAUNCH_FAIL 时复位设备 cache
     * 后原地重试一次（cache 错位是 launch 失败常见源）；E_AL_NET_NOT_MATCH 为 shape 类
     * 错误，重试无意义。连续失败仅计数上报，连续 3 次升级 logErr；成功即清零连击。 */
    std::vector<float> probs;
    int rc = launchT(m_fifo, probs);
    if (E_AL_LAUNCH_FAIL == rc) {
        /* 自愈只复位设备 cache，不动端点状态机 / 滚动缓冲——音频已在滚动缓冲中 */
        resetCache();
        rc = launchT(m_fifo, probs);
        if (E_AL_OK == rc) {
            mng()->log("vad self-heal: launch recovered after cache reset");
            mng()->IncSelfHeal();
        }
    }
    if (E_AL_OK != rc) {
        iFailStreak++;
        std::string msg = "vad launch failed rc=" + std::to_string(rc) +
                          ", fail_streak=" + std::to_string(iFailStreak);
        if (1 == iFailStreak) mng()->logWarn(msg);
        else if (3 == iFailStreak) mng()->logErr(msg + " (persistent)");
        /* 本组无判决：合成"全静音"喂状态机（帧栅格连续，在途段将在 max_end_silence 后收尾） */
        pushDecisions(std::vector<uint8_t>((size_t)real_rows, 0u), frame0, is_final);
        return;
    }
    iFailStreak = 0;

    /* 概率 → 逐帧判决；尾部补零行丢弃（只喂前 real_rows 行） */
    std::vector<uint8_t> dec = probsToDecisions(probs, T);
    if ((int)dec.size() > real_rows) dec.resize((size_t)real_rows);
    pushDecisions(dec, frame0, is_final);
}

std::vector<uint8_t> CVadLink::probsToDecisions(const std::vector<float> &probs, int frames) {
    /* 逐帧判决（funasr FsmnVADStreaming::GetFrameState）：
     *   P_sil = scores[t][sil_pdf_id]（网络输出已含 Softmax，故为概率）
     *   speech iff exp(speech_prob) >= exp(noise_prob) + thres
     *           即 (1 - P_sil) >= P_sil^ratio + speech_noise_thres
     * ratio=1.0、thres=0.6 时等价于 P_sil <= 0.2。 */
    std::vector<uint8_t> is_speech;
    if (frames <= 0) return is_speech;
    is_speech.reserve((size_t)frames);
    int per_frame = (int)(probs.size() / (size_t)frames);
    if (per_frame <= 0) per_frame = 1;
    for (int f = 0; f < frames; f++) {
        float p_sil, p_speech;
        if (per_frame >= 2) {
            int sid = (m_sil_pdf_id >= 0 && m_sil_pdf_id < per_frame) ? m_sil_pdf_id : 0;
            p_sil = probs[(size_t)f * per_frame + sid];
            p_speech = 1.0f - p_sil;
        } else {
            p_speech = probs[(size_t)f];          /* [T] 直读（已是语音概率） */
            p_sil = 1.0f - p_speech;
        }
        is_speech.push_back((p_speech >= std::pow(std::max(p_sil, 1e-12f),
                                                  m_speech_2_noise_ratio) + m_noise_thres)
                                ? 1u : 0u);
    }
    return is_speech;
}

void CVadLink::pushDecisions(const std::vector<uint8_t> &is_speech, int64_t frame0, bool is_final) {
    if (is_speech.empty()) return;
    /* PCM 已由 pushPcmToSeg 推入（空 pcm → 不改变滚动缓冲），此处只推判决。
     * is_final=true 时 state machine 仅把**本批末帧**置 final（funasr is_final_frame）。 */
    m_seg.push(std::vector<int16_t>(), 0, is_speech, frame0, is_final);
    drainSegments();
}

void CVadLink::drainSegments() {
    /* 段结束**不**复位设备 cache：funasr ResetDetection 只复位端点状态机与丢帧，
     * FSMN 记忆跨段保留，以免边界后 ~lorder 帧上下文缺失 */
    for (VadSegment &s : m_seg.pending()) {
        FramePtr seg = acquireFrame();
        if (!seg) { mng()->log("segment dropped: pool full"); continue; }
        seg->start_ms = (int)s.beg_ms;
        seg->end_ms = (int)s.end_ms;
        seg->is_final = s.is_final;
        size_t n = s.pcm.size();
        seg->pcm = std::move(s.pcm);
        mng()->pushSegment(std::move(seg));
        mng()->log("vad segment [" + std::to_string(s.beg_ms) + "," + std::to_string(s.end_ms) +
                   "] ms, " + std::to_string(n) + " samples" + (s.is_final ? " (final)" : ""));
    }
    m_seg.pending().clear();
}

void CVadLink::closeInProgress() {
    m_seg.closeInProgress();
    drainSegments();
}

} /* namespace al */
