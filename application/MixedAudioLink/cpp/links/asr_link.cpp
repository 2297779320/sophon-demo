//===----------------------------------------------------------------------===//
//
// asr_link.cpp - qwen3_asr 实现（LLM 公共路径见 text_llm.h）
//
//===----------------------------------------------------------------------===//
#include "asr_link.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace al {

int CAsrLink::create(const JsonValue *cfg, AL_PARAM param) {
    (void)param;
    const JsonValue *asr_cfg = mng()->section("asr");
    if (asr_cfg) {
        /* enable_history 兼容 BOOL 与 NUM 0/1
         * （专用命令写回经 UpdateCfgNum 为数值，保证 save_params 往返一致） */
        auto eh = asr_cfg->get("enable_history");
        if (eh) m_enable_history = eh->isNum() ? (0 != eh->num) : (JsonValue::BOOL == eh->type && eh->b);
        m_max_new_tokens = (int)asr_cfg->getNum("max_new_tokens", 64);
        m_tokens_per_sec = (int)asr_cfg->getNum("tokens_per_sec", 13);
        m_system_prompt = asr_cfg->getStr("system_prompt", "");
        m_language = asr_cfg->getStr("language", "");   /* null/缺省 = 自动 */
    }

    /* 模型定位：JSON models 键名约定为 "asr" */
    CModelHub &hub = mng()->hub();
    if (!hub.hasModel("asr")) { mng()->log("asr model must be named 'asr' in JSON models"); return E_AL_NET_NOT_MATCH; }
    m_model = "asr";

    /* LLM 公共路径（子网发现/KV/dev_buffer） */
    int rc = m_llm.init(hub, m_model);
    if (rc != E_AL_OK) { mng()->log("asr llm init failed"); return rc; }
    const AsrDims &d = m_llm.dims();

    /* audio 编码子网 */
    std::string audio_net = hub.netByRole(m_model, "asr_audio");
    if (audio_net.empty()) audio_net = "audio";
    if (!hub.netInfo(m_model, audio_net)) return E_AL_NET_NOT_MATCH;
    m_audio_net = audio_net;

    /* mel 前处理 */
    AsrMel::Config mc;
    mc.audio_length = d.AUDIO_LENGTH;
    if (asr_cfg) {
        mc.mel_filters_npy = asr_cfg->getStr("mel_filters", "");
        mc.feature_size = (int)asr_cfg->getNum("mel_bins", 128);
    }
    if (mc.feature_size != d.AUDIO_MEL_BINS)
        mng()->log("warn: mel bins " + std::to_string(mc.feature_size) + " != audio net " +
                   std::to_string(d.AUDIO_MEL_BINS));
    mc.feature_size = d.AUDIO_MEL_BINS;
    m_mel.init(mc);

    /* tokenizer */
    std::string vocab = asr_cfg ? asr_cfg->getStr("vocab", "config/vocab.json") : "config/vocab.json";
    std::string merges = asr_cfg ? asr_cfg->getStr("merges", "config/merges.txt") : "config/merges.txt";
    std::map<std::string, int> specials = {
        {"<|im_end|>", 151645}, {"<|im_start|>", 151644},
        {"<|audio_start|>", 151669}, {"<|audio_end|>", 151670},
        {"<|audio_pad|>", 151676}, {"<asr_text>", 151671},
    };
    if (!m_tok.load(vocab, merges, specials)) {
        mng()->log("tokenizer load failed: " + vocab + " / " + merges);
        return E_AL_MODEL_LOAD_FAIL;
    }
    m_id_im_end = m_tok.tokenToId("<|im_end|>");
    m_id_audio_bos = m_tok.tokenToId("<|audio_start|>");

    mng()->log("asr ready: layers=" + std::to_string(d.NUM_LAYERS) +
               " seqlen=" + std::to_string(d.SEQLEN) +
               " max_input=" + std::to_string(d.MAX_INPUT_LENGTH) +
               " audio_len=" + std::to_string(d.AUDIO_LENGTH));
    return E_AL_OK;
}

/*============================ Ctrl ============================*/

int CAsrLink::ctrl(const std::string &cmd, AL_PARAM param) {
    if (cmd == "set_language") {
        std::lock_guard<std::mutex> lk(mtxStr);
        m_language = (param.data && param.size > 0) ? std::string((const char *)param.data) : "";
        return E_AL_OK;
    }
    if (cmd == "clear_history") { m_llm.clearHistory(); return E_AL_OK; }
    /*---- 专用配置命令（数值统一 double 协议：AL_PARAM{&dVal, sizeof(double)}；
     * set_system_prompt 为字符串参数 AL_PARAM{str, len}，空串 = 清空提示词）----
     * 校验失败 setLastError（含期望范围与实际值）+ E_AL_PARAM_ERR，成员不变；
     * 生效后写回内存配置树（失败仅告警，运行期修改仍有效，只是不随 save_params 持久化） */
    if (cmd == "set_max_new_tokens") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("asr.set_max_new_tokens: expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        if ((!std::isfinite(dVal)) || (dVal < 1) || (dVal != std::floor(dVal)))
            return mng()->setLastError("asr.set_max_new_tokens: expect int > 0, got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        m_max_new_tokens = (int)dVal;
        if (!mng()->UpdateCfgNum("asr", "max_new_tokens", dVal))
            mng()->logWarn("asr.set_max_new_tokens: cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "set_enable_history") {
        if ((NULL == param.data) || (param.size < (int)sizeof(double)))
            return mng()->setLastError("asr.set_enable_history: expect AL_PARAM{&double, sizeof(double)}"),
                   E_AL_PARAM_ERR;
        double dVal = *(const double *)param.data;
        if ((!std::isfinite(dVal)) || ((0 != dVal) && (1 != dVal)))
            return mng()->setLastError("asr.set_enable_history: expect 0/1, got " +
                                       std::to_string(dVal)),
                   E_AL_PARAM_ERR;
        bool en = (1 == dVal);
        if (en != m_enable_history) {
            m_enable_history = en;
            ctrl("clear_history", AL_PARAM{nullptr, 0});   /* 切换时自动清 KV history */
        }
        if (!mng()->UpdateCfgNum("asr", "enable_history", dVal))
            mng()->logWarn("asr.set_enable_history: cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "set_system_prompt") {
        if ((NULL == param.data) || (param.size <= 0))
            return mng()->setLastError("asr.set_system_prompt: empty param"), E_AL_PARAM_ERR;
        std::string prompt((const char *)param.data,
                           strnlen((const char *)param.data, (size_t)param.size));
        std::lock_guard<std::mutex> lk(mtxStr);
        m_system_prompt = prompt;   /* 空串 = 清空提示词 */
        if (!mng()->UpdateCfgStr("asr", "system_prompt", prompt))
            mng()->logWarn("asr.set_system_prompt: cfg writeback failed, not persisted");
        return E_AL_OK;
    }
    if (cmd == "get_params") {
        if (param.data && param.size > 0) {
            /* system_prompt/language 转义 " \ 与 <0x20 控制符（含 \n\r\t），保证 JSON 合法 */
            std::lock_guard<std::mutex> lk(mtxStr);
            std::string sp, lg;
            for (char ch : m_system_prompt) {
                unsigned char uc = (unsigned char)ch;
                if ('"' == ch || '\\' == ch) sp += '\\';
                else if ('\n' == ch) { sp += "\\n"; continue; }
                else if ('\r' == ch) { sp += "\\r"; continue; }
                else if ('\t' == ch) { sp += "\\t"; continue; }
                else if (uc < 0x20) { char e[8]; snprintf(e, sizeof(e), "\\u%04X", uc); sp += e; continue; }
                sp += ch;
            }
            for (char ch : m_language) {
                unsigned char uc = (unsigned char)ch;
                if ('"' == ch || '\\' == ch) lg += '\\';
                else if ('\n' == ch) { lg += "\\n"; continue; }
                else if ('\r' == ch) { lg += "\\r"; continue; }
                else if ('\t' == ch) { lg += "\\t"; continue; }
                else if (uc < 0x20) { char e[8]; snprintf(e, sizeof(e), "\\u%04X", uc); lg += e; continue; }
                lg += ch;
            }
            std::string js = "{\"max_new_tokens\":" + std::to_string(m_max_new_tokens) +
                             ",\"system_prompt\":\"" + sp + "\",\"enable_history\":" +
                             (m_enable_history ? "1" : "0") + ",\"language\":\"" + lg + "\"}";
            snprintf((char *)param.data, param.size, "%s", js.c_str());
        }
        return E_AL_OK;
    }
    if (cmd == "set_partial_cb") {
        struct { AL_PARTIAL_CB cb; void *user; } *p = (decltype(p))param.data;
        if (param.size < (int)sizeof(*p) || !p) return E_AL_PARAM_ERR;
        m_partial_cb = p->cb;
        m_partial_user = p->user;
        return E_AL_OK;
    }
    if (cmd == "get_model_info") {
        if (param.data && param.size > 0) {
            const AsrDims &d = m_llm.dims();
            snprintf((char *)param.data, param.size,
                     "{\"loaded\":%s,\"layers\":%d,\"seqlen\":%d,\"max_input\":%d,\"hidden\":%d,"
                     "\"audio_length\":%d,\"tokens_per_sec\":%d,\"history\":%s}",
                     m_llm.ready() ? "true" : "false", d.NUM_LAYERS, d.SEQLEN,
                     d.MAX_INPUT_LENGTH, d.HIDDEN_SIZE, d.AUDIO_LENGTH, m_tokens_per_sec,
                     m_enable_history ? "true" : "false");
        }
        return E_AL_OK;
    }
    return E_AL_PARAM_ERR;
}

/*============================ 主流程 ============================*/

int CAsrLink::processSegment(const std::vector<float> &mel, bool is_final, std::string &out_text) {
    (void)is_final;
    out_text.clear();
    if (!m_llm.ready()) return E_AL_NO_FRAME;   /* unload_model 期间 */
    if (mel.empty()) return E_AL_NO_FRAME;

    const AsrDims &d = m_llm.dims();
    int per_chunk = d.AUDIO_MEL_BINS * d.AUDIO_LENGTH;
    int n_times = (int)(mel.size() / per_chunk);
    if (n_times <= 0) return E_AL_NO_FRAME;

    int tokens_per_chunk = tokensPerChunk();

    /* prompt → input_ids（audio_bos 后插入 n_times*tpc 个 audio_pad） */
    std::vector<int> ids;
    int audio_bos_pos = -1;
    if (buildInputIds(n_times, ids, audio_bos_pos) != E_AL_OK) return E_AL_NET_NOT_MATCH;
    if ((int)ids.size() + m_llm.historyLength() > d.SEQLEN - 128 ||
        (m_enable_history && m_llm.historyLength() > d.PREFILL_KV_LENGTH)) {
        m_llm.clearHistory();
        m_llm.setHistoryMaxPosid(0);
        if ((int)ids.size() + m_llm.historyLength() > d.SEQLEN - 128) {
            mng()->log("asr prompt too long: " + std::to_string(ids.size()));
            return E_AL_PARAM_ERR;
        }
    }
    if (!m_enable_history) { m_llm.clearHistory(); m_llm.setHistoryMaxPosid(0); }

    /* 1. forward_embed（launch 失败终止生成链） */
    int rc = m_llm.forwardEmbed(ids);
    if (rc != E_AL_OK) return rc;

    /* 2. forward_audio：逐块推理，embed 写入 dev_buffer 的 audio token 偏移 */
    std::vector<int> offsets(n_times);
    for (int i = 0; i < n_times; i++)
        offsets[i] = audio_bos_pos + 1 + i * tokens_per_chunk;
    rc = forwardAudio(mel.data(), n_times, offsets);
    if (rc != E_AL_OK) return rc;

    /* 3. mrope position_ids（文本退化为三路 arange + history 偏移） */
    int seq = (int)ids.size();
    std::vector<int> pos3((size_t)seq * 3);
    for (int i = 0; i < seq; i++)
        for (int d3 = 0; d3 < 3; d3++) pos3[(size_t)d3 * seq + i] = i + m_llm.historyMaxPosid();

    /* 4. prefill + 解码（max_posid = 末位 position_id，对齐 pipeline.py:171） */
    int token = m_llm.prefill(pos3, seq - 1 + m_llm.historyMaxPosid());
    if (token < 0) return E_AL_LAUNCH_FAIL;
    rc = m_llm.decodeLoop(token, m_tok, m_id_im_end, m_max_new_tokens,
                          m_partial_cb, m_partial_user, out_text, "<asr_text>");
    return rc;
}

/*============================ 分步实现 ============================*/

int CAsrLink::tokensPerChunk() {
    /* audio 子网每块实际输出 13 个 token：由 output mem 字节数推导
     * （out_bytes = tpc × HIDDEN_SIZE × sizeof(uint16_t)），推导失败回退 13 */
    CModelHub &hub = mng()->hub();
    const bm_net_info_t *net_audio = hub.netInfo(m_model, m_audio_net);
    const AsrDims &d = m_llm.dims();
    if (net_audio && d.HIDDEN_SIZE > 0) {
        size_t out_bytes = bm_mem_get_device_size(net_audio->stages[0].output_mems[0]);
        int tpc = (int)(out_bytes / ((size_t)d.HIDDEN_SIZE * sizeof(uint16_t)));
        if (tpc > 0) return tpc;
    }
    return 13;
}

int CAsrLink::buildInputIds(int n_times, std::vector<int> &ids, int &audio_bos_pos) {
    int tpc = tokensPerChunk();
    std::string sys, lang;
    {   /* 字符串参数快照：与 ctrl 线程写侧互斥（防 std::string 重分配 UAF） */
        std::lock_guard<std::mutex> lk(mtxStr);
        sys = m_system_prompt;
        lang = m_language;
    }
    std::string prompt = m_tok.renderPrompt(sys, n_times * tpc, lang);
    ids = m_tok.encode(prompt);
    if (ids.size() > (size_t)m_llm.dims().MAX_INPUT_LENGTH) return E_AL_PARAM_ERR;
    audio_bos_pos = -1;
    for (size_t i = 0; i < ids.size(); i++)
        if (ids[i] == m_id_audio_bos) { audio_bos_pos = (int)i; break; }
    if (audio_bos_pos < 0) return E_AL_NET_NOT_MATCH;
    return E_AL_OK;
}

int CAsrLink::forwardAudio(const float *chunks, int n_times, const std::vector<int> &offsets) {
    CModelHub &hub = mng()->hub();
    const bm_net_info_t *net_audio = hub.netInfo(m_model, m_audio_net);
    if (!net_audio || !m_llm.ready()) return E_AL_NET_NOT_MATCH;
    bm_handle_t h = m_llm.handle();
    const AsrDims &d = m_llm.dims();

    int zero = 0;
    for (int i = 0; i < net_audio->input_num; i++) /* empty_net */
        bm_memset_device_ext(h, &zero, 1, net_audio->stages[0].input_mems[i]);
    auto &in_mem = net_audio->stages[0].input_mems[0];
    auto &out_mem = net_audio->stages[0].output_mems[0];
    size_t in_bytes = bm_mem_get_device_size(in_mem);
    size_t out_bytes = bm_mem_get_device_size(out_mem);

    std::vector<bm_tensor_t> ins(net_audio->input_num), outs(net_audio->output_num);
    for (int i = 0; i < n_times; i++) {
        if (bm_memcpy_s2d_partial(h, in_mem,
                                  (void *)(chunks + (size_t)i * in_bytes / sizeof(float)),
                                  in_bytes) != BM_SUCCESS)
            return E_AL_LAUNCH_FAIL;
        for (int k = 0; k < net_audio->input_num; k++)
            bmrt_tensor_with_device(&ins[k], net_audio->stages[0].input_mems[k],
                                    net_audio->input_dtypes[k],
                                    net_audio->stages[0].input_shapes[k]);
        for (int k = 0; k < net_audio->output_num; k++)
            bmrt_tensor_with_device(&outs[k], net_audio->stages[0].output_mems[k],
                                    net_audio->output_dtypes[k],
                                    net_audio->stages[0].output_shapes[k]);
        if (!bmrt_launch_tensor_ex(hub.model(m_model)->bmrt, net_audio->name, ins.data(),
                                   net_audio->input_num, outs.data(), net_audio->output_num,
                                   true, false))
            return E_AL_LAUNCH_FAIL;
        int dst_offset = offsets[i] * d.HIDDEN_SIZE * (int)sizeof(uint16_t);
        if (bm_memcpy_d2d_byte(h, m_llm.devBuffer(), dst_offset, out_mem, 0, out_bytes) != BM_SUCCESS)
            return E_AL_LAUNCH_FAIL;
    }
    return E_AL_OK;
}

} /* namespace al */
