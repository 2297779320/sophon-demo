//===----------------------------------------------------------------------===//
//
// text_llm.h - CTextLLM：text-only LLM 推理公共类
// 从 asr_link 提取的 LLM-TPU chat.cpp 通用路径：子网发现（init_by_names 模式）、
// 维度推导、KV cache、forwardEmbed/forwardFirst/forwardNext/greedy、逐词 decode。
// 使用者：CAsrLink（叠加 audio 编码分支）、MeetingSummary 模块（纯文本生成）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_TEXT_LLM_H
#define AL_TEXT_LLM_H

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "al_log.h"
#include "audio_link.h"
#include "asr_tokenizer.h"
#include "model_hub.h"

namespace al {

class CTextLLM {
public:
    CTextLLM() = default;
    ~CTextLLM() { deinit(); }
    CTextLLM(const CTextLLM &) = delete;
    CTextLLM &operator=(const CTextLLM &) = delete;

    /** 子网发现 + KV/dev_buffer 分配（模型须已由 hub 加载） */
    int init(CModelHub &hub, const std::string &model_name) {
        auto m = hub.modelRef(model_name);   /* shared_ptr：跨并发 load/unload 不会悬垂 */
        if (!m || !m->bmrt) return E_AL_NET_NOT_MATCH;
        m_model = model_name;
        m_bmrt = m->bmrt;
        m_handle = (bm_handle_t)bmrt_get_bm_handle(m_bmrt);

        m_d = hub.asrDims();
        if (!m_d.valid) return E_AL_NET_NOT_MATCH;

        m_net_embed = hub.netInfo(m_model, "embedding");
        m_net_embed_cache = hub.netInfo(m_model, "embedding_cache");
        m_net_lm = hub.netInfo(m_model, "lm_head");
        m_net_greedy = hub.netInfo(m_model, "greedy_head");
        for (int i = 0; i < m_d.NUM_LAYERS; i++) {
            m_net_blocks.push_back(hub.netInfo(m_model, "block_" + std::to_string(i)));
            m_net_blocks_cache.push_back(hub.netInfo(m_model, "block_cache_" + std::to_string(i)));
        }
        if (!m_net_embed || !m_net_embed_cache || !m_net_lm ||
            m_net_blocks.empty() || m_net_blocks_cache.empty())
            return E_AL_NET_NOT_MATCH;

        m_past_key.resize(m_d.NUM_LAYERS);
        m_past_value.resize(m_d.NUM_LAYERS);
        for (int i = 0; i < m_d.NUM_LAYERS; i++) {
            m_past_key[i] = m_net_blocks_cache[i]->stages[0].input_mems[3];
            m_past_value[i] = m_net_blocks_cache[i]->stages[0].input_mems[4];
            int v = 0;
            bm_memset_device_ext(m_handle, &v, 1, m_past_key[i]);
            bm_memset_device_ext(m_handle, &v, 1, m_past_value[i]);
        }
        /* io_alone 双重检测（对齐 LLM-TPU addr_mode/io_alone 惯例）：
         * ① 声明初判：block_cache_0 的 addr_mode==1；
         * ② 运行时实证：block_cache_0 的 KV 输入(input_mems[3]/[4]) 与 KV 输出
         *    (output_mems[1]/[2]) 设备地址一致（参照 Qwen3 chat.cpp is_same_addr 思路）。
         * io_alone=true 时 prefill/decode 的 KV 输出直接重绑写入 past 缓存，免逐层 d2d 回拷；
         * 检测结果随 reload（deinit→init）自动重检，无状态残留 */
        m_io_alone = false;
        {
            int addr_mode0 = m_net_blocks_cache[0]->addr_mode;
            for (int i = 1; i < m_d.NUM_LAYERS; i++) {
                if (addr_mode0 != m_net_blocks_cache[i]->addr_mode) {
                    AL_LOG_E("text_llm", "[%s] block_cache_%d addr_mode(%d) != block_cache_0(%d), 取首层判定\n",
                             m_model.c_str(), i, m_net_blocks_cache[i]->addr_mode, addr_mode0);
                    break;
                }
            }
            if (1 == addr_mode0) {
                /* 层间 I/O 数量一致性校验：ff_ins/ff_outs 按 block_0 数量分配，
                 * 各层不一致会导致越界写——不一致则放弃 io_alone 走安全回退 */
                for (int i = 1; i < m_d.NUM_LAYERS && m_io_alone; i++) {
                    if (m_net_blocks[i]->input_num != m_net_blocks[0]->input_num ||
                        m_net_blocks[i]->output_num != m_net_blocks[0]->output_num ||
                        m_net_blocks_cache[i]->input_num != m_net_blocks_cache[0]->input_num ||
                        m_net_blocks_cache[i]->output_num != m_net_blocks_cache[0]->output_num) {
                        AL_LOG_E("text_llm", "[%s] block_%d io num != block_0, disable io_alone\n",
                                 m_model.c_str(), i);
                        m_io_alone = false;
                    }
                }
                uint64_t k_in = bm_mem_get_device_addr(m_net_blocks_cache[0]->stages[0].input_mems[3]);
                uint64_t k_out = bm_mem_get_device_addr(m_net_blocks_cache[0]->stages[0].output_mems[1]);
                uint64_t v_in = bm_mem_get_device_addr(m_net_blocks_cache[0]->stages[0].input_mems[4]);
                uint64_t v_out = bm_mem_get_device_addr(m_net_blocks_cache[0]->stages[0].output_mems[2]);
                m_io_alone = m_io_alone && (k_in == k_out) && (v_in == v_out);
            }
            AL_LOG_I("text_llm", "[%s] io_alone detect: addr_mode=%d, io_alone=%d\n",
                     m_model.c_str(), addr_mode0, (int)m_io_alone);
        }
        size_t buf_bytes = bm_mem_get_device_size(m_net_embed->stages[0].output_mems[0]);
        if (BM_SUCCESS != bm_malloc_device_byte(m_handle, &m_dev_buffer, buf_bytes))
            return E_AL_INTERNAL_ERR;
        m_buffer_ok = true;
        /* decode 复用缓冲（按静态 shape 一次性分配） */
        m_next_mask.assign(m_d.SEQLEN + 1, 0);
        m_dec_ins.resize(5);
        m_dec_outs.resize(3);
        m_ready = true;
        return E_AL_OK;
    }

    /** 模型重载后重新绑定子网指针（unload 期间不得调用本类接口） */
    int reload(CModelHub &hub, const std::string &model_name) {
        deinit();
        return init(hub, model_name);
    }

    void deinit() {
        if (m_buffer_ok) { bm_free_device(m_handle, m_dev_buffer); m_buffer_ok = false; }
        m_net_embed = m_net_embed_cache = m_net_lm = m_net_greedy = nullptr;
        m_net_blocks.clear();
        m_net_blocks_cache.clear();
        m_past_key.clear();
        m_past_value.clear();
        m_next_mask.clear();
        m_dec_ins.clear();
        m_dec_outs.clear();
        m_ready = false;
        m_io_alone = false;   /* reload 后 init 重新检测 */
        m_token_length = m_history_length = m_max_posid = m_history_max_posid = 0;
    }

    bool ready() const { return m_ready; }
    const AsrDims &dims() const { return m_d; }
    const std::string &model() const { return m_model; }
    bm_handle_t handle() const { return m_handle; }
    bm_device_mem_t &devBuffer() { return m_dev_buffer; }

    void clearHistory() {
        if (!m_d.support_history) return;
        int v = 0;
        for (int i = 0; i < m_d.NUM_LAYERS; i++) {
            bm_memset_device_ext(m_handle, &v, 1, m_past_key[i]);
            bm_memset_device_ext(m_handle, &v, 1, m_past_value[i]);
        }
        m_history_length = 0;
    }

    /*=================== 一体化文本生成 ===================*/

    /** prompt ids → greedy 生成至 im_end / max_new；逐词回调（cb 可空） */
    int generate(const std::vector<int> &prompt_ids, QwenTokenizer &tok, int im_end,
                 int max_new, AL_PARTIAL_CB cb, void *user, std::string &out_text,
                 int history_pos_offset = 0) {
        out_text.clear();
        if (!m_ready) return E_AL_NET_NOT_MATCH;
        auto t0 = std::chrono::steady_clock::now();
        int rc = forwardEmbed(prompt_ids);
        if (rc != E_AL_OK) return rc;
        int seq = m_token_length;
        std::vector<int> pos3((size_t)seq * 3);
        for (int i = 0; i < seq; i++)
            for (int d3 = 0; d3 < 3; d3++) pos3[(size_t)d3 * seq + i] = i + history_pos_offset;
        m_max_posid = seq - 1 + history_pos_offset;
        int token = forwardFirst(pos3);
        auto t1 = std::chrono::steady_clock::now();
        if (token < 0) return E_AL_LAUNCH_FAIL;
        m_last_ftl_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        rc = decodeLoop(token, tok, im_end, max_new, cb, user, out_text);
        auto t2 = std::chrono::steady_clock::now();
        double decode_s = std::chrono::duration<double>(t2 - t1).count();
        m_last_tps = decode_s > 1e-6 ? m_last_tok_num / decode_s : 0.0;
        return rc;
    }

    /** asr 用：自己完成 forwardEmbed + forwardAudio 后调 prefill；
     *  max_posid = 末位 position_id（对齐 pipeline.py: max_posid = position_ids.max()），
     *  内部设置 m_max_posid 供 decodeLoop 首个 forwardNext 使用 */
    int prefill(const std::vector<int> &position_ids3, int max_posid) {
        m_max_posid = max_posid;
        int token = forwardFirst(position_ids3);
        return token; /* <0 为失败 */
    }

    /** 逐词解码循环（对齐 pipeline.py：替换符检测 + 双 token 去重叠前缀）。
     *  reset_token 非空时：解码出该词则清空 out_text 并按 "\n" 处理（asr 的 <asr_text> 语义） */
    int decodeLoop(int token, QwenTokenizer &tok, int im_end, int max_new,
                   AL_PARTIAL_CB cb, void *user, std::string &out_text,
                   const std::string &reset_token = "") {
        std::vector<int> word_tokens;
        m_last_tok_num = 0;
        for (int i = 0; i < max_new && token != im_end && m_history_length < m_d.SEQLEN; i++) {
            m_last_tok_num++;
            word_tokens.push_back(token);
            std::string word = tok.decode(word_tokens);
            if (word.find("\xEF\xBF\xBD") == std::string::npos) { /* 无替换符即完整词 */
                if (word_tokens.size() == 1 && !tok.isSpecial(tok.idToToken(token))) {
                    std::string two = tok.decode({token, token});
                    std::string pre = tok.decode({token});
                    if (two.size() > pre.size() && two.compare(0, pre.size(), pre) == 0)
                        word = two.substr(pre.size());
                }
                if (!reset_token.empty() && word == reset_token) {
                    out_text.clear();
                    word = "\n";
                }
                out_text += word;
                if (cb) cb(out_text.c_str(), user);
                word_tokens.clear();
            }
            m_max_posid++;
            int pos_next[3] = {m_max_posid, m_max_posid, m_max_posid};
            token = forwardNext(pos_next);
            if (token < 0) return E_AL_LAUNCH_FAIL;
        }
        m_history_max_posid = m_max_posid + 2;
        return E_AL_OK;
    }

    /*=================== 分步接口（asr audio 分支用） ===================*/

    /** 子网 embed：返回 E_AL_OK / 失败码（launch 失败时调用方终止生成链） */
    int forwardEmbed(const std::vector<int> &tokens) {
        std::vector<int> input_ids(m_d.MAX_INPUT_LENGTH, 0); /* 静态 shape 补齐 */
        size_t num = std::min(tokens.size(), (size_t)m_d.MAX_INPUT_LENGTH);
        std::copy(tokens.begin(), tokens.begin() + num, input_ids.begin());
        auto &in_mem = m_net_embed->stages[0].input_mems[0];
        auto &out_mem = m_net_embed->stages[0].output_mems[0];
        bm_memcpy_s2d(m_handle, in_mem, (void *)input_ids.data());
        if (!netLaunch(m_net_embed)) return E_AL_LAUNCH_FAIL;
        d2d(m_dev_buffer, out_mem);
        m_token_length = (int)tokens.size();
        return E_AL_OK;
    }

    int forwardFirst(const std::vector<int> &position_ids3) {
        std::vector<uint16_t> attention_mask;
        if (m_d.prefill_mask) {
            size_t n = m_d.is_dynamic ? (size_t)m_token_length * m_token_length
                                      : (size_t)m_d.MAX_INPUT_LENGTH * m_d.MAX_INPUT_LENGTH;
            attention_mask.assign(n, m_d.mask_value);
            size_t row = m_d.is_dynamic ? (size_t)m_token_length : (size_t)m_d.MAX_INPUT_LENGTH;
            for (int i = 0; i < m_token_length; i++)
                for (int j = 0; j <= i; j++) attention_mask[(size_t)i * row + j] = 0;
        }
        std::vector<int> pos_pad;
        if (m_d.is_dynamic) {
            pos_pad = position_ids3;
        } else {
            pos_pad.assign((size_t)3 * m_d.MAX_INPUT_LENGTH, 0);
            int ori = m_token_length;
            for (int d3 = 0; d3 < 3; d3++)
                std::copy(position_ids3.begin() + (size_t)d3 * ori,
                          position_ids3.begin() + (size_t)(d3 + 1) * ori,
                          pos_pad.begin() + (size_t)d3 * m_d.MAX_INPUT_LENGTH);
        }

        bm_device_mem_t out_mem = m_dev_buffer;
        /* io_alone 分支复用张量句柄（各层 input_num/output_num 一致） */
        std::vector<bm_tensor_t> ff_ins, ff_outs;
        if (m_io_alone) {
            ff_ins.resize((size_t)m_net_blocks[0]->input_num);
            ff_outs.resize((size_t)m_net_blocks[0]->output_num);
        }
        /* block_0 全部输入先 empty_net 清零（chat.cpp forward_first 惯例，无效区置安全值） */
        {
            int zero = 0;
            auto &net0 = m_net_blocks[0];
            for (int i = 0; i < net0->input_num; i++)
                bm_memset_device_ext(m_handle, &zero, 1, net0->stages[0].input_mems[i]);
        }
        for (int idx = 0; idx < m_d.NUM_LAYERS; idx++) {
            auto &net = m_net_blocks[idx];
            auto &in0 = net->stages[0].input_mems[0];
            auto &in1 = net->stages[0].input_mems[1];
            auto &in2 = net->stages[0].input_mems[2];
            d2d(in0, out_mem);
            if (idx == 0) {
                bm_memcpy_s2d_partial(m_handle, in1, (void *)pos_pad.data(),
                                      pos_pad.size() * sizeof(int));
                if (m_d.prefill_mask)
                    bm_memcpy_s2d_partial(m_handle, in2, (void *)attention_mask.data(),
                                          attention_mask.size() * sizeof(uint16_t));
            }
            if (m_io_alone) {
                /* io_alone：outs[1]/[2] 直接重绑到 past 缓存（offset 0），
                 * prefill KV 原地写，免逐层 d2d 回拷 */
                for (int i = 0; i < net->input_num; i++)
                    bmrt_tensor_with_device(&ff_ins[i], net->stages[0].input_mems[i],
                                            net->input_dtypes[i], net->stages[0].input_shapes[i]);
                bmrt_tensor_with_device(&ff_outs[0], net->stages[0].output_mems[0],
                                        net->output_dtypes[0], net->stages[0].output_shapes[0]);
                bmrt_tensor_with_device(&ff_outs[1], m_past_key[idx], net->output_dtypes[1],
                                        net->stages[0].output_shapes[1]);
                bmrt_tensor_with_device(&ff_outs[2], m_past_value[idx], net->output_dtypes[2],
                                        net->stages[0].output_shapes[2]);
                if (!bmrt_launch_tensor_ex(m_bmrt, net->name, ff_ins.data(), ff_ins.size(),
                                           ff_outs.data(), ff_outs.size(), true, false))
                    return -E_AL_LAUNCH_FAIL;   /* 负错误码，调用方按 token<0 判失败 */
            } else {
                if (!netLaunch(net)) return -E_AL_LAUNCH_FAIL;   /* 负错误码，调用方按 token<0 判失败 */
                d2d(m_past_key[idx], net->stages[0].output_mems[1]);
                d2d(m_past_value[idx], net->stages[0].output_mems[2]);
            }
            out_mem = net->stages[0].output_mems[0];
        }

        int bytes = m_d.HIDDEN_SIZE * (int)sizeof(uint16_t);
        auto &lm_in = m_net_lm->stages[0].input_mems[0];
        auto &lm_out = m_net_lm->stages[0].output_mems[0];
        bm_memcpy_d2d_byte(m_handle, lm_in, 0, out_mem, (m_token_length - 1) * bytes, bytes);
        if (!netLaunch(m_net_lm)) return -E_AL_LAUNCH_FAIL;
        int token = 0;
        if (m_d.lmhead_with_topk) {
            bm_memcpy_d2s(m_handle, (void *)&token, lm_out);
        } else {
            /* 无 greedy_head：返回负错误码（调用方按 token<0 判失败，防错误码当 token） */
            if (!m_net_greedy) return -E_AL_NET_NOT_MATCH;
            token = greedySearch(lm_out);
        }
        m_token_length++;
        m_history_length = m_token_length;
        return token;
    }

    int forwardNext(const int position_ids[3]) {
        /* 复用成员缓冲：mask 原地刷新（0→[history-1,SEQLEN) 置 mask_value），
         * ins/outs 不再每 token 每层堆分配 */
        std::fill(m_next_mask.begin(), m_next_mask.end(), (uint16_t)0);
        for (int i = m_history_length - 1; i < m_d.SEQLEN; i++) m_next_mask[i] = m_d.mask_value;
        const uint16_t *attention_mask = m_next_mask.data();

        auto &lm_in = m_net_lm->stages[0].input_mems[0];
        auto &lm_out = m_net_lm->stages[0].output_mems[0];

        auto &ec_in = m_net_embed_cache->stages[0].input_mems[0];
        auto &ec_out = m_net_embed_cache->stages[0].output_mems[0];
        d2d(ec_in, lm_out);
        if (!netLaunch(m_net_embed_cache)) return -E_AL_LAUNCH_FAIL;

        int token_offset = (m_history_length - 1) * m_d.KV_BYTES;
        bm_device_mem_t out_mem = ec_out;
        for (int idx = 0; idx < m_d.NUM_LAYERS; idx++) {
            auto &net = m_net_blocks_cache[idx];
            auto &ins = m_dec_ins;
            auto &outs = m_dec_outs;
            /* pos/mask 所有层复用 block_cache_0 的 mem（idx==0 写入后各层共享，
             * 对齐 chat.cpp net_launch_decode 的 idx>0 分支） */
            auto &in1 = m_net_blocks_cache[0]->stages[0].input_mems[1];
            auto &in2 = m_net_blocks_cache[0]->stages[0].input_mems[2];
            bmrt_tensor_with_device(&ins[0], out_mem, net->input_dtypes[0],
                                    net->stages[0].input_shapes[0]);
            if (idx == 0) {
                bm_memcpy_s2d(m_handle, in1, (void *)position_ids);
                bm_memcpy_s2d(m_handle, in2, (void *)attention_mask);
            }
            bmrt_tensor_with_device(&ins[1], in1, net->input_dtypes[1], net->stages[0].input_shapes[1]);
            bmrt_tensor_with_device(&ins[2], in2, net->input_dtypes[2], net->stages[0].input_shapes[2]);
            bmrt_tensor_with_device(&ins[3], net->stages[0].input_mems[3], net->input_dtypes[3],
                                    net->stages[0].input_shapes[3]);
            bmrt_tensor_with_device(&ins[4], net->stages[0].input_mems[4], net->input_dtypes[4],
                                    net->stages[0].input_shapes[4]);
            bmrt_tensor_with_device(&outs[0], net->stages[0].output_mems[0], net->output_dtypes[0],
                                    net->stages[0].output_shapes[0]);
            if (m_io_alone) {
                /* io_alone：KV 输出按偏移原地写 past 缓存（零拷贝） */
                auto k_mem = bm_mem_from_device(m_past_key[idx].u.device.device_addr + token_offset,
                                                m_d.KV_BYTES);
                auto v_mem = bm_mem_from_device(m_past_value[idx].u.device.device_addr + token_offset,
                                                m_d.KV_BYTES);
                bmrt_tensor_with_device(&outs[1], k_mem, net->output_dtypes[1],
                                        net->stages[0].output_shapes[1]);
                bmrt_tensor_with_device(&outs[2], v_mem, net->output_dtypes[2],
                                        net->stages[0].output_shapes[2]);
            } else {
                /* BASIC（非 io_alone）：KV 输出走 stage 独立缓冲，launch 后 d2d 回写
                 * past+offset（对齐 ChatGLM3 chat.cpp decode 逐层回拷路径） */
                bmrt_tensor_with_device(&outs[1], net->stages[0].output_mems[1], net->output_dtypes[1],
                                        net->stages[0].output_shapes[1]);
                bmrt_tensor_with_device(&outs[2], net->stages[0].output_mems[2], net->output_dtypes[2],
                                        net->stages[0].output_shapes[2]);
            }
            if (!bmrt_launch_tensor_ex(m_bmrt, net->name, ins.data(), ins.size(), outs.data(),
                                       outs.size(), true, false))
                return -E_AL_LAUNCH_FAIL;   /* 负错误码，decodeLoop 按 token<0 判失败 */
            if (!m_io_alone) {
                /* BASIC 回退：stage KV 输出 → past 缓存对应 token 偏移 */
                bm_memcpy_d2d_byte(m_handle, m_past_key[idx], token_offset,
                                   net->stages[0].output_mems[1], 0, m_d.KV_BYTES);
                bm_memcpy_d2d_byte(m_handle, m_past_value[idx], token_offset,
                                   net->stages[0].output_mems[2], 0, m_d.KV_BYTES);
            }
            out_mem = net->stages[0].output_mems[0];
        }

        d2d(lm_in, out_mem);
        if (!netLaunch(m_net_lm)) return -E_AL_LAUNCH_FAIL;
        int token = 0;
        if (m_d.lmhead_with_topk) {
            bm_memcpy_d2s(m_handle, (void *)&token, lm_out);
        } else {
            /* 无 greedy_head：返回负错误码（调用方按 token<0 判失败，防错误码当 token） */
            if (!m_net_greedy) return -E_AL_NET_NOT_MATCH;
            token = greedySearch(lm_out);
        }
        m_token_length++;
        m_history_length++;
        return token;
    }

    int historyLength() const { return m_history_length; }
    bool ioAlone() const { return m_io_alone; }   /* init 检测结果（日志/排障用） */
    void setHistoryMaxPosid(int v) { m_history_max_posid = v; }
    int historyMaxPosid() const { return m_history_max_posid; }
    /* 性能统计（generate 路径有效；asr 的 prefill/decodeLoop 分步路径仅 tok 数有效） */
    double lastFtlMs() const { return m_last_ftl_ms; }
    double lastTps() const { return m_last_tps; }
    int lastTokNum() const { return m_last_tok_num; }

protected:
    /** launch 一个子网（stages mems 直用）；false = launch 失败 */
    bool netLaunch(const bm_net_info_t *net) {
        std::vector<bm_tensor_t> ins(net->input_num), outs(net->output_num);
        for (int i = 0; i < net->input_num; i++)
            bmrt_tensor_with_device(&ins[i], net->stages[0].input_mems[i], net->input_dtypes[i],
                                    net->stages[0].input_shapes[i]);
        for (int i = 0; i < net->output_num; i++)
            bmrt_tensor_with_device(&outs[i], net->stages[0].output_mems[i], net->output_dtypes[i],
                                    net->stages[0].output_shapes[i]);
        return bmrt_launch_tensor_ex(m_bmrt, net->name, ins.data(), net->input_num, outs.data(),
                                     net->output_num, true, false);
    }

    int greedySearch(bm_device_mem_t &logits_mem) {
        if (!m_net_greedy) return -E_AL_NET_NOT_MATCH;   /* 负错误码，防错误码当 token */
        auto &in_mem = m_net_greedy->stages[0].input_mems[0];
        auto &out_mem = m_net_greedy->stages[0].output_mems[0];
        d2d(in_mem, logits_mem);
        if (!netLaunch(m_net_greedy)) return -E_AL_LAUNCH_FAIL;
        int token = 0;
        bm_memcpy_d2s(m_handle, (void *)&token, out_mem);
        return token;
    }

    void d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
        bm_memcpy_d2d_byte(m_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
    }

    std::string m_model;
    bm_handle_t m_handle = nullptr;
    void *m_bmrt = nullptr;
    const bm_net_info_t *m_net_embed = nullptr, *m_net_embed_cache = nullptr;
    const bm_net_info_t *m_net_lm = nullptr, *m_net_greedy = nullptr;
    std::vector<const bm_net_info_t *> m_net_blocks, m_net_blocks_cache;
    std::vector<bm_device_mem_t> m_past_key, m_past_value;
    std::vector<uint16_t> m_next_mask;       /* decode 复用：attention mask */
    std::vector<bm_tensor_t> m_dec_ins, m_dec_outs; /* decode 复用：张量句柄 */
    bm_device_mem_t m_dev_buffer{};
    bool m_buffer_ok = false;
    bool m_ready = false;
    bool m_io_alone = false;   /* io_alone(addr_mode=1)：KV 输出可原地写 past 缓存 */
    AsrDims m_d;
    int m_token_length = 0, m_history_length = 0;
    int m_max_posid = 0, m_history_max_posid = 0;
    /* 性能统计 */
    double m_last_ftl_ms = 0, m_last_tps = 0;
    int m_last_tok_num = 0;
};

} /* namespace al */
#endif /* AL_TEXT_LLM_H */
