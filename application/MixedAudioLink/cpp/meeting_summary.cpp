//===----------------------------------------------------------------------===//
//
// meeting_summary.cpp - MeetingSummary 会议内容总结模块实现
//
//===----------------------------------------------------------------------===//
#include "meeting_summary.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <string>

#include "al_json.h"
#include "al_log.h"
#include "al_path.h"
#include "link_mng.h"      /* 复用 model_hub / text_llm / tokenizer */
#include "text_llm.h"

using namespace al;

class CSummary {
public:
    CModelHub hub;
    CTextLLM llm;
    QwenTokenizer tok;
    int im_end_id = -1;
    int max_new_tokens = 512;
    std::string prompt_template;
    std::string system_prompt;
    std::string last_err;
    std::string cfg_dir;   /* 启动 JSON 所在目录（配置内资产相对路径基准） */
    /* 可选流式回调（Ctrl "set_partial_cb"）：单块/合并终总结阶段逐词透传 */
    AL_PARTIAL_CB partial_cb = nullptr;
    void *partial_user = nullptr;

    /* 配置内资产路径解析：相对路径以 cfg_dir 为基准（实现见 cpp/al_path.h，规则同 CLinkMng::resolveAsset） */
    std::string resolveAsset(const std::string &p) const {
        return al::resolveAsset(cfg_dir, p);
    }
};

/* 统一错误出口：记录 last_err 并同步 al_log ERROR（本模块无 mgr 句柄，tag="summary"） */
static void summarySetErr(CSummary *s, const std::string &e) {
    s->last_err = e;
    AL_LOG_E("summary", "%s", e.c_str());
}

/* 字段化纪要模板（对齐官方 application/meeting_summary/python/prompt.txt 的全字段结构） */
static const char *kDefaultSystemPrompt =
    "你是一个会议纪要助手，你的作用是将一段会议文本总结为会议摘要。";

static const char *kDefaultPromptTemplate =
    "请以以下特定的会议纪要格式总结会议内容，如果找不到相关内容可以填\"未提及\"或者不填写：\n"
    "\n"
    "【会议基本信息】\n"
    "会议主题：\n"
    "会议时间：\n"
    "参会人员：\n"
    "\n"
    "【议程】\n"
    "1. \n"
    "2. \n"
    "\n"
    "【会议摘要】\n"
    "（按发言人归纳主要观点，格式：讲话人N：观点）\n"
    "\n"
    "【行动项】\n"
    "（格式：事项 - 责任人 - 完成时间）\n"
    "\n"
    "【下次会议】\n"
    "\n"
    "会议内容如下：\n{transcript}\n\n请总结：";

namespace {

std::string readFileAll2(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return "";
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
    fclose(fp);
    /* 剥离 UTF-8 BOM（EF BB BF）：Windows 编辑器常写入，避免污染 prompt/transcript */
    if (out.size() >= 3 && (unsigned char)out[0] == 0xEF && (unsigned char)out[1] == 0xBB &&
        (unsigned char)out[2] == 0xBF)
        out.erase(0, 3);
    return out;
}

/* Qwen 文本对话模板（无音频分支） */
std::string renderTextPrompt(const std::string &system, const std::string &user) {
    return "<|im_start|>system\n" + system + "<|im_end|>\n" +
           "<|im_start|>user\n" + user + "<|im_end|>\n" +
           "<|im_start|>assistant\n";
}

/* 分段总结合并模板（map-reduce 的 reduce 阶段，参考 meeting_summary 例程） */
const char *kMergeTemplate =
    "以下是同一会议的分段总结，请将其合并为一份最终会议总结：\n{transcript}";

} /* namespace */

extern "C" {

/*==================== 内部实现（map-reduce，参考 application/meeting_summary） ====================*/

/* 输入 token 预算：prefill ≤ MAX_INPUT_LENGTH 且 prefill+max_new ≤ SEQLEN-128 */
static int inputBudget(CSummary *s) {
    const AsrDims &d = s->llm.dims();
    int budget = d.MAX_INPUT_LENGTH;
    int seqlen_limit = d.SEQLEN - 128 - s->max_new_tokens;
    if (seqlen_limit > 0 && seqlen_limit < budget) budget = seqlen_limit;
    return budget;
}

/* 模板开销（不含转写内容）：system+模板骨架+对话包裹的 token 数 */
static int templateOverhead(CSummary *s, const std::string &templ) {
    std::string user = templ;
    size_t pos = user.find("{transcript}");
    if (pos != std::string::npos) user.replace(pos, 12, "");
    std::string prompt = renderTextPrompt(s->system_prompt, user);
    return (int)s->tok.encode(prompt).size();
}

/* 单次推理：user 文本填充模板 → encode → 校验预算 → greedy 生成（cb 可选流式） */
static int runOnce(CSummary *s, const std::string &templ, const std::string &body,
                   std::string &out, AL_PARTIAL_CB cb = nullptr, void *cb_user = nullptr) {
    std::string user = templ;
    size_t pos = user.find("{transcript}");
    if (pos != std::string::npos) user.replace(pos, 12, body);
    else user += "\n" + body;

    std::string prompt = renderTextPrompt(s->system_prompt, user);
    std::vector<int> ids = s->tok.encode(prompt);
    const AsrDims &d = s->llm.dims();
    if (ids.empty() || (int)ids.size() > d.MAX_INPUT_LENGTH ||
        (int)ids.size() + s->max_new_tokens > d.SEQLEN - 128) {
        summarySetErr(s, "prompt too long: " + std::to_string((int)ids.size()));
        return E_AL_PARAM_ERR;
    }

    s->llm.clearHistory();
    s->llm.setHistoryMaxPosid(0);
    return s->llm.generate(ids, s->tok, s->im_end_id, s->max_new_tokens,
                           cb, cb_user, out, /*history_pos_offset=*/0);
}

/* 将字节位置推进到 UTF-8 字符起始边界（跳过 0x80-0xBF 续字节），避免硬切截断多字节汉字 */
static size_t utf8NextCharStart(const std::string &s, size_t pos) {
    while (pos < s.size() && ((unsigned char)s[pos] & 0xC0) == 0x80) pos++;
    return pos;
}

/* 按行边界分块（对齐 meeting_summary 的 RecursiveCharacterTextSplitter 思路，
 * 但以"讲话人N：xxx"的行自然边界切分）；单行超预算时按比例字符硬切 */
static std::vector<std::string> splitByBudget(CSummary *s, const std::string &text,
                                              const std::string &templ) {
    std::vector<std::string> chunks;
    int budget = inputBudget(s) - templateOverhead(s, templ) - 16;
    if (budget <= 0) {
        /* 模板/对话骨架本身已吞掉输入预算：配置不可用，明确报错而非整文硬塞 */
        summarySetErr(s, "template overhead exceeds input budget (" +
                      std::to_string(templateOverhead(s, templ)) + " >= " +
                      std::to_string(inputBudget(s)) + "), shrink prompt_template");
        return chunks;   /* 空 = 出错，调用方检查 */
    }

    std::string cur;
    int cur_tokens = 0;   /* 增量 token 计数：每行只 encode 一次，避免 O(N²) 重编码 */
    size_t start = 0;
    while (start < text.size()) {
        size_t eol = text.find('\n', start);
        std::string line = text.substr(start, (eol == std::string::npos ? text.size() : eol + 1) - start);
        start = (eol == std::string::npos) ? text.size() : eol + 1;

        int lt = (int)s->tok.encode(line).size();
        int join_cost = cur_tokens > 0 ? 1 : 0;   /* 行间换行的近似 token 开销 */
        if (cur_tokens + lt + join_cost <= budget) {
            cur += line;
            cur_tokens += lt + join_cost;
            continue;
        }
        if (!cur.empty()) { chunks.push_back(cur); cur.clear(); cur_tokens = 0; }
        /* 单行超预算：按 token 比例估字符切分（UTF-8 边界对齐，防切断汉字） */
        int line_tokens = (int)s->tok.encode(line).size();
        if (line_tokens > budget) {
            size_t piece = std::max(1, (int)line.size() * budget / std::max(1, line_tokens));
            size_t p = 0;
            while (p < line.size()) {
                size_t e = utf8NextCharStart(line, p + piece);
                if (e <= p) e = utf8NextCharStart(line, p + 1);
                if (e > line.size()) e = line.size();
                chunks.push_back(line.substr(p, e - p));
                p = e;
            }
        } else {
            cur = line;
            cur_tokens = lt;
        }
    }
    if (!cur.empty()) chunks.push_back(cur);
    if (chunks.empty()) chunks.push_back(text);
    return chunks;
}

static int summaryReduce(CSummary *s, std::vector<std::string> &summaries,
                         AL_PARTIAL_CB cb = nullptr, void *cb_user = nullptr) {
    /* 合并终总结：分段摘要拼合后若仍超预算则迭代 reduce（上限 4 轮） */
    for (int iter = 0; iter < 4; iter++) {
        std::string merged;
        for (size_t i = 0; i < summaries.size(); i++)
            merged += summaries[i] + (i + 1 < summaries.size() ? "\n\n" : "");
        auto parts = splitByBudget(s, merged, kMergeTemplate);
        if (parts.size() == 1) {
            std::string final_out;
            int rc = runOnce(s, kMergeTemplate, merged, final_out, cb, cb_user);
            if (rc == E_AL_OK) summaries.assign(1, final_out);
            return rc;
        }
        /* 仍超预算：对每段再做一轮摘要压缩 */
        std::vector<std::string> next;
        for (auto &p : parts) {
            std::string one;
            int rc = runOnce(s, s->prompt_template, p, one);
            if (rc != E_AL_OK) return rc;
            next.push_back(one);
        }
        summaries.swap(next);
    }
    summarySetErr(s, "summary reduce exceeded 4 iterations");
    return E_AL_INTERNAL_ERR;
}

int MeetingSummaryRun(AL_HANDLE h, const char *transcript, AL_PARAM out) {
    if (!h) return E_AL_PARAM_ERR;
    CSummary *s = (CSummary *)h;
    if (!transcript) { summarySetErr(s, "Run: transcript is NULL"); return E_AL_PARAM_ERR; }
    if (!s->llm.ready()) { summarySetErr(s, "llm not ready"); return E_AL_NET_NOT_MATCH; }
    if (!out.data || out.size <= 0) { summarySetErr(s, "Run: output buffer invalid"); return E_AL_PARAM_ERR; }

    std::string text(transcript);
    auto chunks = splitByBudget(s, text, s->prompt_template);
    std::string result;

    if (chunks.empty()) return E_AL_PARAM_ERR;   /* splitByBudget 已写 last_err */
    if (chunks.size() == 1) {
        /* 短转写：单次生成（流式回调生效） */
        int rc = runOnce(s, s->prompt_template, text, result, s->partial_cb, s->partial_user);
        if (rc != E_AL_OK) return rc;
    } else {
        /* 长转写：map-reduce（参考 application/meeting_summary：分块摘要 → 合并终总结）。
         * map 阶段不推流，reduce 合并终总结阶段流式回调生效 */
        s->last_err = "";
        std::vector<std::string> summaries;
        for (size_t i = 0; i < chunks.size(); i++) {
            std::string one;
            int rc = runOnce(s, s->prompt_template, chunks[i], one);
            if (rc != E_AL_OK) return rc;
            summaries.push_back(one);
        }
        if (summaries.size() == 1) result = summaries[0];
        else {
            int rc = summaryReduce(s, summaries, s->partial_cb, s->partial_user);
            if (rc != E_AL_OK) return rc;
            result = summaries[0];
        }
    }
    snprintf((char *)out.data, out.size, "%s", result.c_str());
    return E_AL_OK;
}

int MeetingSummaryRunFile(AL_HANDLE h, const char *text_path, AL_PARAM out) {
    if (!h) return E_AL_PARAM_ERR;
    CSummary *s = (CSummary *)h;
    if (!text_path) { summarySetErr(s, "RunFile: text_path is NULL"); return E_AL_PARAM_ERR; }
    if (!out.data || out.size <= 0) { summarySetErr(s, "RunFile: output buffer invalid"); return E_AL_PARAM_ERR; }
    /* 读入整份转写文本文件（空文件/读失败明确报错，不静默送空串进 map-reduce） */
    std::string text = readFileAll2(text_path);
    if (text.empty()) {
        summarySetErr(s, "RunFile: read failed or empty file: " + std::string(text_path));
        return E_AL_PARAM_ERR;
    }
    /* 转调内存文本接口：map-reduce / 流式回调 / 预算校验全部继承 */
    return MeetingSummaryRun(h, text.c_str(), out);
}

AL_HANDLE MeetingSummaryInit(const char *json_cfg, int dev_id) {
    auto *s = new (std::nothrow) CSummary();
    if (!s) return nullptr;

    std::string text = readFileAll2(json_cfg);
    if (text.empty()) { summarySetErr(s, "config read failed"); goto fail; }
    {
        std::string err;
        auto cfg = JsonParser::parse(text, err);
        if (!cfg) { summarySetErr(s, "config parse: " + err); goto fail; }
        s->cfg_dir = json_cfg;   /* 资产相对路径解析基准（JSON 所在目录） */

        /* 模型加载（独立 CModelHub 实例，独立 bm_handle）；models.*.path 以
         * **应用根（启动 CWD）**为解析基准（与 CLinkMng::resolveModelPath 一致） */
        std::string hub_err = s->hub.init(cfg->get("models").get(), dev_id);
        if (!hub_err.empty()) { summarySetErr(s, hub_err); goto fail; }
        /* 定位唯一的 LLM 模型并推导维度 */
        auto models = cfg->get("models");
        std::string model_key = models && models->isObj() && !models->obj.empty()
                                    ? models->obj.begin()->first : "";
        if (model_key.empty()) { summarySetErr(s, "models empty"); goto fail; }
        {
            std::string warn = s->hub.deriveAsrDims(model_key, nullptr);
            if (!warn.empty()) { s->last_err = "dims warn: " + warn; AL_LOG_W("summary", "%s", warn.c_str()); }
        }

        /* LLM 公共路径 */
        AL_LOG_I("summary", "summary init: hub ok, model '%s'", model_key.c_str());
        if (s->llm.init(s->hub, model_key) != E_AL_OK) {
            summarySetErr(s, "llm init failed (需要 chat 格式子网: embedding/block_i/block_cache_i/lm_head)");
            goto fail;
        }
        AL_LOG_I("summary", "summary init: llm ok (io_alone=%d, layers=%d)",
                 (int)s->llm.ioAlone(), s->llm.dims().NUM_LAYERS);

        /* tokenizer（相对路径以启动 JSON 所在目录为基准解析） */
        auto sc = cfg->get("summary");
        std::string vocab = s->resolveAsset(sc ? sc->getStr("vocab", "summary_vocab.json") : "summary_vocab.json");
        std::string merges = s->resolveAsset(sc ? sc->getStr("merges", "summary_merges.txt") : "summary_merges.txt");
        std::map<std::string, int> specials = {
            {"<|im_end|>", 151645}, {"<|im_start|>", 151644},
        };
        if (!s->tok.load(vocab, merges, specials)) {
            summarySetErr(s, "tokenizer load failed: " + vocab);
            goto fail;
        }
        AL_LOG_I("summary", "summary init: tokenizer ok (%s)", vocab.c_str());
        s->im_end_id = s->tok.tokenToId("<|im_end|>");
        s->max_new_tokens = sc ? (int)sc->getNum("max_new_tokens", 512) : 512;
        s->system_prompt = sc ? sc->getStr("system_prompt", kDefaultSystemPrompt)
                              : kDefaultSystemPrompt;
        /* prompt 加载顺序：prompt_file（外置文本文件）> prompt_template（JSON 内联）> 内置默认模板 */
        std::string pf = s->resolveAsset(sc ? sc->getStr("prompt_file", "") : "");
        if (!pf.empty()) {
            std::string ftext = readFileAll2(pf.c_str());
            if (ftext.empty() || ftext.find("{transcript}") == std::string::npos) {
                summarySetErr(s, "prompt_file invalid: " + pf);
                goto fail;
            }
            s->prompt_template = ftext;
        } else {
            std::string pt = sc ? sc->getStr("prompt_template", "") : "";
            s->prompt_template = pt.empty() ? kDefaultPromptTemplate : pt;
        }
        AL_LOG_I("summary", "MeetingSummary ready, %s", AudioLinkGetVersion());
        return (AL_HANDLE)s;
    }
fail:
    /* 顺序：先释放 LLM 设备资源（dev_buffer 用到有效 bm_handle），再卸 hub */
    s->llm.deinit();
    s->hub.deinit();
    delete s;
    return nullptr;
}

int MeetingSummaryCtrl(AL_HANDLE h, const char *cmd, AL_PARAM param) {
    if (!h || !cmd) return E_AL_PARAM_ERR;
    CSummary *s = (CSummary *)h;
    std::string c = cmd;
    if (c == "get_last_error") {
        if (param.data && param.size > 0)
            snprintf((char *)param.data, param.size, "%s", s->last_err.c_str());
        return E_AL_OK;
    }
    if (c == "set_prompt") {
        if (!param.data || param.size <= 0) {
            summarySetErr(s, "set_prompt: invalid param");
            return E_AL_PARAM_ERR;
        }
        std::string t((const char *)param.data);
        if (t.find("{transcript}") == std::string::npos) {
            summarySetErr(s, "set_prompt: template missing {transcript}");
            return E_AL_PARAM_ERR;
        }
        s->prompt_template = t;
        return E_AL_OK;
    }
    if (c == "set_max_tokens") {
        if (!param.data || param.size < (int)sizeof(int)) {
            summarySetErr(s, "set_max_tokens: invalid param");
            return E_AL_PARAM_ERR;
        }
        int v = *(const int *)param.data;
        if (v < 1 || v > s->llm.dims().SEQLEN - 128) {
            summarySetErr(s, "set_max_tokens: value out of range (" + std::to_string(v) + ")");
            return E_AL_PARAM_ERR;
        }
        s->max_new_tokens = v;
        return E_AL_OK;
    }
    if (c == "set_partial_cb") {
        struct { AL_PARTIAL_CB cb; void *user; } *p = (decltype(p))param.data;
        if (!p || param.size < (int)sizeof(*p)) {
            summarySetErr(s, "set_partial_cb: invalid param");
            return E_AL_PARAM_ERR;
        }
        s->partial_cb = p->cb;   /* NULL 取消 */
        s->partial_user = p->user;
        return E_AL_OK;
    }
    if (c == "get_model_info") {
        if (param.data && param.size > 0) {
            const AsrDims &d = s->llm.dims();
            snprintf((char *)param.data, param.size,
                     "{\"loaded\":%s,\"seqlen\":%d,\"max_input\":%d,\"max_new_tokens\":%d,"
                     "\"last_ftl_ms\":%.1f,\"last_tps\":%.2f,\"last_tok_num\":%d}",
                     s->llm.ready() ? "true" : "false", d.SEQLEN, d.MAX_INPUT_LENGTH,
                     s->max_new_tokens, s->llm.lastFtlMs(), s->llm.lastTps(),
                     s->llm.lastTokNum());
        }
        return E_AL_OK;
    }
    summarySetErr(s, "ctrl: unknown command: " + c);
    return E_AL_PARAM_ERR;
}

int MeetingSummaryDeinit(AL_HANDLE h) {
    if (!h) return E_AL_NOT_INIT;
    CSummary *s = (CSummary *)h;
    s->llm.deinit();
    s->hub.deinit();
    delete s;
    return E_AL_OK;
}

} /* extern "C" */
