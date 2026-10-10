//===----------------------------------------------------------------------===//
//
// asr_tokenizer.h - Qwen BPE（GPT-2 字节级 BPE，vocab.json + merges.txt）
// + 简化 chat 模板渲染（chat_template.json 语义的 C++ 硬编码等价实现）。
// 特殊 token：tokenizer_config.json 中的 added_tokens（加载时解析，或 JSON asr 节指定）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_ASR_TOKENIZER_H
#define AL_ASR_TOKENIZER_H

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "al_json.h"

namespace al {

class QwenTokenizer {
public:
    /** 加载 vocab.json / merges.txt；specials 键值对（"<|im_end|>"→id 等） */
    bool load(const std::string &vocab_path, const std::string &merges_path,
              const std::map<std::string, int> &specials) {
        m_id2tok.clear();
        m_tok2id.clear();
        m_specials = specials;
        for (auto &kv : specials) m_special_set.insert(kv.first);

        /* ---- vocab.json ---- */
        FILE *fp = fopen(vocab_path.c_str(), "rb");
        if (!fp) return false;
        std::string text;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) text.append(buf, n);
        fclose(fp);
        std::string err;
        JsonPtr root = JsonParser::parse(text, err);
        if (!root || !root->isObj()) return false;
        for (auto &kv : root->obj) {
            if (!kv.second->isNum()) continue;
            int id = (int)kv.second->num;
            m_tok2id[kv.first] = id;
            if ((int)m_id2tok.size() <= id) m_id2tok.resize(id + 1);
            m_id2tok[id] = kv.first;
        }

        /* ---- merges.txt ---- */
        fp = fopen(merges_path.c_str(), "r");
        if (!fp) return false;
        char line[512];
        int rank = 0;
        while (fgets(line, sizeof(line), fp)) {
            std::string l(line);
            if (!l.empty() && l.back() == '\n') l.pop_back();
            if (l.empty() || l[0] == '#' || l.find(' ') == std::string::npos) continue;
            size_t sp = l.find(' ');
            m_bpe_ranks[l.substr(0, sp) + " " + l.substr(sp + 1)] = rank++;
        }
        fclose(fp);
        buildByteTables();
        return !m_id2tok.empty();
    }

    /*================== encode ==================*/

    /** 文本 → ids；模板中的特殊 token 一并编码（先按特殊 token 切分） */
    std::vector<int> encode(const std::string &text) const {
        std::vector<int> out;
        size_t pos = 0;
        while (pos < text.size()) {
            /* 优先匹配最长特殊 token */
            std::string hit;
            for (auto &s : m_special_set) {
                if (text.compare(pos, s.size(), s) == 0 && (hit.empty() || s.size() > hit.size()))
                    hit = s;
            }
            if (!hit.empty()) {
                /* 特殊 token 多为 added token（不在 vocab.json），优先 specials 表 */
                auto sp = m_specials.find(hit);
                out.push_back(sp != m_specials.end() ? sp->second : m_tok2id.at(hit));
                pos += hit.size();
                continue;
            }
            /* 简化 GPT-2 预切分：一个"空格+词"组为一个 pre-token */
            size_t end = pos;
            bool started = false;
            while (end < text.size()) {
                unsigned char c = text[end];
                if (started && c == ' ') break;
                started = true;
                /* UTF-8 多字节整体吞入 */
                if (c >= 0xF0) end += 4;
                else if (c >= 0xE0) end += 3;
                else if (c >= 0xC0) end += 2;
                else end += 1;
            }
            bpe(text.substr(pos, end - pos), out);
            pos = end;
        }
        return out;
    }

    /*================== decode ==================*/

    /** ids → UTF-8 文本；特殊 token 原样输出其字面量 */
    std::string decode(const std::vector<int> &ids) const {
        std::string bytes;
        for (int id : ids) {
            if (id < 0 || id >= (int)m_id2tok.size()) continue;
            const std::string &tok = m_id2tok[id];
            if (m_special_set.count(tok)) { bytes += tok; continue; }
            /* byte-unicode 逆映射：ASCII 可打印区单字节，其余（高字节可打印区与
             * 0x100+b 区）均为 2 字节 UTF-8（c>=0xC0 时按 2 字节读，与 enc 一致） */
            size_t i = 0;
            while (i < tok.size()) {
                unsigned char c = (unsigned char)tok[i];
                size_t len = (c >= 0xC0) ? 2 : 1;
                std::string seq = tok.substr(i, len);
                i += len;
                auto it = m_byte_dec_multi.find(seq);
                bytes += (it != m_byte_dec_multi.end()) ? it->second : seq;
            }
        }
        return bytes;
    }

    int tokenToId(const std::string &tok) const {
        auto it = m_tok2id.find(tok);
        if (it != m_tok2id.end()) return it->second;
        auto sp = m_specials.find(tok);
        return sp != m_specials.end() ? sp->second : -1;
    }
    const std::string &idToToken(int id) const {
        static const std::string empty;
        if (id < 0 || id >= (int)m_id2tok.size()) return empty;
        return m_id2tok[id];
    }
    bool isSpecial(const std::string &tok) const { return m_special_set.count(tok) > 0; }

    /*================== chat 模板（chat_template.json 等价硬编码） ==================*/

    /**
     * 渲染 prompt：
     *   <|im_start|>system\n{system}<|im_end|>\n
     *   <|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n
     *   <|im_start|>assistant\n
     * @param audio_pad_repeat <|audio_pad|> 重复次数（n_times * tokens_per_chunk）
     * @param language         强制语言（可空）；追加 "language xx<asr_text>"
     */
    std::string renderPrompt(const std::string &system, int audio_pad_repeat,
                             const std::string &language) const {
        std::string p = "<|im_start|>system\n" + system + "<|im_end|>\n";
        p += "<|im_start|>user\n<|audio_start|>";
        for (int i = 0; i < audio_pad_repeat; i++) p += "<|audio_pad|>";
        p += "<|audio_end|><|im_end|>\n";
        p += "<|im_start|>assistant\n";
        if (!language.empty()) p += "language " + language + "<asr_text>";
        return p;
    }

private:
    /** GPT-2 BPE 合并（按 rank 贪心） */
    void bpe(const std::string &word, std::vector<int> &out) const {
        if (word.empty()) return;
        /* 字节级编码 */
        std::vector<std::string> parts;
        for (unsigned char c : word) parts.push_back(m_byte_enc.at(c));
        if (parts.size() == 1) {
            out.push_back(tokenToId(parts[0]));
            return;
        }
        while (true) {
            int best_rank = 1 << 30;
            size_t best_i = SIZE_MAX;
            for (size_t i = 0; i + 1 < parts.size(); i++) {
                auto it = m_bpe_ranks.find(parts[i] + " " + parts[i + 1]);
                if (it != m_bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_i == SIZE_MAX) break;
            parts[best_i] = parts[best_i] + parts[best_i + 1];
            parts.erase(parts.begin() + best_i + 1);
        }
        for (auto &p : parts) out.push_back(tokenToId(p));
    }

    void buildByteTables() {
        /* GPT-2 bytes_to_unicode：ASCII 可打印区（0x21-0x7E）单字节原样；
         * 高字节"可打印"区（0xA1-0xAC、0xAE-0xFF）cp=b，vocab.json 中以
         * U+00XX 的 UTF-8 两字节存储（如 byte 0xE4 → "ä" = C3 A4）；
         * 其余（0x00-0x20、0x7F-0xA0、0xAD）cp=0x100+b，同为两字节 UTF-8。 */
        for (int b = 0; b < 256; b++) {
            std::string u;
            if (b >= 0x21 && b <= 0x7E) {
                u = std::string(1, (char)b);
            } else {
                int cp = ((b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) ? b : 0x100 + b;
                u += (char)(0xC0 | (cp >> 6));
                u += (char)(0x80 | (cp & 0x3F));
            }
            m_byte_enc[b] = u;
            m_byte_dec_multi[u] = std::string(1, (char)b);
        }
    }

    std::unordered_map<std::string, int> m_tok2id;
    std::vector<std::string> m_id2tok;
    std::map<std::string, int> m_specials;
    std::set<std::string> m_special_set;
    std::map<std::string, int> m_bpe_ranks;
    std::map<unsigned char, std::string> m_byte_enc;
    std::map<std::string, std::string> m_byte_dec_multi;
};

} /* namespace al */
#endif /* AL_ASR_TOKENIZER_H */
