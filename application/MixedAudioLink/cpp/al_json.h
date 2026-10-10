//===----------------------------------------------------------------------===//
//
// al_json.h - 极简 JSON 解析器（header-only，仅内部使用）
// 支持对象/数组/字符串/数值/bool/null；字符串支持 \" \\ \/ \b \f \n \r \t \uXXXX
//
//===----------------------------------------------------------------------===//
#ifndef AL_JSON_H
#define AL_JSON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace al {

class JsonValue;
using JsonPtr = std::shared_ptr<JsonValue>;

class JsonValue {
public:
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };

    Type type = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JsonPtr> arr;
    /* OBJ 用 std::map：**按键字典序**存储（非插入序）；
     * 依赖"首个模型"之类语义的调用方不要依赖声明顺序 */
    std::map<std::string, JsonPtr> obj;

    bool isNull() const { return type == NUL; }
    bool isObj() const { return type == OBJ; }
    bool isArr() const { return type == ARR; }
    bool isStr() const { return type == STR; }
    bool isNum() const { return type == NUM; }

    JsonPtr get(const std::string &key) const {
        if (type != OBJ) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : it->second;
    }
    /* 取字符串，缺省返回 def */
    std::string getStr(const std::string &key, const std::string &def = "") const {
        auto v = get(key);
        return (v && v->isStr()) ? v->str : def;
    }
    /* 取数值，"auto"/缺失返回 def（auto 语义由调用方决定） */
    double getNum(const std::string &key, double def) const {
        auto v = get(key);
        if (!v) return def;
        if (v->isNum()) return v->num;
        return def;
    }
    bool getBool(const std::string &key, bool def) const {
        auto v = get(key);
        return v ? (v->type == BOOL ? v->b : def) : def;
    }

    /*---- 序列化/写入（动态配置写回 + save_params 持久化用） ----*/

    /* OBJ 节点写入/覆盖键值（非 OBJ 节点调用直接忽略；键可新增可覆盖） */
    void Set(const std::string &key, double v) {
        if (type != OBJ) return;
        auto p = std::make_shared<JsonValue>();
        p->type = NUM;
        p->num = v;
        obj[key] = std::move(p);
    }
    void Set(const std::string &key, const std::string &v) {
        if (type != OBJ) return;
        auto p = std::make_shared<JsonValue>();
        p->type = STR;
        p->str = v;
        obj[key] = std::move(p);
    }
    void Set(const std::string &key, bool v) {
        if (type != OBJ) return;
        auto p = std::make_shared<JsonValue>();
        p->type = BOOL;
        p->b = v;
        obj[key] = std::move(p);
    }

    /* 递归序列化：indent = 每层空格数（0 = 紧凑输出）；
     * 键按字典序（std::map 遍历序），NUL 子节点跳过，原注释/键顺序不保留 */
    std::string Dump(int indent = 2) const {
        std::string out;
        dumpTo(out, indent, 0);
        return out;
    }

private:
    /* 递归输出实现（dump 辅助）：depth 从 0 起，pad = indent × depth 空格 */
    void dumpTo(std::string &out, int indent, int depth) const {
        bool pretty = indent > 0;
        std::string nl = pretty ? "\n" : "";
        std::string pad = pretty ? std::string((size_t)indent * (size_t)depth, ' ') : "";
        std::string pad1 = pretty ? std::string((size_t)indent * (size_t)(depth + 1), ' ') : "";

        if (OBJ == type) {
            out += '{';
            bool first = true;
            for (auto &kv : obj) {
                if (!kv.second || NUL == kv.second->type) continue;   /* null 值键跳过 */
                out += first ? nl : ("," + nl);
                first = false;
                out += pad1 + '"';
                out += kv.first;
                out += "\": ";
                kv.second->dumpTo(out, indent, depth + 1);
            }
            if (!first) out += nl;
            out += pad + '}';
        } else if (ARR == type) {
            out += '[';
            bool first = true;
            for (auto &v : arr) {
                if (!v || NUL == v->type) continue;
                out += first ? nl : ("," + nl);
                first = false;
                out += pad1;
                v->dumpTo(out, indent, depth + 1);
            }
            if (!first) out += nl;
            out += pad + ']';
        } else if (STR == type) {
            DumpEsc(out, str);
        } else if (NUM == type) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.15g", num);
            out += buf;
        } else if (BOOL == type) {
            out += b ? "true" : "false";
        }
        /* NUL 类型：无输出（容器内已跳过，根节点 NUL 输出空串） */
    }

    /* 字符串转义：" \ 与全部 <0x20 控制符（\n \t \r \b \f 用短转义，其余 \u00XX），
     * 保证 dump 输出为 RFC 8259 合法 JSON（外部工具可解析） */
    static void DumpEsc(std::string &out, const std::string &s) {
        out += '"';
        for (char c : s) {
            unsigned char uc = (unsigned char)c;
            if ('"' == c) out += "\\\"";
            else if ('\\' == c) out += "\\\\";
            else if ('\n' == c) out += "\\n";
            else if ('\t' == c) out += "\\t";
            else if ('\r' == c) out += "\\r";
            else if ('\b' == c) out += "\\b";
            else if ('\f' == c) out += "\\f";
            else if (uc < 0x20) {   /* 其余控制符：\u00XX */
                char esc[8];
                snprintf(esc, sizeof(esc), "\\u%04X", uc);
                out += esc;
            } else out += c;
        }
        out += '"';
    }
};

class JsonParser {
public:
    /* 解析成功返回根节点，失败返回 nullptr 并填 err */
    static JsonPtr parse(const std::string &text, std::string &err) {
        JsonParser p(text.c_str(), text.size());
        p.skipWs();
        JsonPtr v = p.parseValue();
        p.skipWs();
        if (!v || p.pos != p.len) {
            err = "json parse error at offset " + std::to_string(p.pos);
            return nullptr;
        }
        return v;
    }

private:
    const char *s;
    size_t len, pos = 0;
    JsonParser(const char *str, size_t l) : s(str), len(l) {}

    void skipWs() {
        while (pos < len && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r' ||
                             s[pos] == '/' /* 容忍 // 注释 */
                             )) {
            if (s[pos] == '/' && pos + 1 < len && s[pos + 1] == '/') {
                while (pos < len && s[pos] != '\n') pos++;
            } else if (s[pos] == '/') {
                break;
            } else {
                pos++;
            }
        }
    }
    bool consume(char c) {
        if (pos < len && s[pos] == c) { pos++; return true; }
        return false;
    }
    JsonPtr parseValue() {
        skipWs();
        if (pos >= len) return nullptr;
        char c = s[pos];
        if (c == '{') return parseObj();
        if (c == '[') return parseArr();
        if (c == '"') { auto v = std::make_shared<JsonValue>(); v->type = JsonValue::STR; v->str = parseStr(); return v; }
        if (c == 't' && pos + 3 < len && strncmp(s + pos, "true", 4) == 0) { pos += 4; auto v = std::make_shared<JsonValue>(); v->type = JsonValue::BOOL; v->b = true; return v; }
        if (c == 'f' && pos + 4 < len && strncmp(s + pos, "false", 5) == 0) { pos += 5; auto v = std::make_shared<JsonValue>(); v->type = JsonValue::BOOL; v->b = false; return v; }
        if (c == 'n' && pos + 3 < len && strncmp(s + pos, "null", 4) == 0) { pos += 4; return std::make_shared<JsonValue>(); }
        if (c == '-' || (c >= '0' && c <= '9')) {
            char *end = nullptr;
            double d = strtod(s + pos, &end);
            if (end == s + pos) return nullptr;
            pos = end - s;
            auto v = std::make_shared<JsonValue>();
            v->type = JsonValue::NUM; v->num = d;
            return v;
        }
        return nullptr;
    }
    JsonPtr parseObj() {
        if (!consume('{')) return nullptr;
        auto v = std::make_shared<JsonValue>();
        v->type = JsonValue::OBJ;
        skipWs();
        if (consume('}')) return v;
        while (true) {
            skipWs();
            if (pos >= len || s[pos] != '"') return nullptr;
            std::string key = parseStr();
            skipWs();
            if (!consume(':')) return nullptr;
            JsonPtr val = parseValue();
            if (!val) return nullptr;
            v->obj[key] = val;
            skipWs();
            if (consume(',')) continue;
            if (consume('}')) return v;
            return nullptr;
        }
    }
    JsonPtr parseArr() {
        if (!consume('[')) return nullptr;
        auto v = std::make_shared<JsonValue>();
        v->type = JsonValue::ARR;
        skipWs();
        if (consume(']')) return v;
        while (true) {
            JsonPtr val = parseValue();
            if (!val) return nullptr;
            v->arr.push_back(val);
            skipWs();
            if (consume(',')) continue;
            if (consume(']')) return v;
            return nullptr;
        }
    }
    std::string parseStr() {
        std::string out;
        if (!consume('"')) return out;
        while (pos < len && s[pos] != '"') {
            char c = s[pos++];
            if (c != '\\') { out += c; continue; }
            if (pos >= len) break;
            char e = s[pos++];
            switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                /* \uXXXX：不足 4 个 hex 字符时消费剩余字符原样保留（防错位） */
                int got = 0;
                unsigned cp = 0;
                while (got < 4 && pos < len) {
                    char h = s[pos];
                    int v;
                    if (h >= '0' && h <= '9') v = h - '0';
                    else if (h >= 'a' && h <= 'f') v = h - 'a' + 10;
                    else if (h >= 'A' && h <= 'F') v = h - 'A' + 10;
                    else break;
                    cp = (cp << 4) + (unsigned)v;
                    pos++;
                    got++;
                }
                if (got == 0) out += 'u';
                if (cp == 0 && got < 4) break;
                /* UTF-8 编码 */
                if (cp < 0x80) out += (char)cp;
                else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                break;
            }
            default: break;
            }
        }
        consume('"');
        return out;
    }
};

} // namespace al
#endif /* AL_JSON_H */
