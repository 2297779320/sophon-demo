//===----------------------------------------------------------------------===//
//
// al_out_test.cpp - 出参缓冲写入（al_out.h）的行为与 ABI 回归
//
// 目的：E_AL_BUFFER_TOO_SMALL 是新增的公共错误码，且 12 处出参写入全部改走
//       alCopyOut/alFormatOut。若其判定边界写错（典型：把 `n >= size` 写成
//       `n > size`），调用方会拿到被截断的 JSON/摘要却仍收到 E_AL_OK ——
//       正是本次要消灭的"静默截断"。本测试锁死这些契约。
//
// 纯 CPU，不依赖 Sophon SDK / TPU 设备，退出码可接 CI。
//
// 运行：g++ -O2 -std=c++17 -I cpp tools/al_out_test.cpp -o /tmp/al_out_test
//       /tmp/al_out_test
// 期望：N passed, 0 failed -> PASS（退出码 0）
//
//===----------------------------------------------------------------------===//
#include <cstdio>
#include <cstring>
#include <string>

#include "al_out.h"

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char *what, const std::string &detail = "") {
    if (ok) { g_pass++; printf("  [ok]   %s\n", what); }
    else    { g_fail++; printf("  [FAIL] %s   %s\n", what, detail.c_str()); }
}

/** 构造一个填充了哨兵字节的缓冲，用于验证"截断时不越界写" */
struct Buf {
    char data[256];
    Buf() { memset(data, 0x7F, sizeof(data)); }
    AL_PARAM param(int size) { AL_PARAM p; p.data = data; p.size = size; return p; }
};

int main() {
    printf("=== case 1: ABI 错误码取值（新增码不得与既有码碰撞）===\n");
    check(E_AL_BUFFER_TOO_SMALL != E_AL_OK, "BUFFER_TOO_SMALL != OK");
    check(E_AL_BUFFER_TOO_SMALL != E_AL_PARAM_ERR, "BUFFER_TOO_SMALL != PARAM_ERR");
    check(E_AL_BUFFER_TOO_SMALL != E_AL_INTERNAL_ERR, "BUFFER_TOO_SMALL != INTERNAL_ERR");
    check(E_AL_BUFFER_TOO_SMALL != E_AL_NET_NOT_MATCH, "BUFFER_TOO_SMALL != NET_NOT_MATCH");

    printf("\n=== case 2: alCopyOut 正例（缓冲充足）===\n");
    {
        Buf b; AL_PARAM p = b.param(64);
        int rc = al::alCopyOut(p, "hello");
        check(rc == E_AL_OK, "rc == E_AL_OK", "rc=" + std::to_string(rc));
        check(std::string(b.data) == "hello", "内容完整写入");
        check(strlen(b.data) == 5, "NUL 结尾正确");
    }

    printf("\n=== case 3: alCopyOut 边界（size 恰好够 / 差 1 字节）===\n");
    {
        const std::string s = "abcd";            /* 需要 5 字节（含 '\0'） */
        Buf b1; AL_PARAM p1 = b1.param(5);
        check(al::alCopyOut(p1, s) == E_AL_OK, "size=5 恰好够 -> OK");

        Buf b2; AL_PARAM p2 = b2.param(4);
        int rc2 = al::alCopyOut(p2, s);
        check(rc2 == E_AL_BUFFER_TOO_SMALL, "size=4 差 1 字节 -> TOO_SMALL",
              "rc=" + std::to_string(rc2));
    }

    printf("\n=== case 4: alCopyOut 截断不得越界，且缓冲仍 NUL 结尾 ===\n");
    {
        const std::string s(100, 'x');
        Buf b; AL_PARAM p = b.param(8);
        /* 先把 8 字节之后全部填哨兵，验证只有前 8 字节被触碰 */
        memset(b.data + 8, 0x5A, sizeof(b.data) - 8);
        int rc = al::alCopyOut(p, s);
        check(rc == E_AL_BUFFER_TOO_SMALL, "长串截断 -> TOO_SMALL");
        check(strlen(b.data) == 7, "写入 7 字符 + NUL（未越界）",
              "strlen=" + std::to_string(strlen(b.data)));
        bool tail_intact = true;
        for (size_t i = 8; i < sizeof(b.data); i++)
            if ((unsigned char)b.data[i] != 0x5A) { tail_intact = false; break; }
        check(tail_intact, "size 之后的字节未被污染（无越界写）");
    }

    printf("\n=== case 5: alCopyOut 参数错误（PARAM_ERR 与 TOO_SMALL 须区分）===\n");
    {
        Buf b;
        AL_PARAM p_null; p_null.data = nullptr; p_null.size = 64;
        check(al::alCopyOut(p_null, "x") == E_AL_PARAM_ERR, "data==NULL -> PARAM_ERR");
        AL_PARAM p_zero; p_zero.data = b.data; p_zero.size = 0;
        check(al::alCopyOut(p_zero, "x") == E_AL_PARAM_ERR, "size=0 -> PARAM_ERR");
        AL_PARAM p_neg; p_neg.data = b.data; p_neg.size = -1;
        check(al::alCopyOut(p_neg, "x") == E_AL_PARAM_ERR, "size<0 -> PARAM_ERR");
    }

    printf("\n=== case 6: alCopyOut 空串 ===\n");
    {
        Buf b; AL_PARAM p = b.param(16);
        int rc = al::alCopyOut(p, "");
        check(rc == E_AL_OK, "空串 -> OK");
        check(strlen(b.data) == 0, "写出空 C 串");
        Buf b1; AL_PARAM p1 = b1.param(1);
        check(al::alCopyOut(p1, "") == E_AL_OK, "空串 size=1 恰好够 -> OK");
    }

    printf("\n=== case 7: alFormatOut 正例（多参数格式化）===\n");
    {
        Buf b; AL_PARAM p = b.param(128);
        int rc = al::alFormatOut(p, "{\"a\":%d,\"b\":%.2f,\"c\":\"%s\"}", 42, 3.5, "xy");
        check(rc == E_AL_OK, "格式化 -> OK");
        check(std::string(b.data) == "{\"a\":42,\"b\":3.50,\"c\":\"xy\"}",
              "格式化结果正确", b.data);
    }

    printf("\n=== case 8: alFormatOut 截断 ===\n");
    {
        Buf b; AL_PARAM p = b.param(8);
        int rc = al::alFormatOut(p, "%s", "0123456789ABCDEF");
        check(rc == E_AL_BUFFER_TOO_SMALL, "超长格式化 -> TOO_SMALL");
        check(strlen(b.data) == 7, "仅写入 7 字符 + NUL");
    }

    printf("\n=== case 9: alFormatOut 参数错误 ===\n");
    {
        Buf b;
        AL_PARAM p_null; p_null.data = nullptr; p_null.size = 32;
        check(al::alFormatOut(p_null, "%d", 1) == E_AL_PARAM_ERR, "data==NULL -> PARAM_ERR");
        AL_PARAM p_zero; p_zero.data = b.data; p_zero.size = 0;
        check(al::alFormatOut(p_zero, "%d", 1) == E_AL_PARAM_ERR, "size=0 -> PARAM_ERR");
    }

    printf("\n=== case 10: alOutNeedBytes 与 alCopyOut 边界一致 ===\n");
    {
        const std::string s = "hello";
        int need = al::alOutNeedBytes(s);
        check(need == 6, "need == len+1 == 6", "need=" + std::to_string(need));
        Buf b_ok; AL_PARAM p_ok = b_ok.param(need);
        check(al::alCopyOut(p_ok, s) == E_AL_OK, "按 need 分配 -> OK");
        Buf b_bad; AL_PARAM p_bad = b_bad.param(need - 1);
        check(al::alCopyOut(p_bad, s) == E_AL_BUFFER_TOO_SMALL, "need-1 -> TOO_SMALL");
        check(al::alOutNeedBytes("") == 1, "空串 need == 1");
    }

    printf("\n=== case 11: 含内嵌 NUL / 多字节字符不破坏契约 ===\n");
    {
        std::string with_nul("ab\0cd", 5);
        check(al::alOutNeedBytes(with_nul) == 6, "含 NUL 的 std::string 按 size 计长");
        Buf b; AL_PARAM p = b.param(32);
        int rc = al::alCopyOut(p, with_nul);
        check(rc == E_AL_OK, "含 NUL -> 仍按 std::string 语义返回 OK");
        check(strlen(b.data) == 2, "C 串视角在第 1 个 NUL 处终止（预期行为）");

        const std::string utf8 = u8"中文测试";
        Buf b2; AL_PARAM p2 = b2.param(64);
        check(al::alCopyOut(p2, utf8) == E_AL_OK, "UTF-8 写入 -> OK");
        check(std::string(b2.data) == utf8, "UTF-8 内容逐字节一致");
    }

    printf("\n%d passed, %d failed  ->  %s\n", g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
