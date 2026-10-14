//===----------------------------------------------------------------------===//
//
// al_out.h - 出参缓冲写入的统一收口（截断不再静默返回成功）
//
// 背景：C ABI 的出参一律是 AL_PARAM{void* data, int size}。此前各 link 直接
//       snprintf((char*)param.data, param.size, "%s", js.c_str())，
//       既不检查返回值也不检查 size —— 缓冲不足时调用方拿到的是被截断的
//       JSON/摘要，而返回值仍是 E_AL_OK，错误被完全吞掉。
//
// 约定：所有"把字符串写回调用方缓冲"的路径都走 alCopyOut / alFormatOut，
//       缓冲不足统一返回 E_AL_BUFFER_TOO_SMALL。
//       调用方需要多大？E_AL_BUFFER_TOO_SMALL 时可用 alOutNeedBytes() 查询
//       （= 含结尾 '\0' 的字节数），或直接用更大的缓冲重试。
//
// 注意：data==NULL 或 size<=0 视为调用方参数错误（E_AL_PARAM_ERR），
//       与"缓冲给小了"（E_AL_BUFFER_TOO_SMALL）区分开。
//
//===----------------------------------------------------------------------===//
#ifndef AL_OUT_H
#define AL_OUT_H

#include <cstdarg>
#include <cstdio>
#include <string>

#include "audio_link.h"

namespace al {

/** 字符串写入出参缓冲：成功 E_AL_OK；截断 E_AL_BUFFER_TOO_SMALL */
inline int alCopyOut(AL_PARAM &out, const std::string &s) {
    if (!out.data || out.size <= 0) return E_AL_PARAM_ERR;
    int n = snprintf((char *)out.data, (size_t)out.size, "%s", s.c_str());
    if (n < 0) return E_AL_INTERNAL_ERR;          /* 编码/格式化失败 */
    if (n >= out.size) return E_AL_BUFFER_TOO_SMALL;   /* 截断：snprintf 返回"本应写入"的长度 */
    return E_AL_OK;
}

/** 变长格式化写入出参缓冲（语义同 alCopyOut） */
inline int alFormatOut(AL_PARAM &out, const char *fmt, ...) {
    if (!out.data || out.size <= 0) return E_AL_PARAM_ERR;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf((char *)out.data, (size_t)out.size, fmt, ap);
    va_end(ap);
    if (n < 0) return E_AL_INTERNAL_ERR;
    if (n >= out.size) return E_AL_BUFFER_TOO_SMALL;
    return E_AL_OK;
}

/** 完整写入 s 所需的字节数（含结尾 '\0'）—— 供调用方"先问大小再分配" */
inline int alOutNeedBytes(const std::string &s) { return (int)s.size() + 1; }

} /* namespace al */

#endif /* AL_OUT_H */
