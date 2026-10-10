//===----------------------------------------------------------------------===//
//
// al_log.h - 全局统一日志（header-only，C++17 inline 保证跨编译单元单例）
// 输出格式：[YYYY-MM-DD HH:MM:SS.mmm][LEVEL][tag] msg
//   - 级别：AL_LOG_ERROR(0) / WARN(1) / INFO(2) / DEBUG(3)，默认 INFO；
//   - 输出：默认 stderr，al_log_set_file() 重定向到文件（单文件超 max_bytes
//     时改名 path+".1" 并重建，仅保留 1 份历史）；
//   - 线程安全：写路径与配置接口互斥锁保护（级别快读为无锁良性竞态）。
// 使用：Link/管理器统一走 AL_LOG_E/W/I/D(tag, fmt, ...)（tag=link 名或模块名）；
//       meeting_summary 等无 mgr 句柄的模块直接用全局函数（tag 用 "summary"）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_LOG_H
#define AL_LOG_H

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>

/* 日志级别（数值越小越严重） */
#define AL_LOG_ERROR 0
#define AL_LOG_WARN  1
#define AL_LOG_INFO  2
#define AL_LOG_DEBUG 3

/* 便捷宏：tag=link 名或模块名 */
#define AL_LOG_E(tag, ...) al_log(AL_LOG_ERROR, tag, __VA_ARGS__)
#define AL_LOG_W(tag, ...) al_log(AL_LOG_WARN, tag, __VA_ARGS__)
#define AL_LOG_I(tag, ...) al_log(AL_LOG_INFO, tag, __VA_ARGS__)
#define AL_LOG_D(tag, ...) al_log(AL_LOG_DEBUG, tag, __VA_ARGS__)

namespace al {

/*============================ 单例状态 ============================*/
/* 函数内 static 引用：规避跨编译单元静态初始化顺序问题 */

inline std::mutex &al_log_mtx() { static std::mutex m; return m; }
inline FILE *&al_log_fp() { static FILE *fp = nullptr; return fp; }   /* null = stderr */
inline std::string &al_log_path() { static std::string p; return p; }
inline long long &al_log_bytes() { static long long b = 0; return b; }
inline long long &al_log_max() { static long long v = 8LL * 1024 * 1024; return v; }
inline int &al_log_level() { static int lv = AL_LOG_INFO; return lv; }

inline const char *al_log_level_name(int lv) {
    switch (lv) {
    case AL_LOG_ERROR: return "ERROR";
    case AL_LOG_WARN:  return "WARN";
    case AL_LOG_INFO:  return "INFO";
    default:           return "DEBUG";
    }
}

/* 关闭当前日志文件回到 stderr（仅内部使用，调用方需已持锁） */
inline void al_log_close_file_locked() {
    if (al_log_fp()) { fclose(al_log_fp()); al_log_fp() = nullptr; }
    al_log_path().clear();
    al_log_bytes() = 0;
}

/** 设置日志级别（0..3，超界钳位） */
inline void al_log_set_level(int level) {
    if (level < AL_LOG_ERROR) level = AL_LOG_ERROR;
    if (level > AL_LOG_DEBUG) level = AL_LOG_DEBUG;
    std::lock_guard<std::mutex> lk(al_log_mtx());
    al_log_level() = level;
}

/**
 * 重定向日志到文件；path=NULL/空串回退 stderr。
 * 简单轮转：单文件超过 max_bytes（<=0 表示不轮转）时改名 path+".1"
 * （覆盖旧历史，仅保留 1 份）后重建。打开失败保持 stderr。
 */
inline void al_log_set_file(const char *path, long long max_bytes = 8LL * 1024 * 1024) {
    std::lock_guard<std::mutex> lk(al_log_mtx());
    al_log_close_file_locked();
    if (max_bytes > 0) al_log_max() = max_bytes;
    if (!path || !path[0]) return;
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    al_log_fp() = fp;
    al_log_path() = path;
    fseek(fp, 0, SEEK_END);
    al_log_bytes() = ftell(fp);
}

/** 当前级别下是否输出（供调用方避免无谓的字符串拼接） */
inline bool al_log_enabled(int level) { return level <= al_log_level(); }

/*============================ 核心输出 ============================*/

/** 全局日志入口：al_log(level, tag, fmt, ...)，宏 AL_LOG_E/W/I/D 封装 */
inline void al_log(int level, const char *tag, const char *fmt, ...) {
    if (level > al_log_level()) return;   /* 级别过滤（int 读，良性竞态） */

    /* 时间戳 [YYYY-MM-DD HH:MM:SS.mmm] */
    struct timespec ts;
#if defined(_WIN32)
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    struct tm tmv;
#if defined(_WIN32)
    localtime_s(&tmv, &ts.tv_sec);
#else
    localtime_r(&ts.tv_sec, &tmv);
#endif

    char head[80];
    snprintf(head, sizeof(head), "[%04d-%02d-%02d %02d:%02d:%02d.%03d][%s][%s]",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec, (int)(ts.tv_nsec / 1000000),
             al_log_level_name(level), tag ? tag : "al");

    /* 消息格式化：先栈缓冲，超长再堆一次（fmt 与入参独立，两遍安全） */
    char stackbuf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stackbuf, sizeof(stackbuf), fmt, ap);
    va_end(ap);
    std::string big;
    const char *msg = stackbuf;
    if (n < 0) {
        msg = "(log format error)";
    } else if (n >= (int)sizeof(stackbuf)) {
        big.resize((size_t)n + 1);   /* 含空终结符，&big[0] 写 n+1 字节不越界 */
        va_start(ap, fmt);
        vsnprintf(&big[0], (size_t)n + 1, fmt, ap);
        va_end(ap);
        msg = big.c_str();
    }
    size_t msg_len = strlen(msg);
    size_t head_len = strlen(head);

    std::lock_guard<std::mutex> lk(al_log_mtx());
    /* 轮转检查：预判写入后超限则先轮转（字节数含 head+msg+换行，避免低估 30-80%） */
    FILE *out = al_log_fp();
    if (out && al_log_max() > 0 &&
        al_log_bytes() + (long long)(head_len + msg_len + 2) > al_log_max()) {
        fclose(out);
        al_log_fp() = nullptr;
        std::string old = al_log_path() + ".1";
        remove(old.c_str());                        /* 仅保留 1 份历史 */
        rename(al_log_path().c_str(), old.c_str());
        out = fopen(al_log_path().c_str(), "a");    /* 失败则本次退回 stderr */
        al_log_fp() = out;
        al_log_bytes() = 0;
    }
    out = al_log_fp() ? al_log_fp() : stderr;
    fprintf(out, "%s %s\n", head, msg);
    fflush(out);   /* 逐条落盘：板端异常退出场景保日志 */
    if (al_log_fp()) al_log_bytes() += (long long)(head_len + msg_len + 2);
}

} /* namespace al */
#endif /* AL_LOG_H */
