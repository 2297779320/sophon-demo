//===----------------------------------------------------------------------===//
//
// audio_link.cpp - C API 桥接实现（audio_link.h → CAudioLinkMng）
//
//===----------------------------------------------------------------------===//
#include "audio_link.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

#include "al_log.h"
#include "al_out.h"   /* alCopyOut：出参写入统一查截断 */
#include "audio_link_mng.h"

using namespace al;

static inline CLinkMng *M(AL_HANDLE h) { return (CLinkMng *)h; }

/* Init 失败后句柄已 delete，"get_last_error" 仍须可查（audio_link.h 声明语义）：
 * 用文件级静态最近错误承载（仅 Init 失败路径写入） */
static std::mutex g_init_err_mtx;
static std::string g_init_last_err;

static void setInitError(const std::string &e) {
    std::lock_guard<std::mutex> lk(g_init_err_mtx);
    g_init_last_err = e;
}

static std::string getInitError() {
    std::lock_guard<std::mutex> lk(g_init_err_mtx);
    return g_init_last_err;
}

extern "C" {

/* 版本静态字符串（函数内 static：C++11 线程安全初始化） */
const char *AudioLinkGetVersion(void) {
    static const std::string ver = [] {
        char buf[128];
        snprintf(buf, sizeof(buf), "MixedAudioLink %d.%d.%d (%s %s)",
                 AL_VERSION_MAJOR, AL_VERSION_MINOR, AL_VERSION_PATCH,
                 AL_GIT_HASH, AL_BUILD_DATE);
        return std::string(buf);
    }();
    return ver.c_str();
}

AL_HANDLE AudioLinkInit(const char *json_cfg, int dev_id) {
    /* 音频工作流管理器（CLinkMng 通用基座 + 音频子类）：
     * pushPcm/ctrl 等经基座虚函数多态分发 */
    auto *mng = new (std::nothrow) CAudioLinkMng();
    if (!mng) return nullptr;
    if (mng->init(json_cfg, dev_id) != E_AL_OK) {
        AL_LOG_E("audiolink", "init failed: %s", mng->lastError().c_str());
        setInitError(mng->lastError());
        /* 资源释放统一由 virtual ~CLinkMng::deinit() 完成（幂等），此处不重复调用 */
        delete mng;
        return nullptr;
    }
    return (AL_HANDLE)mng;
}

int AudioLinkCreate(AL_HANDLE mgr, const char *name, const char *type, AL_PARAM param) {
    if (!mgr || !name || !type) return E_AL_PARAM_ERR;
    return M(mgr)->createLink(name, type, param);
}

int AudioLinkBind(AL_HANDLE mgr, const char *src, const char *dst) {
    if (!mgr || !src || !dst) return E_AL_PARAM_ERR;
    return M(mgr)->bind(src, dst);
}

int AudioLinkStart(AL_HANDLE mgr) {
    if (!mgr) return E_AL_NOT_INIT;
    return M(mgr)->start();
}

int AudioLinkStop(AL_HANDLE mgr) {
    if (!mgr) return E_AL_NOT_INIT;
    return M(mgr)->stop();
}

int AudioLinkDeinit(AL_HANDLE mgr) {
    if (!mgr) return E_AL_NOT_INIT;
    delete M(mgr);
    return E_AL_OK;
}

int AudioLinkPushPcm(AL_HANDLE mgr, const void *pcm, int bytes) {
    if (!mgr) return E_AL_NOT_INIT;
    return M(mgr)->pushPcm(pcm, bytes);
}

int AudioLinkPushPcmEx(AL_HANDLE mgr, const void *pcm, int bytes, int sample_rate, int channels, int bits) {
    if (!mgr) return E_AL_NOT_INIT;
    return M(mgr)->pushPcmEx(pcm, bytes, sample_rate, channels, bits);
}

int AudioLinkCtrl(AL_HANDLE mgr, const char *name, const char *strCmd, AL_PARAM param) {
    if (!strCmd) return E_AL_PARAM_ERR;
    if (!mgr) {
        /* Init 失败（句柄 NULL）后仅 "get_last_error" 可查：读静态最近错误 */
        if (std::string(strCmd) == "get_last_error") {
            return al::alCopyOut(param, getInitError());
        }
        return E_AL_NOT_INIT;
    }
    return M(mgr)->ctrl(name, strCmd, param);
}

} /* extern "C" */
