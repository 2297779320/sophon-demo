//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// MixedAudioLink - 混合 bmodel 音频推理框架 C API
//
//===----------------------------------------------------------------------===//
#ifndef AUDIO_LINK_H
#define AUDIO_LINK_H

#include <stddef.h>

/*============================ 版本 ============================*/

/* 版本号宏（语义化版本；构建脚本可经 -DAL_GIT_HASH/-DAL_BUILD_DATE 注入，
 * CMake 侧默认 "unknown" 占位） */
#define AL_VERSION_MAJOR 0
#define AL_VERSION_MINOR 1
#define AL_VERSION_PATCH 0

#ifndef AL_GIT_HASH
#define AL_GIT_HASH "unknown"
#endif
#ifndef AL_BUILD_DATE
#define AL_BUILD_DATE "unknown"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 返回版本静态字符串（进程生命周期内有效，勿释放）：
 * 格式 "MixedAudioLink <major>.<minor>.<patch> (<git_hash> <build_date>)"
 */
const char *AudioLinkGetVersion(void);

/*============================ 通用类型 ============================*/

/** 通用句柄：管理器 / Link 实例均用它，内部结构对外不可见 */
typedef void *AL_HANDLE;

/** 统一状态码（对齐 rk_libs E_StateCode 风格） */
typedef enum {
    E_AL_OK = 0,            /* 成功 */
    E_AL_PARAM_ERR,         /* 参数非法 */
    E_AL_NOT_INIT,          /* 未初始化 */
    E_AL_LINK_NOT_FOUND,    /* 名字未注册/不存在 */
    E_AL_MODEL_LOAD_FAIL,   /* bmodel 加载失败 */
    E_AL_NET_NOT_MATCH,     /* 子网缺失/shape 不符 */
    E_AL_LAUNCH_FAIL,       /* 推理 launch 失败 */
    E_AL_NO_FRAME,          /* 拉模型无数据 */
    E_AL_TIMEOUT,           /* 对象池满（背压语义，非错误） */
    E_AL_BUSY,              /* 资源忙 */
    E_AL_INTERNAL_ERR,      /* 内部错误 */
    E_AL_BUFFER_TOO_SMALL,  /* 出参缓冲不足（内容被截断，未写入完整结果） */
    E_AL_MODEL_UNLOADED     /* 目标模型当前不可用：已卸载/被重载（推理前有效性校验失败） */
} E_AL_STATE;

/** Link 类型常量（AudioLinkCreate 的 type 参数） */
#define AL_LINK_PCM     "pcm"   /* PCM 输入 sink（AudioLinkPushPcm 喂数） */
#define AL_LINK_VAD     "vad"   /* FSMN VAD 流式检测 + 切段 */
#define AL_LINK_FEAT    "feat"  /* 特征提取（mel128 / fbank 多支路） */
#define AL_LINK_ASR     "asr"   /* qwen3_asr 整句识别 */
#define AL_LINK_SPK     "spk"   /* CAM++ 声纹 embedding */
#define AL_LINK_DIARIZE "diarize" /* 说话人归属（在线聚类）+ 会议转写缓冲 */
#define AL_LINK_SINK    "sink"  /* 结果汇聚 → 用户回调 */

/**
 * 通用参数：void* + size 风格，结构体可扩展不破坏 ABI。
 * 具体语义由每个 API / Ctrl 命令定义。
 */
typedef struct {
    const void *data;   /* 入参为数据/结构体指针；出参由框架填写 */
    int         size;   /* 字节数 */
} AL_PARAM;

/** 识别+声纹结果（回调期间有效，回调返回后失效；需保留请自行拷贝） */
typedef struct {
    int          start_ms;   /* 语音段起始（相对 PushPcm 音频流） */
    int          end_ms;     /* 语音段结束 */
    const char  *text;       /* UTF-8 识别文本，未启用 ASR 时为 NULL */
    const float *spk_emb;    /* 声纹向量，未启用声纹时为 NULL */
    int          spk_dim;    /* 声纹向量维度 */
    int          speaker_id; /* 说话人编号（1 起始，diarize 在线聚类；0=未归属） */
    const char  *line;       /* "讲话人N：xxx" 格式化行（diarize 填写，可为 NULL） */
} AL_RESULT;

/** 整段结果回调（在 sink 线程上下文中调用，勿长时间阻塞） */
typedef void (*AL_RESULT_CB)(const AL_RESULT *result, void *user);

/** 可选的增量回调：ASR 解码过程中逐词推送部分文本（Ctrl "set_partial_cb" 注册） */
typedef void (*AL_PARTIAL_CB)(const char *partial_text, void *user);

/*============================ 生命周期 ============================*/

/**
 * 初始化管理器并加载 JSON 配置中声明的全部模型。
 * @param json_cfg 配置文件路径（audio_link.json）
 * @param dev_id   TPU 设备号
 * @return 管理器句柄；失败返回 NULL（错误码通过 AudioLinkCtrl "get_last_error" 查询）
 */
AL_HANDLE AudioLinkInit(const char *json_cfg, int dev_id);

/** 创建一个 Link 实例（Start 前调用）。param.data 可指向各类型的创建参数结构体，可传 NULL */
int AudioLinkCreate(AL_HANDLE mgr, const char *name, const char *type, AL_PARAM param);

/** 拉模型连线：下游 dst 可拉取上游 src 的输出帧（Start 前调用，形成拓扑后校验） */
int AudioLinkBind(AL_HANDLE mgr, const char *src, const char *dst);

/** 启动链路（内部起 VAD 线程 + 任务线程） */
int AudioLinkStart(AL_HANDLE mgr);

/** 停止链路（未冲刷的语音段将被丢弃；冲刷请先 Ctrl "flush"） */
int AudioLinkStop(AL_HANDLE mgr);

/** 逆初始化并释放全部资源（含模型） */
int AudioLinkDeinit(AL_HANDLE mgr);

/*============================ 数据接口 ============================*/

/**
 * 推入 PCM 音频（非阻塞；环形缓冲满时丢最旧并计数告警）。
 * 格式按 JSON pcm.input_fmt 解释（默认 16kHz/mono/int16，此时行为与历史版本
 * 逐字节一致）；非 16k 输入请用 AudioLinkPushPcmEx。
 */
int AudioLinkPushPcm(AL_HANDLE mgr, const void *pcm, int bytes);

/**
 * 显式格式推入 PCM（非阻塞，环形缓冲语义同 AudioLinkPushPcm）。
 * bits 仅支持 16；sample_rate/channels 任意（首次成功推入即锁存格式，
 * 此后不一致返回 E_AL_PARAM_ERR）；非 16k/mono 输入内部经多相 FIR
 * 重采样到 16k/mono 后进链。AudioLinkPushPcm 等价于按 JSON pcm.input_fmt
 * （默认 16k/1/16）调用本接口。
 */
int AudioLinkPushPcmEx(AL_HANDLE mgr, const void *pcm, int bytes, int sample_rate, int channels, int bits);

/**
 * 万能控制口
 *
 * 管理器级（name 传 NULL 或 ""）：
 *   "get_last_error" : param.data=char[256] 返回最近错误描述
 *                      （Init 失败返回 NULL 后仍可查，静态最近错误；其余命令 NULL 句柄返回 E_AL_NOT_INIT）
 *   "get_stats"      : param.data=char[1024] 返回统计 JSON（PCM 丢弃计数等）
 *   "flush"          : 冲刷 VAD 残留语音段（is_final 语义）
 *   "new_meeting"    : 清空说话人质心与转写缓冲，开启新会议
 *   "set_status_file": param.data=状态快照路径；Stop 退出前自动写入 Markdown
 *                      运行状态（版本/时长/段数/丢弃/转写统计），NULL=关闭
 *   "list_nets"      : param.data=char[4096] 枚举全部已加载子网及角色
 *   "set_log"        : param.data=int 日志级别：-1=关闭，0..3=AL_LOG_ERROR/WARN/INFO/DEBUG
 *                      （作用域：仅音频流管理器日志；summary 模块与全局 al_log
 *                       不受影响，如需全关请直接调 al_log_set_level）
 *   "set_log_file"   : param.data=日志文件路径字符串（AL_PARAM{path,字节数}）；NULL=回退 stderr
 *   "save_params"    : 把运行期专用配置命令修改写回启动 JSON（先写 <cfg>.tmp 再 rename
 *                      原子替换；重排为字典序键 + 2 空格缩进标准 JSON，原注释不保留）
 *
 * Link 级（name=Link 名，如 "asr"）：
 *   "set_result_cb"  : param.data=&AL_RESULT_CB 结构 {cb, user}（推荐用 sink 的 create 参数）
 *   "set_partial_cb" : param.data=&AL_PARTIAL_CB 结构，仅 asr 支持，NULL 取消
 *   "set_language"   : param.data=const char* 语言名（asr；NULL 取消强制语言）
 *   "clear_history"  : 清空 asr KV history（enable_history=true 时有效）
 *   专用配置命令      : 数值统一 double 协议（AL_PARAM{&dVal, sizeof(double)}；
 *                      set_system_prompt 为字符串 AL_PARAM{str, len}），运行期热调
 *                      立即生效——vad: set_max_end_silence_ms /
 *                      set_sil_to_speech_time_ms / set_noise_thres / set_max_seg_s /
 *                      set_gate_enable / set_gate_rms / set_gate_hold_ms；
 *                      asr: set_max_new_tokens /
 *                      set_system_prompt / set_enable_history；diarize:
 *                      set_threshold / set_max_speakers；pcm: set_ring_ms
 *                      （协议与参数清单见 docs/API.md 动态配置节）
 *   "get_params"     : param.data=char[N] 出参，当前参数 JSON（vad/asr/diarize/pcm 支持）
 *   "get_model_info" : param.data=char[2048] 返回该 Link 关联模型维度推导结果
 *   "unload_model"   : 真实卸载该 Link 的 bmodel 释放显存（v1 仅 asr；先停用推理再调）；
 *                      卸载期间段仍做声纹聚类，无文本不追加转写行
 *   "load_model"     : 重新加载（重绑定子网/维度推导/重建 dev_buffer）
 *
 * diarize 级（name="diarize"，会议转写）：
 *   "get_transcript"     : param.data=char[N] 出参，全量转写文本（"讲话人N：xxx\n…"），
 *                          即会议总结模块（meeting_summary.h）的输入
 *   "new_meeting"        : 清空质心与转写缓冲（等价管理器级 new_meeting）
 *   "get_speaker_stats"  : param.data=char[1024] 出参，各说话人段数统计 JSON
 *
 * 子网级（name="<model名>.<子网名>"，如 "asr.block_0"）：
 *   "enable" / "disable" : 子网启停标记（当前版本仅登记状态供 list_nets 查询，
 *                          不改变调度行为；能力级开关用各 Link 的 Ctrl 命令）
 */
int AudioLinkCtrl(AL_HANDLE mgr, const char *name, const char *strCmd, AL_PARAM param);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_LINK_H */
