//===----------------------------------------------------------------------===//
//
// link_mng.h - CBaseLink 抽象类 / CLinkMng 通用管理器基座 / 帧对象池
// 调度模型（决策已确认）：
//   上半段（实时流）：PCM → VAD 为拉模型 + 帧对象池（借用-归还，池满=背压）
//   下半段（段级任务）：VAD 切段 → 任务队列 → 任务线程串行 feat→asr→spk→sink
// 框架通用化分层：本类只保留工作流通用骨架（配置/对象池/段队列/线程/统计/
// 通用 Ctrl），音频工作流专属逻辑下沉至派生类 CAudioLinkMng（audio_link_mng.h），
// 经 protected 虚钩子回接。
//
//===----------------------------------------------------------------------===//
#ifndef AL_LINK_MNG_H
#define AL_LINK_MNG_H

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "audio_link.h"
#include "model_hub.h"

namespace al {

class JsonValue;
class CLinkMng;

/*============================ 帧对象 ============================*/

/** 链路帧：池中循环复用（借用-归还协议） */
struct ALFrame {
    std::vector<int16_t> pcm;    /* 16k mono int16 */
    int start_ms = 0;
    int end_ms = 0;
    bool is_final = false;       /* flush 冲刷产生的尾段 */
};

using FramePtr = std::shared_ptr<ALFrame>;

/*============================ Link 基类 ============================*/

class CBaseLink {
public:
    CBaseLink(CLinkMng *mng, std::string name, std::string type);
    virtual ~CBaseLink();

    CBaseLink(const CBaseLink &) = delete;
    CBaseLink &operator=(const CBaseLink &) = delete;

    /** 创建（cfg=JSON 全局配置，param=外部参数） */
    virtual int create(const JsonValue *cfg, AL_PARAM param) = 0;
    virtual int start() { return E_AL_OK; }
    virtual int stop() { return E_AL_OK; }

    /** 拉模型：下游取一帧输出；无数据返回 E_AL_NO_FRAME */
    virtual int getFrame(FramePtr &out) { (void)out; return E_AL_NO_FRAME; }
    /** 万能控制口，命令语义见 audio_link.h AudioLinkCtrl 注释 */
    virtual int ctrl(const std::string &cmd, AL_PARAM param);

    /** Bind 注入：设置上游（仅允许一个源，rk_libs 拓扑约束） */
    int setSource(CBaseLink *src);

    const std::string &name() const { return m_name; }
    const std::string &type() const { return m_type; }

protected:
    /* 从管理器对象池借一帧；池满返回 nullptr（背压语义） */
    FramePtr acquireFrame();
    /* 上游拉一帧 */
    int pullSource(FramePtr &out);
    /** 取模型的可推理引用（**防悬垂**：unload/load 会让裸指针与 net_info 失效）。
     *  非空 ⇒ 该模型此刻 loaded 且 bmrt 有效，且代次与 gen 一致（gen=0 表示不校验代次）；
     *  空   ⇒ 已卸载或已被重载（缓存的 net_info/mems 指针已失效），调用方须放弃本次推理
     *        并返回 E_AL_MODEL_UNLOADED，不得再触碰 m_info / 设备 mem。
     *  注意：shared_ptr 只保证 SModel 对象存活，**不保证 bmrt 仍有效**，故这里的
     *        loaded/bmrt/gen 三连校验是必需的，不能只判指针非空。 */
    std::shared_ptr<SModel> acquireModel(const std::string &model, uint32_t gen);
    CLinkMng *mng() { return m_mng; }
    const JsonValue *cfg() const;

    CLinkMng *m_mng;
    std::string m_name;
    std::string m_type;
    CBaseLink *m_src = nullptr;
};

/*============================ 管理器基座 ============================*/

/** 帧对象池上限（满时 frameAcquire 返回空 FramePtr = 背压） */
#define AL_POOL_MAX_FRAMES 64

class CLinkMng {
public:
    CLinkMng() = default;
    virtual ~CLinkMng() { deinit(); }   /* 虚析构：经基座指针 delete 派生类实例 */

    CLinkMng(const CLinkMng &) = delete;
    CLinkMng &operator=(const CLinkMng &) = delete;

    /** AudioLinkInit 实现：读 JSON、加载模型、创建默认链路（子类经 createLinks 钩子落地） */
    int init(const char *json_cfg, int dev_id);
    /** AudioLinkDeinit 实现 */
    void deinit();

    /** 创建 Link 实例（工厂分支由子类 createLinkImpl 提供） */
    int createLink(const std::string &name, const std::string &type, AL_PARAM param);
    int bind(const std::string &src, const std::string &dst);
    int start();
    int stop();
    /** 推入 PCM（工作流专属数据口：基座默认不支持，子类按需覆写） */
    virtual int pushPcm(const void *pcm, int bytes);
    /** 显式格式推入 PCM（工作流专属数据口：基座默认不支持，子类按需覆写） */
    virtual int pushPcmEx(const void *pcm, int bytes, int rate, int ch, int bits);
    int ctrl(const char *name, const char *cmd, AL_PARAM param);

    CModelHub &hub() { return m_hub; }
    /** 模型名的**唯一来源**：按角色 role 跨模型反查，无角色配置时退回键名 fallbackKey。
     *  各 Link（asr/spk/vad）定位模型必须走它，不要各写一套键名规则 ——
     *  否则键一改名就会出现"开得了、卸不掉"。
     *  返回空串 = 既无该角色、也无同名模型键。 */
    std::string modelNameByRole(const std::string &role, const std::string &fallbackKey) const;
    std::string asrModelName() const;   /* = modelNameByRole("asr_llm", "asr") */
    const JsonValue *rootCfg() const { return m_cfg.get(); }
    const JsonValue *section(const std::string &key) const;

    /** 配置内资产路径解析：相对路径以启动 JSON 所在目录为基准
     *  （空串/绝对路径原样返回；返回统一 POSIX '/' 分隔符） */
    std::string resolveAsset(const std::string &p) const;

    /** 模型文件路径解析：相对路径以**应用根（启动 CWD）**为基准
     *  （空串/绝对路径原样返回；返回统一 POSIX '/' 分隔符）。
     *  与 resolveAsset 区分：models.*.path 随应用部署走，配置资产随 JSON 走 */
    std::string resolveModelPath(const std::string &p) const;

    /* ---- 动态配置写回（Link 专用配置命令配套，修改立即反映到内存配置树） ----
     * sec 必须已存在（不存在返回 false），key 允许新增到已存在 sec；
     * 四参重载的 sub 可传空串，sub 节不存在时自动创建（落在 sec.sub.key 原位置） */
    bool UpdateCfgNum(const std::string &sec, const std::string &key, double v);
    bool UpdateCfgNum(const std::string &sec, const std::string &sub, const std::string &key, double v);
    bool UpdateCfgStr(const std::string &sec, const std::string &key, const std::string &v);

    /* ---- 帧对象池（借还协议） ---- */
    FramePtr frameAcquire();
    void frameRelease(ALFrame *f);

    /* ---- 段任务队列（生产线程 → 任务线程） ---- */
    void pushSegment(FramePtr seg);
    bool popSegment(FramePtr &seg, int timeout_ms);

    /* ---- 统计/日志 ---- */
    void addDropped(int64_t bytes) { m_dropped_bytes += bytes; }
    int64_t droppedBytes() const { return m_dropped_bytes.load(); }
    void log(const std::string &msg);       /* INFO 级（m_log_en 关时不输出） */
    void logWarn(const std::string &msg);   /* WARN 级（钳制/降级告警） */
    void logErr(const std::string &msg);    /* ERROR 级（setLastError 失败路径配套） */
    void setLogEnabled(bool en) { m_log_en = en; }
    void setLastError(const std::string &e);
    std::string lastError();

    /** 任务线程是否正在处理段（unload_model 前轮询等待空闲） */
    bool taskBusy() const { return m_task_busy.load(); }

    /* 运行统计（退出快照用） */
    void addPushed(int64_t bytes) { m_pushed_bytes += bytes; }
    void addSegment() { m_seg_total++; }
    void IncSelfHeal() { m_self_heal_count++; }   /* 自愈成功计数（工作流子类上报） */

    /** 配置/查询退出状态快照路径（Stop 时写入 Markdown，空=关闭） */
    void setStatusFile(const std::string &path) { m_status_path = path; }
    const std::string &statusFile() const { return m_status_path; }

    CBaseLink *findLink(const std::string &name);

    bool running() const { return m_running.load(); }

protected:
    /*---- 框架通用化虚钩子（默认空实现；音频工作流在 CAudioLinkMng 落地） ----*/

    /** 音频（工作流）专属配置键校验：ValidateConfig 两段末尾各调一次，
     *  返回空 = 通过，非空 = 首个错误描述（含完整 JSON 路径） */
    virtual std::string validateAudio(bool after_hub);
    /** init 末尾（strCfgPath 赋值前）调用：创建工作流默认链路并连线 */
    virtual int createLinks();
    /** start 的线程启动处 / stop 的线程 teardown 处（基座默认不起任何线程） */
    virtual int startWorkers();
    virtual void stopWorkers();
    /** start() 失败回滚：逆序停掉 started 里已启动成功的 link（不持 m_links_mtx 调用） */
    void rollbackStarted(const std::vector<std::string> &started);
    /** ctrl 工作流命令转发：管理器级通用命令全不匹配时，及 Link 级
     *  unload_model/load_model 命令字命中时调用（未识别返回 E_AL_PARAM_ERR） */
    virtual int onCtrl(const std::string &name, const std::string &cmd, AL_PARAM param);
    /** get_stats JSON 拼接末尾（追加工作流增量字段，如 ,"k":v） */
    virtual void appendStats(std::string &js);
    /** dumpStatusMd 返回前：追加工作流专属快照段 */
    virtual void appendStatusMd(std::string &md);
    /** Link 工厂分支：type 未知返回 nullptr（基座 createLink 统一报错） */
    virtual CBaseLink *createLinkImpl(const std::string &type, const std::string &name);

    /* 校验共享工具（validateAudio 等子类实现复用） */
    static bool configFileExists(const std::string &path);
    static std::string rangeCheck(const JsonValue *sec, const char *sec_name, const char *key,
                                  double lo, double hi, bool lo_open, bool hi_open);

    void teardownThreads();    /* join 全部工作线程（线程对象在基座，析构安全） */
    void dumpStatusOnStop();   /* Stop 最前：写 Markdown 运行状态快照 */
    std::string dumpStatusMd();   /* 快照正文（通用段 + appendStatusMd 钩子） */

    /**
     * 配置全量校验（P1③）：返回空 = 通过，非空 = 首个错误描述（含完整 JSON 路径）。
     * after_hub=false：JSON parse 后调用，查通用必需键 / 文件存在 / log 范围
     *                  （工作流专属键经 validateAudio 钩子）；
     * after_hub=true ：hub.init 后调用，查各模型 roles 是否命中子网。
     * strict=true 校验失败拒绝启动；false 仅告警沿用默认/钳制值。
     */
    std::string ValidateConfig(bool after_hub);

    /*---- 以下成员派生类（工作流线程/命令实现）可见 ----*/
    CModelHub m_hub;
    std::shared_ptr<JsonValue> m_cfg;
    std::map<std::string, std::unique_ptr<CBaseLink>> m_links;
    mutable std::mutex m_links_mtx;

    /* 对象池 */
    std::vector<ALFrame *> m_pool;
    int m_pool_size = 0;
    std::mutex m_pool_mtx;

    /* 段任务队列 */
    std::deque<FramePtr> m_seg_queue;
    std::mutex m_seg_mtx;
    std::condition_variable m_seg_cv;

    /* 线程（对象留基座：先 join 再析构，保证任何派生类析构路径安全） */
    std::thread m_vad_thread, m_task_thread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_task_busy{false};   /* 任务线程处理段期间置位（popSegment 持锁置位） */
    std::atomic<bool> m_task_pause{false};  /* 卸载闸门：置位后任务线程不再取新段 */

    /* 统计与错误 */
    std::atomic<int64_t> m_dropped_bytes{0};
    std::atomic<bool> m_log_en{true};
    std::atomic<int64_t> m_pushed_bytes{0};
    std::atomic<int64_t> m_seg_total{0};
    std::atomic<int> m_self_heal_count{0};   /* 自愈成功次数（get_stats 上报） */
    std::chrono::steady_clock::time_point m_start_time{};
    std::string m_status_path;             /* 退出状态快照路径（空=关闭） */
    std::string strCfgPath;                /* 启动 JSON 路径（Ctrl save_params 写回用） */
    mutable std::mutex m_err_mtx;
    std::string m_last_err;
};

} /* namespace al */
#endif /* AL_LINK_MNG_H */
