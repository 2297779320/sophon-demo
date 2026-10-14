//===----------------------------------------------------------------------===//
//
// model_hub.h - CModelHub：多 bmodel 多上下文管理
// 每个 bmodel 文件一个 bmrt 上下文；动态枚举子网；JSON 关键字映射角色；
// 维度自动推导器（init_by_names 模式，JSON 仅作回退）。
//
// 【线程模型】（2026-10-14 明确）
//   * 本类内部自带 shared_mutex：读接口（查询/枚举）持共享锁，写接口
//     （init/deinit/unload/load/setNetEnabled/deriveAsrDims）持独占锁。
//     因此任意线程并发调用本类的任意方法都是安全的（不会出现 map 迭代与
//     插入/替换交错的 UB）。
//   * 但**指针生命周期不在锁的保护范围内**：model()/netInfo() 返回的是裸指针，
//     调用方拿到后锁即释放。若此刻另一线程 unload/load 该模型，SModel 会被
//     析构（bmrt 句柄失效），裸指针即悬垂。
//     => 推理线程与卸载/重载必须互斥：上层（CAudioLinkMng）在 unload/load
//        前等待任务线程空闲，并在卸载后让 Link 走 onModelUnload 联动。
//        本锁只保证"不会因并发访问容器本身而崩"，不替代这层协议。
//   * 推理侧另有**防御性校验**：CBaseLink::acquireModel() 在每次推理前取
//     modelRef() + 校验 loaded/bmrt/gen，失效即放弃本次推理（E_AL_MODEL_UNLOADED）。
//     它把"误用悬垂指针"降级为"返回错误"，是上层的兜底而非替代。
//
//===----------------------------------------------------------------------===//
#ifndef AL_MODEL_HUB_H
#define AL_MODEL_HUB_H

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <vector>

#include <bmruntime_interface.h>
#include <bmlib_runtime.h>

namespace al {

class JsonValue;

/** qwen3_asr 推导维度（对应 LLM-TPU chat.cpp init_by_names 的公开常量） */
struct AsrDims {
    int SEQLEN = 0;
    int MAX_INPUT_LENGTH = 0;
    int HIDDEN_SIZE = 0;
    int NUM_LAYERS = 0;
    int AUDIO_LENGTH = 0;      /* audio 子网输入最后一维（mel 帧数/块） */
    int AUDIO_MEL_BINS = 0;    /* audio 子网输入倒数第 2 维（mel bins，实测 128） */
    int KV_BYTES = 0;          /* 单 token KV 字节数 */
    int PREFILL_KV_LENGTH = 0; /* support_history 时 block_0 的 kv 输入长度 */
    uint16_t mask_value = 0;   /* fp16=0xF0E2 / bf16=0xC61C */
    bool lmhead_with_topk = false;
    bool support_history = false;
    bool is_dynamic = false;
    bool prefill_mask = false;
    bool valid = false;
};

/** 单个 bmodel 上下文 */
struct SModel {
    std::string name;                        /* JSON models 下的键 */
    std::string path;
    const JsonValue *cfg = nullptr;          /* 原始 JSON 配置（重载用） */
    void *bmrt = nullptr;                    /* bmruntime 上下文（卸载后为 null） */
    std::vector<std::string> net_names;      /* 动态枚举的全部子网 */
    std::map<std::string, const bm_net_info_t *> nets;
    std::map<std::string, std::string> role_of_net;   /* 子网名 → 角色 */
    std::set<std::string> disabled;          /* Ctrl enable/disable 管理 */
    bool loaded = false;
    /** 加载代次：每次成功 loadOne 递增（unload 不变）。调用方（Link）在 create 时
     *  记下 gen，推理前比对 —— 不一致说明模型已被重载，缓存的 net_info/mems
     *  指针已随旧 bmrt 销毁而失效，必须停止使用（Link 侧重新绑定）。 */
    uint32_t gen = 0;
};

/**
 * CModelHub - 多模型管理器
 * ⚠️ 当前设计：**所有模型均为独立 bmodel 文件**（每个文件一个独立 bmrt 上下文，
 * 见 SModel），**不存在融合模型**（单文件多子网混合 bmodel 不在支持范围）。
 * 一个 bmodel 内的"多子网"（如 qwen3_asr 的 embedding/block_i/.../audio）是
 * 转换产物自身的子网划分，跨模型（asr/spk/vad/summary）之间无任何共享。
 * 各模型独立加载/卸载（unloadModel/loadModel），显存占用 = Σ(常驻模型)。
 */
class CModelHub {
public:
    CModelHub() = default;
    ~CModelHub();

    CModelHub(const CModelHub &) = delete;
    CModelHub &operator=(const CModelHub &) = delete;

    /** 请求设备并加载 JSON "models" 声明的全部 bmodel（每个 bmodel 独立 bmrt
     *  上下文，无融合模型）；返回错误描述，空=成功 */
    std::string init(const class JsonValue *models_cfg, int dev_id);
    void deinit();

    bm_handle_t handle() const { return m_handle; }
    bool hasModel(const std::string &name) const {
        std::shared_lock<std::shared_mutex> lk(m_mtx);
        return m_models.count(name) > 0;
    }
    int modelCount() const {
        std::shared_lock<std::shared_mutex> lk(m_mtx);
        return (int)m_models.size();
    }
    /** 取模型对象（裸指针）。调用方不得跨 unload/load 持有它 —— 见文件头"线程模型" */
    SModel *model(const std::string &name);
    /** 取模型对象的共享引用（**推荐**，裸指针的安全替代）：锁内拷贝 shared_ptr，
     *  调用方持有期间 SModel 对象不会被析构 —— 这正是 model() 无法保证的一点。
     *  注意：只保证对象存活，不保证 bmrt 仍有效（unload 会销毁 bmrt 句柄），
     *        因此用前仍须按上层协议确认目标模型处于已加载状态。 */
    std::shared_ptr<SModel> modelRef(const std::string &name) const;

    /** 按角色取第一个匹配子网名（在 model 内）；找不到返回 "" */
    std::string netByRole(const std::string &model, const std::string &role) const;
    /** 按角色反查模型名（跨全部已加载模型）：返回首个含该角色子网的模型名，找不到返回 ""。
     *  这是"模型名 ↔ 角色"的唯一权威来源 —— Link 定位模型必须走它，不要各写一套键名规则。
     *  注意：unloadModel 会清空 role_of_net（模型保留、loaded=false），故卸载后本函数查不到，
     *        调用方应在 init 期解析一次并持有模型名。 */
    std::string modelByRole(const std::string &role) const;
    const bm_net_info_t *netInfo(const std::string &model, const std::string &net) const;

    /** 子网启停标记（Ctrl enable/disable 维护 disabled 集合，list_nets 可查） */
    void setNetEnabled(const std::string &model, const std::string &net, bool enable);

    /** 按模型真实卸载/重载（显存受限场景：总结时卸 asr，总结后重载）。
     *  bm_handle 保留；bmrt 销毁后该模型全部 net_info 指针失效，
     *  调用方须保证相关 Link 已停止使用（asr: onModelUnload 联动）。 */
    int unloadModel(const std::string &name, std::string &err);
    int loadModel(const std::string &name, std::string &err);
    bool modelLoaded(const std::string &name) const;

    /** 全部模型枚举信息（list_nets Ctrl 用） */
    std::string dumpNets() const;

    /** qwen3_asr 维度推导：对含 asr_llm/asr_audio 角色的模型调用一次 */
    /* 按值返回（内部持共享锁）：避免返回引用后锁已释放、与 deriveAsrDims 写竞争 */
    AsrDims asrDims() const {
        std::shared_lock<std::shared_mutex> lk(m_mtx);
        return m_asr_dims;
    }
    std::string deriveAsrDims(const std::string &model, const JsonValue *dims_cfg);

private:
    /** 无锁版本，仅供已持锁的成员函数内部调用（shared_mutex 不可重入）：
     *  deinit() 加锁后转调它；init() 加载失败回滚也直接用它（此时已持独占锁）。 */
    void deinitLocked();
    /** 无锁版本，仅供已持锁的成员函数内部调用（shared_mutex 不可重入） */
    SModel *modelLocked(const std::string &name) const;
    const bm_net_info_t *netInfoLocked(const std::string &model, const std::string &net) const;
    std::string netByRoleLocked(const std::string &model, const std::string &role) const;

    bm_handle_t m_handle = nullptr;
    bool m_handle_ok = false;
    /* 保护 m_models / m_asr_dims：读接口共享锁、写接口独占锁 */
    mutable std::shared_mutex m_mtx;
    std::map<std::string, std::shared_ptr<SModel>> m_models;
    /* 加载代次发号器：仅在持独占锁的 loadOne 内 ++（SModel::gen 的来源） */
    uint32_t m_gen = 0;
    AsrDims m_asr_dims;
    std::string m_last_err;

    std::string loadOne(const std::string &name, const JsonValue *cfg);
    static bool matchKeywords(const std::string &net_name, const std::string &keywords);
};

} /* namespace al */
#endif /* AL_MODEL_HUB_H */
