//===----------------------------------------------------------------------===//
//
// model_hub.h - CModelHub：多 bmodel 多上下文管理
// 每个 bmodel 文件一个 bmrt 上下文；动态枚举子网；JSON 关键字映射角色；
// 维度自动推导器（init_by_names 模式，JSON 仅作回退）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_MODEL_HUB_H
#define AL_MODEL_HUB_H

#include <cstdint>
#include <map>
#include <memory>
#include <set>
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
    int AUDIO_MEL_BINS = 0;    /* audio 子网输入 dims[1]（128） */
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
    bool hasModel(const std::string &name) const { return m_models.count(name) > 0; }
    int modelCount() const { return (int)m_models.size(); }
    SModel *model(const std::string &name);

    /** 按角色取第一个匹配子网名（在 model 内）；找不到返回 "" */
    std::string netByRole(const std::string &model, const std::string &role) const;
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
    const AsrDims &asrDims() const { return m_asr_dims; }
    std::string deriveAsrDims(const std::string &model, const JsonValue *dims_cfg);

private:
    bm_handle_t m_handle = nullptr;
    bool m_handle_ok = false;
    std::map<std::string, std::shared_ptr<SModel>> m_models;
    AsrDims m_asr_dims;
    std::string m_last_err;

    std::string loadOne(const std::string &name, const JsonValue *cfg);
    static bool matchKeywords(const std::string &net_name, const std::string &keywords);
};

} /* namespace al */
#endif /* AL_MODEL_HUB_H */
