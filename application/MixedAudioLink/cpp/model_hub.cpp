//===----------------------------------------------------------------------===//
//
// model_hub.cpp - CModelHub 实现
// 加载流程对齐 sample/CAM++/cpp/main.cpp 与 LLM-TPU Qwen3_ASR chat.cpp init/init_by_names
//
//===----------------------------------------------------------------------===//
#include "model_hub.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>

#include "al_json.h"
#include "al_log.h"
#include "audio_link.h"   /* E_AL_* 状态码 */

namespace al {

CModelHub::~CModelHub() { deinit(); }

std::string CModelHub::init(const JsonValue *models_cfg, int dev_id) {
    std::unique_lock<std::shared_mutex> lk(m_mtx);   /* init 期独占（正常为单线程，防误用） */
    /* 所有模型均为独立 bmodel 文件：每个键 → bmrt_create + bmrt_load_bmodel，
     * 一个独立 bmrt 上下文；无融合模型（单文件多子网混合 bmodel 不支持）。
     * 显存占用 = Σ(各模型权重 + KV cache)，可经 unloadModel/loadModel 换入换出 */
    if (!models_cfg || !models_cfg->isObj())
        return "models config missing";

    bm_status_t st = bm_dev_request(&m_handle, dev_id);
    if (BM_SUCCESS != st) {
        m_handle_ok = false;
        return "bm_dev_request(dev " + std::to_string(dev_id) + ") failed: " + std::to_string(st);
    }
    m_handle_ok = true;

    for (auto &kv : models_cfg->obj) {
        std::string err = loadOne(kv.first, kv.second.get());
        if (!err.empty()) {
            /* 部分加载失败必须立即回滚：否则已成功的 bmodel（asr 可达 3.81GB）
             * 会一直占着显存与 bmrt 上下文，调用方重试时 bm_dev_request 再次申请，
             * 表现为"换个模型就起不来"的假故障。此处已持独占锁，走无锁版本。 */
            AL_LOG_E("model_hub", "init failed on '%s': %s -> rollback all loaded models",
                     kv.first.c_str(), err.c_str());
            deinitLocked();
            return err;
        }
    }
    return "";
}

void CModelHub::deinit() {
    std::unique_lock<std::shared_mutex> lk(m_mtx);
    deinitLocked();
}

/** 无锁版本：调用方必须已持 m_mtx（shared_mutex 不可重入） */
void CModelHub::deinitLocked() {
    for (auto &kv : m_models) {
        if (kv.second->bmrt) bmrt_destroy(kv.second->bmrt);
        kv.second->bmrt = nullptr;
    }
    m_models.clear();
    if (m_handle_ok) {
        bm_dev_free(m_handle);
        m_handle_ok = false;
    }
}

/*---------- 无锁内部实现（调用方必须已持 m_mtx，shared 或 unique 均可） ----------*/

SModel *CModelHub::modelLocked(const std::string &name) const {
    auto it = m_models.find(name);
    return it == m_models.end() ? nullptr : it->second.get();
}

const bm_net_info_t *CModelHub::netInfoLocked(const std::string &model, const std::string &net) const {
    auto it = m_models.find(model);
    if (it == m_models.end()) return nullptr;
    auto n = it->second->nets.find(net);
    return n == it->second->nets.end() ? nullptr : n->second;
}

std::string CModelHub::netByRoleLocked(const std::string &model, const std::string &role) const {
    auto it = m_models.find(model);
    if (it == m_models.end()) return "";
    for (auto &n : it->second->net_names) {
        auto r = it->second->role_of_net.find(n);
        if (r != it->second->role_of_net.end() && r->second == role) return n;
    }
    return "";
}

/*---------------------------- 带锁公开接口 ----------------------------*/

SModel *CModelHub::model(const std::string &name) {
    std::shared_lock<std::shared_mutex> lk(m_mtx);
    return modelLocked(name);
}

std::shared_ptr<SModel> CModelHub::modelRef(const std::string &name) const {
    std::shared_lock<std::shared_mutex> lk(m_mtx);
    auto it = m_models.find(name);
    /* 返回 shared_ptr 拷贝：即使随后另一线程 loadModel 替换了 map 里的对象，
     * 调用方手上的这份仍有效（对象存活由引用计数保证）。 */
    return it == m_models.end() ? nullptr : it->second;
}

bool CModelHub::matchKeywords(const std::string &net_name, const std::string &keywords) {
    /* 关键字按 '|' 分隔，子串命中即匹配 */
    size_t start = 0;
    while (start <= keywords.size()) {
        size_t end = keywords.find('|', start);
        if (end == std::string::npos) end = keywords.size();
        if (end > start) {
            std::string kw = keywords.substr(start, end - start);
            if (!kw.empty() && net_name.find(kw) != std::string::npos) return true;
        }
        start = end + 1;
    }
    return false;
}

std::string CModelHub::loadOne(const std::string &name, const JsonValue *cfg) {
    if (!cfg || !cfg->isObj()) return "models." + name + " is not an object";
    std::string path = cfg->getStr("path");
    if (path.empty()) return "models." + name + ".path missing";

    auto m = std::make_shared<SModel>();
    m->name = name;
    m->path = path;
    m->cfg = cfg;

    /* 加载进度日志（P0 排障）：文件大小 → bmrt_create → load_bmodel 耗时 → 就绪 */
    long long file_mb = 0;
    {
        std::error_code ec;
        auto sz = std::filesystem::file_size(path, ec);
        if (!ec) file_mb = (long long)(sz / (1024 * 1024));
    }
    AL_LOG_I("model_hub", "load model '%s' (%s, %lld MB) begin", name.c_str(),
             path.c_str(), file_mb);

    m->bmrt = bmrt_create(m_handle);
    if (!m->bmrt) return "bmrt_create failed for " + name;
    /* board 判断省略：BM_RUNTIME_SHARE_MEM 仅 1688/CV186 需要，1684X 不设置 */
    auto t0 = std::chrono::steady_clock::now();
    if (!bmrt_load_bmodel(m->bmrt, path.c_str())) {
        bmrt_destroy(m->bmrt);   /* 失败路径释放已创建的 bmrt */
        m->bmrt = nullptr;
        return "bmrt_load_bmodel(" + path + ") failed for " + name;
    }
    long long load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();

    /* 动态枚举全部子网 */
    int num = bmrt_get_network_number(m->bmrt);
    const char **names = nullptr;
    bmrt_get_network_names(m->bmrt, &names);
    for (int i = 0; i < num; i++) {
        m->net_names.emplace_back(names[i]);
        m->nets[names[i]] = bmrt_get_network_info(m->bmrt, names[i]);
    }
    free(names);
    AL_LOG_I("model_hub", "model '%s' loaded in %lld ms, %d nets", name.c_str(),
             load_ms, num);

    /* JSON roles: 角色 → 关键字串，反向映射 子网 → 角色 */
    auto roles = cfg->get("roles");
    if (roles && roles->isObj()) {
        for (auto &rk : roles->obj) {
            if (!rk.second || !rk.second->isStr()) continue;
            for (auto &n : m->net_names) {
                if (matchKeywords(n, rk.second->str)) m->role_of_net[n] = rk.first;
            }
        }
    }
    m->loaded = true;
    /* 代次发号：必须在替换 map 之前 —— 之后旧 SModel 可能随引用计数归零而析构，
     * 调用方据此判断自己缓存的 net_info/mems 指针是否已随旧 bmrt 失效。
     * （loadOne 全程在 init/loadModel 持有的独占锁下执行，++ 无需额外加锁） */
    m->gen = ++m_gen;
    m_models[name] = m;
    /* 角色映射结果入日志（roles 未命中可在此直接发现） */
    std::string role_str;
    if (roles && roles->isObj()) {
        for (auto &rk : roles->obj) {
            std::string hit = "(miss)";
            for (auto &p : m->role_of_net)
                if (p.second == rk.first) { hit = p.first; break; }
            role_str += (role_str.empty() ? "" : ",") + rk.first + "->" + hit;
        }
    }
    AL_LOG_I("model_hub", "model '%s' ready: roles=[%s]", name.c_str(), role_str.c_str());
    return "";
}

/*=================== 按模型卸载/重载（显存受限场景） ====================*/

int CModelHub::unloadModel(const std::string &name, std::string &err) {
    std::unique_lock<std::shared_mutex> lk(m_mtx);
    auto m = modelLocked(name);
    if (!m) { err = "model not found: " + name; return E_AL_LINK_NOT_FOUND; }
    if (!m->loaded) { err = "model already unloaded: " + name; return E_AL_PARAM_ERR; }
    if (m->bmrt) {
        bmrt_destroy(m->bmrt);
        m->bmrt = nullptr;
    }
    m->nets.clear();
    m->net_names.clear();
    m->role_of_net.clear();
    m->loaded = false;   /* name/path/cfg/disabled 保留供重载 */
    return E_AL_OK;
}

int CModelHub::loadModel(const std::string &name, std::string &err) {
    std::unique_lock<std::shared_mutex> lk(m_mtx);
    auto m = modelLocked(name);
    if (!m) { err = "model not found: " + name; return E_AL_LINK_NOT_FOUND; }
    if (m->loaded) return E_AL_OK;
    /* 关键：loadOne 会用新 SModel 替换 map 里的 shared_ptr，旧对象可能随之析构，
     * 故必须在重建之前把要延续的状态拷出来 —— 否则下面的 m 已是悬垂指针（UAF）。 */
    std::set<std::string> disabled = m->disabled;
    const JsonValue *cfg = m->cfg;
    /* 角色映射增量更新：按原 JSON 配置重建 */
    std::string e = loadOne(name, cfg);
    if (!e.empty()) { err = e; return E_AL_MODEL_LOAD_FAIL; }
    /* 恢复 disabled 集合（loadOne 重建了 SModel） */
    auto nm = modelLocked(name);
    if (nm) nm->disabled = disabled;
    return E_AL_OK;
}

bool CModelHub::modelLoaded(const std::string &name) const {
    std::shared_lock<std::shared_mutex> lk(m_mtx);
    auto it = m_models.find(name);
    return it != m_models.end() && it->second->loaded;
}

std::string CModelHub::netByRole(const std::string &model, const std::string &role) const {
    std::shared_lock<std::shared_mutex> lk(m_mtx);
    return netByRoleLocked(model, role);
}

std::string CModelHub::modelByRole(const std::string &role) const {
    std::shared_lock<std::shared_mutex> lk(m_mtx);
    if (role.empty()) return "";
    /* m_models 为 std::map（按键名有序），多模型命中同一角色时结果稳定 */
    for (auto &kv : m_models) {
        if (!kv.second) continue;
        for (auto &rn : kv.second->role_of_net)
            if (rn.second == role) return kv.first;
    }
    return "";
}

const bm_net_info_t *CModelHub::netInfo(const std::string &model, const std::string &net) const {
    std::shared_lock<std::shared_mutex> lk(m_mtx);
    return netInfoLocked(model, net);
}

void CModelHub::setNetEnabled(const std::string &model, const std::string &net, bool enable) {
    std::unique_lock<std::shared_mutex> lk(m_mtx);
    auto m = modelLocked(model);
    if (!m) return;
    if (enable) m->disabled.erase(net);
    else m->disabled.insert(net);
}

std::string CModelHub::dumpNets() const {
    std::shared_lock<std::shared_mutex> lk(m_mtx);
    std::ostringstream oss;
    oss << "{\"models\":[";
    bool first_m = true;
    for (auto &kv : m_models) {
        if (!first_m) oss << ",";
        first_m = false;
        oss << "{\"name\":\"" << kv.first << "\",\"nets\":[";
        bool first_n = true;
        for (auto &n : kv.second->net_names) {
            if (!first_n) oss << ",";
            first_n = false;
            oss << "{\"name\":\"" << n << "\",\"role\":\"";
            auto r = kv.second->role_of_net.find(n);
            oss << (r == kv.second->role_of_net.end() ? "" : r->second) << "\",\"enabled\":"
                << (kv.second->disabled.count(n) ? "false" : "true") << "}";
        }
        oss << "]}";
    }
    oss << "]}";
    return oss.str();
}

/*=================== qwen3_asr 维度推导（chat.cpp init_by_names 移植） ===================*/

std::string CModelHub::deriveAsrDims(const std::string &model, const JsonValue *dims_cfg) {
    /* 全程独占锁：既写 m_asr_dims，也要保证读取的 net_info 指针在推导期间不被
     * 并发 unload/load 抽走（内部一律走 *Locked 变体，shared_mutex 不可重入） */
    std::unique_lock<std::shared_mutex> lk(m_mtx);
    auto m = modelLocked(model);
    if (!m) return "model not found: " + model;

    AsrDims d;
    const bm_net_info_t *net_embed = netInfoLocked(model, "embedding");
    const bm_net_info_t *net_embed_cache = netInfoLocked(model, "embedding_cache");
    const bm_net_info_t *net_lm = netInfoLocked(model, "lm_head");
    if (!net_embed || !net_embed_cache || !net_lm)
        return "asr model missing embedding/embedding_cache/lm_head nets";

    int num_nets = (int)m->net_names.size();
    auto exist = [&](const std::string &n) { return m->nets.count(n) > 0; };

    /* audio 子网可选：text-only LLM（总结模型）无 audio */
    const bm_net_info_t *net_audio = netInfoLocked(model, netByRoleLocked(model, "asr_audio").empty() ? "audio" : netByRoleLocked(model, "asr_audio"));
    bool has_audio = net_audio != nullptr;
    int fixed_nets = has_audio ? 4 : 3; /* embed, embedding_cache, [audio], lm_head */

    int num_blocks = num_nets - fixed_nets;
    if (exist("greedy_head")) num_blocks--;
    if (exist("sample_head")) num_blocks--;
    d.NUM_LAYERS = num_blocks / 2;

    for (int i = 0; i < d.NUM_LAYERS; i++) {
        if (!exist("block_" + std::to_string(i)) || !exist("block_cache_" + std::to_string(i))) {
            d.NUM_LAYERS = i;
            break;
        }
    }
    d.lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
    d.MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
    d.HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
    const bm_net_info_t *bc0 = netInfoLocked(model, "block_cache_0");
    d.SEQLEN = bc0 ? bc0->stages[0].input_shapes[3].dims[1] : 0;
    if (has_audio) {
        /* 业务维度一律取**末尾两维**，勿写死 dims[1]/dims[3]：audio 子网输入带
         * batch/channel（unsqueeze）等前导维，实测 Qwen3-ASR 签名
         * input_states [1,1,128,100] → mel = dims[n-2] = 128、每块帧数 = dims[n-1] = 100。
         * 写死 dims[1] 会取到 1，令 mel 前端与 host 缓冲步进整体错位（含越界读）。 */
        const bm_shape_t &ash = net_audio->stages[0].input_shapes[0];
        int nd = ash.num_dims;
        d.AUDIO_MEL_BINS = (nd >= 2) ? ash.dims[nd - 2] : 0;
        d.AUDIO_LENGTH = (nd >= 1) ? ash.dims[nd - 1] : 0;
    } else {
        d.AUDIO_MEL_BINS = 0;
        d.AUDIO_LENGTH = 0;
    }
    d.KV_BYTES = bc0 ? (int)bm_mem_get_device_size(bc0->stages[0].output_mems[1]) : 0;

    if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) d.mask_value = 0xF0E2;
    else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) d.mask_value = 0xC61C;
    else return "unsupported attention dtype (need fp16/bf16)";

    const bm_net_info_t *b0 = netInfoLocked(model, "block_0");
    d.support_history = b0 && b0->input_num == 5; /* 带 kv cache 输入 */
    d.PREFILL_KV_LENGTH = (d.support_history && b0) ? b0->stages[0].input_shapes[3].dims[1] : 0;
    d.is_dynamic = b0 && b0->is_dynamic;
    d.prefill_mask = b0 && b0->input_num > 2;
    d.valid = true;
    m_asr_dims = d;

    /* JSON dims 校验/回退（"auto" 或数值不符时以推导值为准并告警） */
    std::ostringstream warn;
    if (dims_cfg && dims_cfg->isObj()) {
        struct { const char *key; int actual; } checks[] = {
            {"seq_len", d.SEQLEN}, {"max_input_length", d.MAX_INPUT_LENGTH},
            {"hidden_size", d.HIDDEN_SIZE}, {"num_layers", d.NUM_LAYERS},
            {"audio_length", d.AUDIO_LENGTH},
        };
        for (auto &c : checks) {
            auto v = dims_cfg->get(c.key);
            if (v && v->isNum() && (int)v->num != c.actual)
                warn << "dims." << c.key << "=" << (int)v->num << " != derived " << c.actual << "; ";
        }
    }
    return warn.str();
}

} /* namespace al */
