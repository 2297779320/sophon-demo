//===----------------------------------------------------------------------===//
//
// link_mng.cpp - CBaseLink / CLinkMng（通用管理器基座）实现
// 音频工作流专属逻辑（默认链路/双线程/pushPcm/音频 Ctrl/音频校验/快照音频段）
// 已下沉至 CAudioLinkMng（audio_link_mng.cpp），本文件仅保留通用骨架。
//
//===----------------------------------------------------------------------===//
#include "link_mng.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>

#include "al_json.h"
#include "al_log.h"
#include "al_out.h"   /* alCopyOut：出参写入统一查截断 */
#include "al_path.h"

namespace al {

/*============================ CBaseLink ============================*/

CBaseLink::CBaseLink(CLinkMng *mng, std::string name, std::string type)
    : m_mng(mng), m_name(std::move(name)), m_type(std::move(type)) {}

CBaseLink::~CBaseLink() = default;

int CBaseLink::setSource(CBaseLink *src) {
    if (!src || src == this) return E_AL_PARAM_ERR;
    m_src = src;
    return E_AL_OK;
}

int CBaseLink::pullSource(FramePtr &out) {
    if (!m_src) return E_AL_LINK_NOT_FOUND;
    return m_src->getFrame(out);
}

FramePtr CBaseLink::acquireFrame() { return m_mng->frameAcquire(); }
const JsonValue *CBaseLink::cfg() const { return m_mng->rootCfg(); }

std::shared_ptr<SModel> CBaseLink::acquireModel(const std::string &model, uint32_t gen) {
    /* modelRef 在锁内拷贝 shared_ptr：即便随后另一线程 loadModel 替换了 map 里的
     * SModel（旧对象析构），手中这份仍可安全读取其成员。 */
    auto m = m_mng->hub().modelRef(model);
    if (!m || !m->loaded || !m->bmrt) return nullptr;   /* 已卸载：bmrt 已 destroy */
    if (gen != 0 && m->gen != gen) return nullptr;      /* 已重载：旧 net_info/mems 失效 */
    return m;
}

int CBaseLink::ctrl(const std::string &cmd, AL_PARAM param) {
    (void)cmd;
    (void)param;
    return E_AL_PARAM_ERR; /* 基类不接受任何命令 */
}

/*============================ 工作流虚钩子（默认实现） ============================*/

std::string CLinkMng::validateAudio(bool after_hub) {
    (void)after_hub;
    return "";
}

int CLinkMng::createLinks() { return E_AL_OK; }

int CLinkMng::startWorkers() { return E_AL_OK; }

void CLinkMng::stopWorkers() {}

int CLinkMng::onCtrl(const std::string &name, const std::string &cmd, AL_PARAM param) {
    (void)name;
    (void)cmd;
    (void)param;
    return E_AL_PARAM_ERR;
}

void CLinkMng::appendStats(std::string &js) { (void)js; }

void CLinkMng::appendStatusMd(std::string &md) { (void)md; }

CBaseLink *CLinkMng::createLinkImpl(const std::string &type, const std::string &name) {
    (void)type;
    (void)name;
    return nullptr;   /* 基座不注册任何 Link 工厂分支，由工作流子类提供 */
}

int CLinkMng::pushPcm(const void *pcm, int bytes) {
    (void)pcm;
    (void)bytes;
    return setLastError("pushPcm: not supported by this workflow"), E_AL_NOT_INIT;
}

int CLinkMng::pushPcmEx(const void *pcm, int bytes, int rate, int ch, int bits) {
    (void)pcm;
    (void)bytes;
    (void)rate;
    (void)ch;
    (void)bits;
    return setLastError("pushPcmEx: not supported by this workflow"), E_AL_NOT_INIT;
}

/*============================ 对象池 ============================*/

FramePtr CLinkMng::frameAcquire() {
    std::lock_guard<std::mutex> lk(m_pool_mtx);
    ALFrame *raw;
    if (!m_pool.empty()) {
        raw = m_pool.back();
        m_pool.pop_back();
    } else {
        if (m_pool_size >= AL_POOL_MAX_FRAMES) return FramePtr();   /* 池满：背压（空 FramePtr） */
        raw = new ALFrame();
        m_pool_size++;
    }
    /* shared_ptr 归还时自动回池 */
    return FramePtr(raw, [this](ALFrame *f) { frameRelease(f); });
}

void CLinkMng::frameRelease(ALFrame *f) {
    if (!f) return;
    f->pcm.clear();
    f->start_ms = f->end_ms = 0;
    f->is_final = false;
    std::lock_guard<std::mutex> lk(m_pool_mtx);
    m_pool.push_back(f);
}

/*============================ 任务队列 ============================*/

void CLinkMng::pushSegment(FramePtr seg) {
    {
        std::lock_guard<std::mutex> lk(m_seg_mtx);
        m_seg_queue.push_back(std::move(seg));
    }
    m_seg_cv.notify_one();
}

bool CLinkMng::popSegment(FramePtr &seg, int timeout_ms) {
    std::unique_lock<std::mutex> lk(m_seg_mtx);
    if (!m_seg_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                           [&] { return !m_seg_queue.empty(); }))
        return false;
    if (m_task_pause.load()) return false; /* 卸载闸门：暂停期间不取新段、不置忙 */
    seg = std::move(m_seg_queue.front());
    m_seg_queue.pop_front();
    /* 持锁置位：unload_model 等待方在 m_seg_mtx 下轮询同一条件，
     * 消除"队列已空但任务尚未置忙"的 TOCTOU 窗口 */
    m_task_busy = true;
    return true;
}

/*============================ 初始化 ============================*/

static std::string readFileAll(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return "";
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
    fclose(fp);
    return out;
}

/* 配置文件存在性测试（fopen "rb" 成功即认为存在） */
bool CLinkMng::configFileExists(const std::string &path) {
    if (path.empty()) return false;
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

/* 数值范围检查：键缺失/非数值跳过（走各 Link 默认值）；越界返回错误描述 */
std::string CLinkMng::rangeCheck(const JsonValue *sec, const char *sec_name, const char *key,
                                 double lo, double hi, bool lo_open, bool hi_open) {
    auto v = sec ? sec->get(key) : nullptr;
    if (!v || !v->isNum()) return "";
    double d = v->num;
    bool ok = (lo_open ? d > lo : d >= lo) && (hi_open ? d < hi : d <= hi);
    if (ok) return "";
    return std::string(sec_name) + "." + key + ": expect " +
           (lo_open ? "(" : "[") + std::to_string(lo) + "," + std::to_string(hi) +
           (hi_open ? ")" : "]") + ", got " + std::to_string(d);
}

std::string CLinkMng::ValidateConfig(bool after_hub) {
    const JsonValue *root = m_cfg.get();
    if (!root) return "config: not parsed";

    if (!after_hub) {
        /*---- 第一段（通用）：必需键 / 文件存在 ----*/
        auto models = root->get("models");
        if (!models || !models->isObj() || models->obj.empty())
            return "models: required object missing";
        for (auto &kv : models->obj) {
            std::string path = kv.second ? kv.second->getStr("path") : "";
            if (path.empty())
                return "models." + kv.first + ".path: required key missing";
            if (!configFileExists(resolveModelPath(path)))
                return "models." + kv.first + ".path: file not found: " + resolveModelPath(path);
        }

        /* 工作流专属键校验（必需键/数值范围，子类钩子） */
        std::string e = validateAudio(false);
        /* 通用：log 级别范围 */
        if (e.empty()) e = rangeCheck(root->get("log").get(), "log", "level", -1, 3, false, false);
        if (!e.empty()) return e;
    } else {
        /*---- 第二段（通用）：各模型 roles 逐角色命中子网（hub.init 后）----*/
        auto models = root->get("models");
        if (models && models->isObj()) {
            for (auto &kv : models->obj) {
                auto roles = kv.second ? kv.second->get("roles") : nullptr;
                if (!roles || !roles->isObj()) continue;
                for (auto &rk : roles->obj) {
                    if (m_hub.netByRole(kv.first, rk.first).empty())
                        return kv.first + ": no net matched role '" + rk.first +
                               "' (check roles keywords)";
                }
            }
        }
        /* 工作流专属第二段校验（钩子，默认无） */
        std::string e = validateAudio(true);
        if (!e.empty()) return e;
    }
    return "";
}

int CLinkMng::init(const char *json_cfg, int dev_id) {
    auto t_init0 = std::chrono::steady_clock::now();   /* 加载总耗时统计 */
    if (!json_cfg) return setLastError("init: json_cfg is NULL"), E_AL_PARAM_ERR;
    std::string text = readFileAll(json_cfg);
    if (text.empty())
        return setLastError("config read failed: " + std::string(json_cfg)),
               logErr("config read failed: " + std::string(json_cfg)), E_AL_PARAM_ERR;

    std::string err;
    m_cfg = JsonParser::parse(text, err);
    if (!m_cfg) return setLastError("config parse: " + err), logErr("config parse: " + err), E_AL_PARAM_ERR;

    /* 记下启动配置路径：ValidateConfig 的资产路径解析（resolveAsset 以此为基准）
     * 即需可用，亦为 Ctrl "save_params" 原子写回用 */
    strCfgPath = json_cfg;

    /* 日志配置（在模型加载前应用，加载期日志即可入文件）：
     * log.level = 0..3（默认 2=INFO），log.file = 空串走 stderr，log.max_bytes 轮转阈值 */
    if (auto lg = m_cfg->get("log")) {
        al_log_set_level((int)lg->getNum("level", AL_LOG_INFO));
        std::string log_file = lg->getStr("file", "");
        long long max_bytes = (long long)lg->getNum("max_bytes", 8388608.0);
        /* 打开失败时 al_log_set_file 内部保持 stderr，不中断初始化 */
        al_log_set_file(log_file.empty() ? nullptr : log_file.c_str(), max_bytes);
    }

    /* 配置全量校验第一段（必需键/文件存在/工作流专属键，validateAudio 钩子参与）：
     * strict=true 校验失败拒绝启动；false 仅告警并沿用默认/钳制值继续 */
    std::string verr = ValidateConfig(false);
    if (!verr.empty()) {
        if (m_cfg->getBool("strict", false)) {
            std::string msg = "config validate failed (strict): " + verr;
            return setLastError(msg), logErr(msg), E_AL_PARAM_ERR;
        }
        logWarn("config validate: " + verr + " (non-strict, continue with defaults)");
    }

    /* models.*.path 相对路径以**应用根（启动 CWD）**为基准解析（resolveModelPath），
     * 与配置资产（resolveAsset，JSON 所在目录）区分开：模型随应用部署，资产随配置走 */
    std::string hub_err = m_hub.init(m_cfg->get("models").get(), dev_id);
    if (!hub_err.empty()) return setLastError(hub_err), logErr(hub_err), E_AL_MODEL_LOAD_FAIL;

    /* 配置全量校验第二段（roles 逐角色命中子网） */
    verr = ValidateConfig(true);
    if (!verr.empty()) {
        if (m_cfg->getBool("strict", false)) {
            std::string msg = "config validate failed (strict): " + verr;
            /* roles 未命中属模型/子网不匹配，语义上 NET_NOT_MATCH 更准确 */
            return setLastError(msg), logErr(msg), E_AL_NET_NOT_MATCH;
        }
        logWarn("config validate: " + verr + " (non-strict, continue)");
    }

    if (!m_cfg->get("models") || !m_cfg->get("models")->isObj())
        return setLastError("config missing 'models'"), logErr("config missing 'models'"), E_AL_PARAM_ERR;

    /* 工作流默认链路创建（子类钩子：内部已 setLastError，失败原样上抛） */
    int rc = createLinks();
    if (rc != E_AL_OK) return rc;
    long long init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t_init0).count();
    log("init done in " + std::to_string(init_ms) + " ms, " +
        std::to_string(m_links.size()) + " links, models=" +
        std::to_string(m_hub.modelCount()));
    return E_AL_OK;
}

int CLinkMng::createLink(const std::string &name, const std::string &type, AL_PARAM param) {
    if (name.empty() || type.empty())
    {
        return setLastError("createLink: name/type empty"), E_AL_PARAM_ERR;
    }

    /* 工厂分支由工作流子类 createLinkImpl 提供（基座通用化） */
    std::unique_ptr<CBaseLink> link(createLinkImpl(type, name));
    if (!link)
        return setLastError("createLink: unknown type '" + type + "'"), E_AL_PARAM_ERR;

    int rc = link->create(m_cfg.get(), param);
    if (rc != E_AL_OK)
        return setLastError("createLink '" + name + "' failed: rc=" + std::to_string(rc)), rc;

    std::lock_guard<std::mutex> lk(m_links_mtx);
    if (m_links.count(name))
        return setLastError("createLink: name '" + name + "' already exists"), E_AL_PARAM_ERR;


    m_links[name] = std::move(link);
    return E_AL_OK;
}

int CLinkMng::bind(const std::string &src, const std::string &dst) {
    CBaseLink *s = findLink(src), *d = findLink(dst);
    if (!s || !d)
        return setLastError("bind: link not found: '" + src + "' -> '" + dst + "'"),
               E_AL_LINK_NOT_FOUND;

    int rc = d->setSource(s);
    if (rc != E_AL_OK)
    {
        setLastError("bind '" + src + "' -> '" + dst + "' failed");
    }

    return rc;
}

int CLinkMng::start() {
    if (m_running.load()) return E_AL_OK;
    m_start_time = std::chrono::steady_clock::now();   /* 运行时长基准（退出快照用） */

    /* 逐个检查返回码：任一 link 启动失败都要回滚已启动的，
     * 否则会进入"工作线程已跑、链路半就绪"的状态，故障被静默吞掉。 */
    std::vector<std::string> started;
    int rc = E_AL_OK;
    {
        std::lock_guard<std::mutex> lk(m_links_mtx);
        for (auto &kv : m_links) {
            int r = kv.second->start();
            if (r != E_AL_OK) {
                rc = r;
                setLastError("link '" + kv.first + "' start failed, rc=" + std::to_string(r));
                logErr("link '" + kv.first + "' start failed, rc=" + std::to_string(r));
                break;
            }
            started.push_back(kv.first);
        }
    }
    if (rc != E_AL_OK) {
        rollbackStarted(started);
        return rc;   /* m_running 保持 false：未启动成功 */
    }

    m_running = true;
    rc = startWorkers();   /* 工作线程启动（子类钩子） */
    if (rc != E_AL_OK) {
        setLastError("startWorkers failed, rc=" + std::to_string(rc));
        logErr("startWorkers failed, rc=" + std::to_string(rc));
        rollbackStarted(started);
        m_running = false;
        return rc;
    }
    return E_AL_OK;
}

std::string CLinkMng::modelNameByRole(const std::string &role, const std::string &fallbackKey) const {
    std::string n = m_hub.modelByRole(role);
    if (n.empty() && m_hub.hasModel(fallbackKey)) n = fallbackKey;
    return n;
}

std::string CLinkMng::asrModelName() const {
    return modelNameByRole("asr_llm", "asr");
}

/** start() 失败回滚：逆序停掉已启动的 link（不含失败者本身，它没起来） */
void CLinkMng::rollbackStarted(const std::vector<std::string> &started) {
    std::lock_guard<std::mutex> lk(m_links_mtx);
    for (auto it = started.rbegin(); it != started.rend(); ++it) {
        auto kv = m_links.find(*it);
        if (kv != m_links.end()) kv->second->stop();
    }
}

int CLinkMng::stop() {
    if (!m_running.load()) return E_AL_OK;
    dumpStatusOnStop();   /* 链路仍存活、统计最完整时写状态快照（配置了才落盘） */
    m_running = false;
    m_seg_cv.notify_all();
    stopWorkers();   /* 工作线程 teardown（子类钩子 join） */
    std::lock_guard<std::mutex> lk(m_links_mtx);
    for (auto &kv : m_links) kv.second->stop();
    return E_AL_OK;
}

void CLinkMng::teardownThreads() {
    if (m_vad_thread.joinable()) m_vad_thread.join();
    if (m_task_thread.joinable()) m_task_thread.join();
}

void CLinkMng::deinit() {
    stop();
    {
        std::lock_guard<std::mutex> lk(m_seg_mtx);
        m_seg_queue.clear();   /* 先清队列，避免析构期 FramePtr 归还竞争 */
    }
    {
        std::lock_guard<std::mutex> lk(m_pool_mtx);
        for (ALFrame *f : m_pool) delete f;   /* 释放对象池全部帧 */
        m_pool.clear();
        m_pool_size = 0;
    }
    std::lock_guard<std::mutex> lk(m_links_mtx);
    m_links.clear();
    m_hub.deinit();
}

/*============================ 数据通路 ============================*/

/* pushPcm/pushPcmEx 为工作流专属数据口：基座默认不支持（子类 CAudioLinkMng 覆写），
 * 默认实现见上文"工作流虚钩子"段 */

/*============================ Ctrl ============================*/

int CLinkMng::ctrl(const char *name, const char *cmd, AL_PARAM param) {
    if (!cmd) return setLastError("ctrl: cmd is NULL"), E_AL_PARAM_ERR;

    std::string n = name ? name : "";
    std::string c = cmd;

    /* 子网级："model.net" + enable/disable */
    size_t dot = n.find('.');
    if (dot != std::string::npos) {
        std::string mn = n.substr(0, dot), nn = n.substr(dot + 1);
        if (c == "enable" || c == "disable") {
            if (!m_hub.hasModel(mn))
                return setLastError("ctrl: model '" + mn + "' not found"), E_AL_LINK_NOT_FOUND;
            m_hub.setNetEnabled(mn, nn, c == "enable");
            return E_AL_OK;
        }
        return setLastError("ctrl: unsupported subnet cmd '" + c + "'"), E_AL_PARAM_ERR;
    }

    /* 管理器级（通用命令；工作流专属命令由 onCtrl 钩子兜底） */
    if (n.empty()) {
        if (c == "get_last_error") {
            /* 出参缓冲不足不再静默返回 OK：返回 E_AL_BUFFER_TOO_SMALL，
             * 调用方可按 audio_link.h 标注的 char[256] 重来 */
            return al::alCopyOut(param, lastError());
        }
        if (c == "get_stats") {
            /* 通用统计字段 + 工作流增量字段（appendStats 钩子）后再闭合 */
            std::string js = "{\"dropped_pcm_bytes\":" +
                             std::to_string((long long)m_dropped_bytes.load()) +
                             ",\"task_busy\":" + (m_task_busy.load() ? "true" : "false") +
                             ",\"self_heal_count\":" + std::to_string(m_self_heal_count.load());
            appendStats(js);
            js += "}";
            return al::alCopyOut(param, js);
        }
        if (c == "list_nets") {
            return al::alCopyOut(param, m_hub.dumpNets());
        }
        if (c == "set_log") {
            /* param.data = int 级别：-1=关闭日志，0..3=AL_LOG_ERROR..DEBUG */
            if (param.data && param.size >= (int)sizeof(int)) {
                int lv = *(const int *)param.data;
                if (lv < 0) {
                    m_log_en = false;
                } else {
                    m_log_en = true;
                    al_log_set_level(lv);
                }
            }
            return E_AL_OK;
        }
        if (c == "set_log_file") {
            /* param.data = 日志文件路径字符串（AL_PARAM{path, len}）；NULL = 回退 stderr。
             * 运行期重定向保留已配置的轮转阈值（al_log_max），不覆盖 JSON "log.max_bytes" */
            const char *path = (const char *)param.data;
            al_log_set_file(path, al_log_max());
            return E_AL_OK;
        }
        if (c == "set_status_file") {
            /* param.data = 状态快照路径（Stop 退出前写入 Markdown 运行状态）；NULL = 关闭 */
            m_status_path = param.data && param.size > 0 ? std::string((const char *)param.data) : "";
            return E_AL_OK;
        }
        if (c == "save_params") {
            /* 运行期专用配置命令修改写回启动 JSON：dump(2) → 写 <cfg>.tmp →
             * rename 原子替换（中途断电不会留下半个配置文件） */
            if (strCfgPath.empty() || !m_cfg)
                return setLastError("save_params: no config path (init with json file first)"),
                       E_AL_NOT_INIT;
            std::string tmp = strCfgPath + ".tmp";
            std::string text = m_cfg->Dump(2);
            remove(tmp.c_str());   /* 先清历史残留 tmp */
            FILE *fp = fopen(tmp.c_str(), "w");
            if (!fp) {
                std::string msg = "save_params: open failed: " + tmp;
                return setLastError(msg), logErr(msg), E_AL_INTERNAL_ERR;
            }
            bool ok = fwrite(text.data(), 1, text.size(), fp) == text.size();
            if (fclose(fp) != 0) ok = false;
            if (!ok || rename(tmp.c_str(), strCfgPath.c_str()) != 0) {
                remove(tmp.c_str());
                std::string msg = "save_params: write/rename failed: " + tmp + " -> " + strCfgPath;
                return setLastError(msg), logErr(msg), E_AL_INTERNAL_ERR;
            }
            log("params saved to " + strCfgPath);
            return E_AL_OK;
        }
        /* 通用管理命令未命中：转发工作流子类（如 flush/new_meeting），
         * 未识别命令由子类统一报 "unknown manager cmd" */
        return onCtrl(n, c, param);
    }

    /* Link 级 */
    CBaseLink *link = findLink(n);
    if (!link) return setLastError("ctrl: link '" + n + "' not found"), E_AL_LINK_NOT_FOUND;

    /* 模型真实卸载/重载（工作流专属命令：基座识别命令字后转发子类处理） */
    if (c == "unload_model" || c == "load_model")
        return onCtrl(n, c, param);

    return link->ctrl(c, param);
}

/*============================ 工具 ============================*/

CBaseLink *CLinkMng::findLink(const std::string &name) {
    std::lock_guard<std::mutex> lk(m_links_mtx);
    auto it = m_links.find(name);
    return it == m_links.end() ? nullptr : it->second.get();
}

const JsonValue *CLinkMng::section(const std::string &key) const {
    if (!m_cfg) return nullptr;
    return m_cfg->get(key).get();
}

/* 配置内资产路径解析：相对路径以启动 JSON 所在目录为基准（实现见 cpp/al_path.h） */
std::string CLinkMng::resolveAsset(const std::string &p) const {
    return al::resolveAsset(strCfgPath, p);
}

/* 模型文件路径解析：相对路径以**应用根（启动 CWD）**为基准（实现见 cpp/al_path.h） */
std::string CLinkMng::resolveModelPath(const std::string &p) const {
    return al::resolveModelPath(p);
}

/*============================ 动态配置写回 ============================*/

bool CLinkMng::UpdateCfgNum(const std::string &sec, const std::string &key, double v) {
    return UpdateCfgNum(sec, "", key, v);
}

bool CLinkMng::UpdateCfgNum(const std::string &sec, const std::string &sub,
                            const std::string &key, double v) {
    if (!m_cfg) { setLastError("UpdateCfgNum: config not loaded"); return false; }
    auto s = m_cfg->get(sec);
    if (!s || !s->isObj()) { setLastError("UpdateCfgNum: section '" + sec + "' not found"); return false; }
    JsonValue *node = s.get();
    if (!sub.empty()) {
        auto sb = s->get(sub);
        if (!sb || !sb->isObj()) {
            /* 子节不存在：自动创建（energy_gate 等可选节缺省场景），保证写回原位置 */
            auto np = std::make_shared<JsonValue>();
            np->type = JsonValue::OBJ;
            s->obj[sub] = np;
            node = np.get();
        } else {
            node = sb.get();
        }
    }
    node->Set(key, v);
    return true;
}

bool CLinkMng::UpdateCfgStr(const std::string &sec, const std::string &key, const std::string &v) {
    if (!m_cfg) { setLastError("UpdateCfgStr: config not loaded"); return false; }
    auto s = m_cfg->get(sec);
    if (!s || !s->isObj()) { setLastError("UpdateCfgStr: section '" + sec + "' not found"); return false; }
    s->Set(key, v);
    return true;
}

void CLinkMng::log(const std::string &msg) {
    if (!m_log_en.load()) return;
    AL_LOG_I("audiolink", "%s", msg.c_str());
}

void CLinkMng::logWarn(const std::string &msg) {
    if (!m_log_en.load()) return;
    AL_LOG_W("audiolink", "%s", msg.c_str());
}

void CLinkMng::logErr(const std::string &msg) {
    if (!m_log_en.load()) return;
    AL_LOG_E("audiolink", "%s", msg.c_str());
}

void CLinkMng::setLastError(const std::string &e) {
    std::lock_guard<std::mutex> lk(m_err_mtx);
    m_last_err = e;
}

std::string CLinkMng::lastError() {
    std::lock_guard<std::mutex> lk(m_err_mtx);
    return m_last_err;
}

/*============================ 退出状态快照 ============================*/

void CLinkMng::dumpStatusOnStop() {
    if (m_status_path.empty()) return;
    std::string md = dumpStatusMd();
    FILE *fp = fopen(m_status_path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "[audiolink] status dump failed: %s\n", m_status_path.c_str());
        return;
    }
    fwrite(md.data(), 1, md.size(), fp);
    fclose(fp);
    log("runtime status written to " + m_status_path);
}

std::string CLinkMng::dumpStatusMd() {
    std::ostringstream oss;
    double uptime_s = 0;
    if (m_running.load())
        uptime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                 m_start_time).count();

    char tsbuf[64] = "unknown";
    {
        time_t now = time(nullptr);
        struct tm tmv;
#if defined(_WIN32)
        localtime_s(&tmv, &now);
#else
        localtime_r(&now, &tmv);
#endif
        strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", &tmv);
    }

    oss << "# MixedAudioLink 运行状态快照\n\n";
    oss << "- 导出时间：" << tsbuf << "\n";
    oss << "- 版本：" << AudioLinkGetVersion() << "\n";
    oss << "- 运行时长：" << (int)uptime_s << " s"
        << (m_running.load() ? "" : "（已停止）") << "\n\n";
    oss << "## 运行统计\n\n";
    oss << "| 指标 | 值 |\n|---|---|\n";
    oss << "| 推入 PCM | " << (m_pushed_bytes.load() / 1024) << " KB |\n";
    oss << "| 识别段数 | " << m_seg_total.load() << " |\n";
    oss << "| 丢弃 PCM | " << (m_dropped_bytes.load() / 1024) << " KB |\n";
    oss << "| 任务线程 | " << (m_task_busy.load() ? "busy" : "idle") << " |\n";

    /* 工作流专属快照段（说话人统计/转写摘录等，子类钩子） */
    std::string md = oss.str();
    appendStatusMd(md);

    std::string err = lastError();
    if (!err.empty()) md += "\n## 最近错误\n\n> " + err + "\n";
    return md;
}

} /* namespace al */
