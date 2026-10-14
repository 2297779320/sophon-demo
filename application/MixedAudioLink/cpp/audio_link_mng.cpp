//===----------------------------------------------------------------------===//
//
// audio_link_mng.cpp - CAudioLinkMng 音频工作流实现
// 逻辑自 link_mng.cpp 平移（框架通用化拆分），行为零变化：
//   默认链路/工厂分支/VAD+任务线程/pushPcm/音频 Ctrl/音频校验/快照音频段。
//
//===----------------------------------------------------------------------===//
#include "audio_link_mng.h"

#include <chrono>
#include <cstring>

#include "al_json.h"
#include "links/pcm_link.h"
#include "links/vad_link.h"
#include "links/feat_link.h"
#include "links/asr_link.h"
#include "links/spk_link.h"
#include "links/sink_link.h"
#include "links/diarize_link.h"

namespace al {

/***********************************************************************
 *                       链路创建与工厂                          *
 ***********************************************************************/

/* init 末尾钩子：按角色定位 asr 模型 → 维度推导 → 默认链路 → 连线
 * （pcm→vad 拉模型 + feat→asr mel128 支路回调），全部音频工作流专属 */
int CAudioLinkMng::createLinks() {
    /* 按角色定位 asr 模型名：与 CAsrLink::create() 共用 CLinkMng::asrModelName()，
     * 不再在本类另扫一遍 JSON（两套规则会因键名变更而漂移） */
    strAsrModel = asrModelName();
    if (!strAsrModel.empty()) {
        std::string warn = hub().deriveAsrDims(strAsrModel, section("dims"));
        if (!warn.empty()) log("dims warn: " + warn);
    }

    /* 默认链路：pcm → vad（拉模型）；feat/asr/spk/diarize/sink 由任务线程段驱动 */
    struct { const char *name; const char *type; } defaults[] = {
        {"pcm", AL_LINK_PCM},
        {"vad", AL_LINK_VAD},
        {"feat", AL_LINK_FEAT},
        {"asr", AL_LINK_ASR},
        {"spk", AL_LINK_SPK},
        {"diarize", AL_LINK_DIARIZE},
        {"sink", AL_LINK_SINK},
    };
    for (auto &d : defaults) {
        int rc = createLink(d.name, d.type, AL_PARAM{nullptr, 0});
        if (rc != E_AL_OK) return rc;   /* createLink 内部已 setLastError */
        log(std::string("link '") + d.name + "' created");
    }
    int brc = bind("pcm", "vad");
    if (brc != E_AL_OK) return brc;     /* bind 内部已 setLastError */
    /* wire mel128 支路：feat → asr（避免 feat→asr 反向依赖） */
    if (CFeatLink *feat = dynamic_cast<CFeatLink *>(findLink("feat"))) {
        if (CAsrLink *asr = dynamic_cast<CAsrLink *>(findLink("asr")))
            feat->setMelComputer([asr](const std::vector<int16_t> &pcm, std::vector<float> &mel) {
                asr->computeMel(pcm, mel);
            });
    }
    return E_AL_OK;
}

/* Link 工厂分支（基座 createLink 经 createLinkImpl 调用） */
CBaseLink *CAudioLinkMng::createLinkImpl(const std::string &type, const std::string &name) {
    if (AL_LINK_PCM == type)
    {
        return new CPcmLink(this, name);
    }
    if (AL_LINK_VAD == type)
    {
        return new CVadLink(this, name);
    }
    if (AL_LINK_FEAT == type)
    {
        return new CFeatLink(this, name);
    }
    if (AL_LINK_ASR == type)
    {
        return new CAsrLink(this, name);
    }
    if (AL_LINK_SPK == type)
    {
        return new CSpkLink(this, name);
    }
    if (AL_LINK_DIARIZE == type)
    {
        return new CDiarizeLink(this, name);
    }
    if (AL_LINK_SINK == type)
    {
        return new CSinkLink(this, name);
    }
    return nullptr;   /* 未知类型：基座统一报 "unknown type" */
}

/***********************************************************************
 *                       线程（VAD + 任务）                      *
 ***********************************************************************/

int CAudioLinkMng::startWorkers() {
    m_vad_thread = std::thread(&CAudioLinkMng::vadThreadFunc, this);
    m_task_thread = std::thread(&CAudioLinkMng::taskThreadFunc, this);
    return E_AL_OK;
}

void CAudioLinkMng::stopWorkers() {
    teardownThreads();   /* join（线程对象在基座，基座析构兜底安全） */
}

void CAudioLinkMng::vadThreadFunc() {
    CVadLink *vad = dynamic_cast<CVadLink *>(findLink("vad"));
    if (!vad) return;
    while (running()) {
        int rc = vad->tick();
        if (rc != E_AL_OK) {
            /* E_AL_NO_FRAME：PCM 无新数据；E_AL_TIMEOUT：对象池满背压；
             * 其余错误（launch 失败等）：避免紧循环空转，统一小睡兜底 */
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

void CAudioLinkMng::taskThreadFunc() {
    CFeatLink *feat = dynamic_cast<CFeatLink *>(findLink("feat"));
    CAsrLink *asr = dynamic_cast<CAsrLink *>(findLink("asr"));
    CSpkLink *spk = dynamic_cast<CSpkLink *>(findLink("spk"));
    CSinkLink *sink = dynamic_cast<CSinkLink *>(findLink("sink"));
    CDiarizeLink *diarize = dynamic_cast<CDiarizeLink *>(findLink("diarize"));

    std::vector<float> mel, emb;
    while (running()) {
        FramePtr seg;
        if (!popSegment(seg, 100)) continue;
        addSegment();   /* 运行统计：已完成调度处理的段数 */

        /* m_task_busy 已由 popSegment 持锁置位（unload_model 前轮询等待依据） */

        AL_RESULT result;
        memset(&result, 0, sizeof(result));
        result.start_ms = seg->start_ms;
        result.end_ms = seg->end_ms;

        std::string text;
        if (asr && asr->modelReady() && !strAsrModel.empty()) {
            /* feat: mel128 支路 */
            if (feat) feat->computeMel128(seg->pcm, mel);
            int arc = asr->processSegment(mel, seg->is_final, text);
            /* ASR 失败自愈（P1②）：E_AL_NO_FRAME 为卸载期/空特征跳过语义不重试；
             * 其余失败清 KV history 后重试一次（history 错位为常见失败源），
             * 重试成功计自愈次数并清零连击 */
            if (arc != E_AL_OK && arc != E_AL_NO_FRAME) {
                asr->ctrl("clear_history", AL_PARAM{nullptr, 0});
                arc = asr->processSegment(mel, seg->is_final, text);   /* 重试真实 rc */
                if (arc == E_AL_OK) {
                    IncSelfHeal();
                    log("asr self-heal: recovered after clear_history, seg [" +
                        std::to_string(seg->start_ms) + "," + std::to_string(seg->end_ms) + "] ms");
                }
            }
            if (arc == E_AL_OK) {
                result.text = text.c_str();
                iAsrFailStreak = 0;
            } else if (arc != E_AL_NO_FRAME) {
                /* 变沿日志：streak==1 首败 WARN、==3 升级 logErr，其后静默防刷屏 */
                iAsrFailStreak++;
                std::string msg = "asr processSegment failed rc=" + std::to_string(arc) +
                                  ", fail_streak=" + std::to_string(iAsrFailStreak.load());
                if (1 == iAsrFailStreak.load()) logWarn(msg);
                else if (3 == iAsrFailStreak.load()) logErr(msg + " (persistent)");
            }
        }
        if (spk) {
            std::vector<float> fbank;
            if (feat && feat->computeFbank80(seg->pcm, fbank) == E_AL_OK) {
                spk->setFeatShape(feat->lastRows(), feat->lastCols());
                if (spk->processSegment(fbank, emb) == E_AL_OK) {
                    result.spk_emb = emb.data();
                    result.spk_dim = (int)emb.size();
                }
            }
        }
        /* asr 卸载期间：无文本段仅更新说话人归属，不追加转写行（diarize 内处理） */
        if (diarize) diarize->process(result);
        else if (sink) sink->emit(result);
        mel.clear();
        emb.clear();
        m_task_busy = false;
    }
}

/***********************************************************************
 *                       数据通路（pushPcm）                     *
 ***********************************************************************/

int CAudioLinkMng::pushPcm(const void *pcm, int bytes) {
    if (!pcm || bytes <= 0)
        return setLastError("pushPcm: invalid param"), E_AL_PARAM_ERR;
    CPcmLink *pcm_link = dynamic_cast<CPcmLink *>(findLink("pcm"));
    if (!pcm_link)
        return setLastError("pushPcm: pcm link not found"), E_AL_LINK_NOT_FOUND;
    /* AudioLinkPushPcm 按 JSON pcm.input_fmt 解释输入格式（默认 16k/mono/int16，
     * 此时 pushEx 内部 bypass，行为与历史版本逐字节一致） */
    int rc = pcm_link->pushEx(pcm, bytes, pcm_link->inRate(), pcm_link->inCh(), pcm_link->inBits());
    if (rc == E_AL_OK) addPushed(bytes);
    else setLastError("pushPcm failed: rc=" + std::to_string(rc));
    return rc;
}

int CAudioLinkMng::pushPcmEx(const void *pcm, int bytes, int rate, int ch, int bits) {
    if (!pcm || bytes <= 0)
        return setLastError("pushPcmEx: invalid param"), E_AL_PARAM_ERR;
    CPcmLink *pcm_link = dynamic_cast<CPcmLink *>(findLink("pcm"));
    if (!pcm_link)
        return setLastError("pushPcmEx: pcm link not found"), E_AL_LINK_NOT_FOUND;
    int rc = pcm_link->pushEx(pcm, bytes, rate, ch, bits);
    if (rc == E_AL_OK) addPushed(bytes);
    else setLastError("pushPcmEx failed: rc=" + std::to_string(rc));
    return rc;
}

/***********************************************************************
 *                       Ctrl（音频专属命令）                    *
 ***********************************************************************/

int CAudioLinkMng::onCtrl(const std::string &name, const std::string &cmd, AL_PARAM param) {
    const std::string &n = name;
    const std::string &c = cmd;

    /* 管理器级音频命令（基座通用命令未命中时转发至此） */
    if (n.empty()) {
        if (c == "flush") {
            CVadLink *vad = dynamic_cast<CVadLink *>(findLink("vad"));
            if (!vad) return setLastError("ctrl: vad link not found"), E_AL_LINK_NOT_FOUND;
            return vad->flush();
        }
        if (c == "new_meeting") {
            CDiarizeLink *dz = dynamic_cast<CDiarizeLink *>(findLink("diarize"));
            if (!dz) return setLastError("ctrl: diarize link not found"), E_AL_LINK_NOT_FOUND;
            return dz->ctrl("new_meeting", param);
        }
        return setLastError("ctrl: unknown manager cmd '" + c + "'"), E_AL_PARAM_ERR;
    }

    /* 模型真实卸载/重载（显存受限：总结时卸 asr，总结后重载；v1 仅支持 asr）。
     * 为什么只对 asr 开门：
     *   - asr 的唯一使用者是**任务线程**（段级、串行），可用 m_task_pause +
     *     队列空闲轮询封死窗口，且 asr 有 onModelUnload/onModelLoad 可重新绑定
     *     子网与 dev_buffer；
     *   - vad/spk 由**流式工作线程**高频访问（vad 每 chunk 推理一次），卸载要么
     *     join 该线程（打断实时流、丢失流状态），要么依赖推理侧的防御性校验把
     *     误用降级为 E_AL_MODEL_UNLOADED（已由 acquireModel 提供），两者都不如
     *     "常驻不卸"划算 —— 且 vad/spk 合计不足 40MB，不是显存瓶颈。
     * 故此处显式拒绝非 asr：宁可报错，也不让常驻模型进入半卸载状态。
     * 基座识别命令字后转发至此，n 已确认命中已注册 Link */
    if (c == "unload_model" || c == "load_model") {
        if (n != strAsrModel || n.empty()) {
            std::string msg = "unload/load_model only supported for asr model";
            logErr(msg);
            return setLastError(msg), E_AL_PARAM_ERR;
        }
        CAsrLink *asr = dynamic_cast<CAsrLink *>(findLink(n));
        if (!asr) return setLastError("ctrl: '" + n + "' is not asr link"), E_AL_PARAM_ERR;
        std::string err;
        /* 卸载/重载闸门：先暂停任务线程取段，封死"判定空闲后新段又入队"的 UAF 窗口 */
        m_task_pause = true;
        if (c == "unload_model") {
            if (!hub().modelLoaded(n)) {
                m_task_pause = false;
                return setLastError("unload_model: model '" + n + "' not loaded"), E_AL_PARAM_ERR;
            }
            /* 等待任务队列清空且任务线程空闲（轮询，最多 2s），消除任务线程
             * 在卸载后仍引用子网 mem 的 UAF 窗口；超时拒绝卸载 */
            {
                bool idle = false;
                for (int i = 0; i < 200 && !idle; i++) {   /* 200 × 10ms = 2s */
                    {
                        std::lock_guard<std::mutex> lk(m_seg_mtx);
                        idle = m_seg_queue.empty() && !m_task_busy.load();
                    }
                    if (!idle) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!idle) {
                    m_task_pause = false;
                    return setLastError("unload_model: task queue busy, try later"), E_AL_BUSY;
                }
            }
            asr->onModelUnload();               /* 先释放 link 侧 dev_buffer/KV 引用 */
            int rc = hub().unloadModel(n, err); /* bmrt_destroy 释放设备内存 */
            if (rc != E_AL_OK) {
                m_task_pause = false;
                return setLastError(err), logErr(err), rc;
            }
            log("model unloaded: " + n);
        } else {
            if (hub().modelLoaded(n)) { m_task_pause = false; return E_AL_OK; }
            int rc = hub().loadModel(n, err);
            if (rc != E_AL_OK) {
                m_task_pause = false;
                return setLastError(err), logErr(err), rc;
            }
            if (n == strAsrModel) {
                std::string warn = hub().deriveAsrDims(n, section("dims"));
                if (!warn.empty()) log("dims warn: " + warn);
            }
            rc = asr->onModelLoad();            /* 重新绑定子网/分配 dev_buffer */
            if (rc != E_AL_OK) {
                m_task_pause = false;
                return setLastError("load_model: link rebind failed: rc=" + std::to_string(rc)), rc;
            }
            log("model loaded: " + n);
        }
        m_task_pause = false;
        return E_AL_OK;
    }

    return E_AL_PARAM_ERR;
}

/***********************************************************************
 *                       配置校验（音频键）                      *
 ***********************************************************************/

/* ValidateConfig 第一段末尾钩子：音频专属键校验
 * （asr 条件必需键 / vad CMVN 可选文件 / vad·spk·pcm 数值范围）；
 * 第二段（roles 命中）由基座通用段覆盖，无需音频增量 */
std::string CAudioLinkMng::validateAudio(bool after_hub) {
    if (after_hub) return "";

    const JsonValue *root = rootCfg();
    auto models = root->get("models");

    /* asr 相关键仅当 models 中存在 asr_llm 角色模型时才必需（支持无 asr 配置） */
    bool has_asr = false;
    for (auto &kv : models->obj) {
        auto roles = kv.second ? kv.second->get("roles") : nullptr;
        if (roles && roles->isObj() && roles->obj.count("asr_llm")) has_asr = true;
    }
    auto asr = root->get("asr");
    if (has_asr) {
        if (!asr || !asr->isObj())
            return "asr: required section missing";
        for (const char *k : {"vocab", "merges"}) {
            std::string path = asr->getStr(k);
            if (path.empty())
                return std::string("asr.") + k + ": required key missing";
            if (!configFileExists(resolveAsset(path)))
                return std::string("asr.") + k + ": file not found: " + resolveAsset(path);
        }
        /* mel_filters 可选：缺省走 slaney 现算回退（asr_mel.h buildSlaneyFilterbank）；
         * 填写即校验文件存在。npy 就位后加回 asr.mel_filters 键即可启用精确滤波器 */
        std::string mf = asr->getStr("mel_filters");
        if (!mf.empty() && !configFileExists(resolveAsset(mf)))
            return "asr.mel_filters: file not found: " + resolveAsset(mf);
    }

    /* 可选文件键：填写即校验（vad cmvn_file(am.mvn) / CMVN npy） */
    auto vad = root->get("vad");
    if (vad && vad->isObj()) {
        for (const char *k : {"cmvn_file", "cmvn_mean", "cmvn_std"}) {
            std::string path = vad->getStr(k);
            if (!path.empty() && !configFileExists(resolveAsset(path)))
                return std::string("vad.") + k + ": file not found: " + resolveAsset(path);
        }
    }

    /* 数值范围（区间开闭：ring_ms/max_seg_s/max_new_tokens 下界含；threshold 双开）；
     * log.level 范围校验留在基座（本钩子返回后基座接续检查，错误优先级不变） */
    std::string e = rangeCheck(root->get("pcm").get(), "pcm", "ring_ms", 1, 1e18, false, true);
    if (e.empty()) e = rangeCheck(vad.get(), "vad", "chunk_ms", 50, 1000, false, false);
    if (e.empty()) e = rangeCheck(vad.get(), "vad", "speech_noise_thres", 0, 1, false, false);
    if (e.empty()) e = rangeCheck(vad.get(), "vad", "max_seg_s", 1, 1e18, false, true);
    if (e.empty()) e = rangeCheck(root->get("spk").get(), "spk", "threshold", 0, 1, true, true);
    if (e.empty() && has_asr) e = rangeCheck(asr.get(), "asr", "max_new_tokens", 0, 1e18, true, true);
    return e;
}

/***********************************************************************
 *                       统计与退出快照（音频段）                 *
 ***********************************************************************/

/* get_stats：通用字段（dropped/task_busy/self_heal_count）全部基座持有并已
 * 内置于 JSON，音频无增量字段——钩子保持空实现（保留扩展点） */
void CAudioLinkMng::appendStats(std::string &js) {
    (void)js;
}

/* dumpStatusMd：说话人统计 + 转写摘录两段（diarize 提供时嵌入） */
void CAudioLinkMng::appendStatusMd(std::string &md) {
    CDiarizeLink *diarize = dynamic_cast<CDiarizeLink *>(findLink("diarize"));
    if (!diarize) return;

    char stats[1024] = {0};
    AL_PARAM p{stats, sizeof(stats)};
    if (diarize->ctrl("get_speaker_stats", p) == E_AL_OK && stats[0])
        md += "\n## 说话人统计\n\n```json\n" + std::string(stats) + "\n```\n";
    char tr[2048] = {0};
    p.data = tr;
    p.size = sizeof(tr);
    if (diarize->ctrl("get_transcript", p) == E_AL_OK && tr[0]) {
        md += "\n## 转写摘录（尾部截断）\n\n```text\n" + std::string(tr) + "\n```\n";
    }
}

/***********************************************************************
 *                       析构                            *
 ***********************************************************************/

/* 派生析构体内先停：此时虚表仍指向子类，stop→stopWorkers 正常 join 双线程，
 * 先于基座析构期 deinit→stop（彼时虚调用已退化为基座版本，无法 join） */
CAudioLinkMng::~CAudioLinkMng() {
    stop();
}

} /* namespace al */
