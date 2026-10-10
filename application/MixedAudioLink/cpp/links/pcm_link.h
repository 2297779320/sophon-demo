//===----------------------------------------------------------------------===//
//
// pcm_link.h - PCM 输入 sink（AudioLinkPushPcm 喂数，非阻塞 + 丢最旧）
//
//===----------------------------------------------------------------------===//
#ifndef AL_PCM_LINK_H
#define AL_PCM_LINK_H

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include "../al_resample.h"
#include "../link_mng.h"

namespace al {

class CPcmLink : public CBaseLink {
public:
    CPcmLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_PCM) {}

    int create(const JsonValue *cfg, AL_PARAM param) override {
        (void)param;
        /* 环形缓冲容量：默认 5s（实时流场景足够大且内存可控），JSON 可调（下界钳制） */
        int ring_ms = 5000;
        auto vad_cfg = mng()->section("vad");
        auto ring = cfg ? cfg->get("pcm") : nullptr;
        if (ring) ring_ms = (int)ring->getNum("ring_ms", 5000);
        else if (vad_cfg) ring_ms = (int)vad_cfg->getNum("ring_ms", 5000);
        if (ring_ms < 1) {
            mng()->logWarn("pcm.ring_ms < 1, fallback 5000");
            ring_ms = 5000;
        }
        iRingMs = ring_ms;
        m_ring_capacity = 16000LL * ring_ms / 1000;   /* int16 样本数 */
        /* 输入格式（P1①）：AudioLinkPushPcm 按 pcm.input_fmt 解释，
         * 默认 16k/1/16（与历史行为一致）；非法取值告警回退默认 */
        if (ring) {
            auto fmt = ring->get("input_fmt");
            if (fmt && fmt->isObj()) {
                iInRate = (int)fmt->getNum("sample_rate", 16000);
                iInCh = (int)fmt->getNum("channels", 1);
                iInBits = (int)fmt->getNum("bits", 16);
            }
        }
        if (iInRate <= 0 || iInCh <= 0 || 16 != iInBits) {
            mng()->logWarn("pcm.input_fmt invalid, fallback 16000/1/16");
            iInRate = 16000;
            iInCh = 1;
            iInBits = 16;
        }
        /* VAD 拉块时长（默认 200ms）：create 一次读取+钳制告警，
         * 避免每次 getFrame 重复解析 JSON 与刷日志 */
        m_chunk_ms = vad_cfg ? (int)vad_cfg->getNum("chunk_ms", 200) : 200;
        if (m_chunk_ms < 1) {
            mng()->logWarn("vad.chunk_ms < 1, fallback 200");
            m_chunk_ms = 200;
        }
        return E_AL_OK;
    }

    /* AudioLinkPushPcm 的解释格式（JSON pcm.input_fmt，create 读取） */
    int inRate() const { return iInRate; }
    int inCh() const { return iInCh; }
    int inBits() const { return iInBits; }

    /**
     * 显式格式推入（P1①，AudioLinkPushPcmEx / pushPcm 共用）：
     * bits 仅支持 16；首次成功推入锁存 rate/ch（此后不一致拒绝，防误配混流）；
     * 16k/mono bypass 直走 push（默认配置逐字节不变），
     * 其余经 AlResampleFrame（多相 FIR + 混缩）转 16k/mono 后入环。
     */
    int pushEx(const void *data, int bytes, int rate, int ch, int bits) {
        if (!data || bytes <= 0 || rate <= 0 || ch <= 0) {
            mng()->setLastError("pushEx: invalid param");
            return E_AL_PARAM_ERR;
        }
        if (16 != bits) {
            mng()->setLastError("pushEx: only bits=16 supported, got " + std::to_string(bits));
            return E_AL_PARAM_ERR;
        }
        std::lock_guard<std::mutex> lk(mtxRs);
        /* 格式锁存：首次成功推入锁定 rate/ch（bits 恒 16 无需锁存） */
        if (!bFmtLocked.load()) {
            iLatchRate = rate;
            iLatchCh = ch;
            bFmtLocked = true;
        } else if (rate != iLatchRate || ch != iLatchCh) {
            mng()->setLastError("pushEx: input format locked to " +
                                std::to_string(iLatchRate) + "Hz/" + std::to_string(iLatchCh) +
                                "ch, got " + std::to_string(rate) + "Hz/" + std::to_string(ch) + "ch");
            return E_AL_PARAM_ERR;
        }
        /* 16k/mono bypass：逐字节直入环形缓冲（默认配置零开销路径） */
        if (16000 == rate && 1 == ch) return push(data, bytes);
        std::vector<int16_t> in((const int16_t *)data, (const int16_t *)data + bytes / 2);
        std::vector<int16_t> out = AlResampleFrame(in, rate, ch, tRs);
        if (out.empty()) return E_AL_OK;   /* 输入过短暂无输出（相位状态已推进） */
        return push(out.data(), (int)out.size() * 2);
    }

    /** 非阻塞推入：满则丢最旧并计数 */
    int push(const void *data, int bytes) {
        const int16_t *s = (const int16_t *)data;
        int n = bytes / 2;
        std::lock_guard<std::mutex> lk(m_mtx);
        if ((int)(m_ring.size() + n) > m_ring_capacity) {
            int overflow = (int)(m_ring.size() + n) - m_ring_capacity;
            if (overflow > (int)m_ring.size()) overflow = (int)m_ring.size();
            m_ring.erase(m_ring.begin(), m_ring.begin() + overflow);
            mng()->addDropped((int64_t)overflow * 2);
            mng()->log("pcm ring overflow, dropped " + std::to_string(overflow * 2) + " bytes");
        }
        m_ring.insert(m_ring.end(), s, s + n);
        return E_AL_OK;
    }

    /** VAD 线程拉取：每次取 vad.chunk_ms 对应样本数 */
    int getFrame(FramePtr &out) override {
        int want = chunkSamples();
        std::lock_guard<std::mutex> lk(m_mtx);
        if ((int)m_ring.size() < want) return E_AL_NO_FRAME;
        out = acquireFrame();
        if (!out) return E_AL_TIMEOUT;             /* 对象池满 = 背压 */
        out->pcm.assign(m_ring.begin(), m_ring.begin() + want);
        m_ring.erase(m_ring.begin(), m_ring.begin() + want);
        return E_AL_OK;
    }

    int chunkSamples() const {
        return 16000 * m_chunk_ms / 1000;   /* 200ms → 3200 样本 */
    }

    /* 专用配置命令：数值统一 double 协议（AL_PARAM{&dVal, sizeof(double)}），
     * 校验失败 setLastError（含期望范围与实际值）+ E_AL_PARAM_ERR，成员不变。
     * set_ring_ms（int，[1,60000]）：m_mtx 下重建容量——缩容裁最旧（保最新音频），
     * 扩容保留现有内容；立即生效。同步写回 cfg（save_params 持久化基础）。 */
    int ctrl(const std::string &cmd, AL_PARAM param) override {
        if (cmd == "set_ring_ms") {
            if ((NULL == param.data) || (param.size < (int)sizeof(double)))
                return mng()->setLastError("pcm.set_ring_ms: expect AL_PARAM{&double, sizeof(double)}"),
                       E_AL_PARAM_ERR;
            double dVal = *(const double *)param.data;
            if ((!std::isfinite(dVal)) || (dVal < 1) || (dVal > 60000) || (dVal != std::floor(dVal)))
                return mng()->setLastError("pcm.set_ring_ms: expect int in [1,60000], got " +
                                           std::to_string(dVal)),
                       E_AL_PARAM_ERR;
            int ring_ms = (int)dVal;
            std::lock_guard<std::mutex> lk(m_mtx);
            size_t cap = (size_t)(16000LL * ring_ms / 1000);
            if (cap < 1) cap = 1;
            if (m_ring.size() > cap) {
                /* 缩容：裁最旧，保留最新 cap 个样本 */
                m_ring.erase(m_ring.begin(), m_ring.end() - (long long)cap);
                mng()->addDropped((int64_t)(m_ring_capacity - cap) * 2);
            }
            m_ring_capacity = cap;
            iRingMs = ring_ms;
            if (!mng()->UpdateCfgNum("pcm", "ring_ms", dVal))
                mng()->logWarn("pcm.set_ring_ms: cfg writeback failed, not persisted");
            mng()->log("pcm ring resized to " + std::to_string(ring_ms) + " ms");
            return E_AL_OK;
        }
        if (cmd == "get_params") {
            if (param.data && param.size > 0)
                snprintf((char *)param.data, param.size,
                         "{\"ring_ms\":%d,\"input_fmt\":{\"sample_rate\":%d,\"channels\":%d,\"bits\":%d}}",
                         iRingMs, iInRate, iInCh, iInBits);
            return E_AL_OK;
        }
        (void)param;
        return E_AL_PARAM_ERR;
    }

private:
    std::vector<int16_t> m_ring;
    size_t m_ring_capacity = 16000 * 5;
    int iRingMs = 5000;   /* 环形缓冲时长 ms（create 读取，set_ring_ms 运行时可调） */
    int m_chunk_ms = 200;   /* VAD 拉块时长 ms（create 读取+钳制） */
    /* 输入格式（JSON pcm.input_fmt）与运行期锁存 */
    int iInRate = 16000;
    int iInCh = 1;
    int iInBits = 16;
    std::atomic<bool> bFmtLocked{false};   /* 首次成功推入置位 */
    int iLatchRate = 0;                    /* 锁存格式（mtxRs 下读写） */
    int iLatchCh = 0;
    /* 重采样跨调用相位状态（非 16k 输入时使用；mtxRs 保护） */
    T_ResampleState tRs;
    std::mutex mtxRs;
    std::mutex m_mtx;
};

} /* namespace al */
#endif /* AL_PCM_LINK_H */
