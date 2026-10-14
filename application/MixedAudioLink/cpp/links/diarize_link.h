//===----------------------------------------------------------------------===//
//
// diarize_link.h - 说话人归属（在线自动聚类）+ 会议转写缓冲
// 策略（已确认）：无预注册；首段 → 讲话人1；后续段与已有质心算余弦，
// ≥ spk.threshold(0.6) 归入并滑动更新质心，否则新建（上限 spk.max_speakers=8）。
// 无声纹段 speaker_id=0；无文本段不追加转写行（仅更新归属）。
// 转写缓冲经 Ctrl "get_transcript" 一次性取走（会议总结为非流式的独立模块，
// 不做成链上节点）；"new_meeting" 清空开新会。
//
//===----------------------------------------------------------------------===//
#ifndef AL_DIARIZE_LINK_H
#define AL_DIARIZE_LINK_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "../al_out.h"    /* alCopyOut / alFormatOut：出参写入统一查截断 */
#include "../link_mng.h"

namespace al {

class CDiarizeLink : public CBaseLink {
public:
    CDiarizeLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_DIARIZE) {}

    int create(const JsonValue *cfg, AL_PARAM param) override {
        (void)param;
        auto spk_cfg = mng()->section("spk");
        if (spk_cfg) {
            m_threshold = (float)spk_cfg->getNum("threshold", 0.6);
            m_max_speakers = (int)spk_cfg->getNum("max_speakers", 8);
        }
        return E_AL_OK;
    }

    /**
     * 任务线程调用：归属说话人 → 填 speaker_id/line → 追加转写 → 转发 sink。
     * r.text/r.spk_emb 在调用期间有效。
     */
    void process(AL_RESULT &r) {
        std::string line;
        int spk_id = 0;
        if (r.spk_emb && r.spk_dim > 0)
            spk_id = assignSpeaker(r.spk_emb, r.spk_dim);
        r.speaker_id = spk_id;
        if (r.text && *r.text) {
            if (spk_id > 0) line = "讲话人" + std::to_string(spk_id) + "：" + r.text;
            else line = r.text;
            appendTranscript(spk_id, r.start_ms, r.end_ms, r.text);
        }
        m_line = line;
        r.line = m_line.empty() ? nullptr : m_line.c_str();

        /* 转发 sink（回调带归属结果） */
        if (CSinkLink *sink = dynamic_cast<CSinkLink *>(mng()->findLink("sink")))
            sink->emit(r);
    }

    int ctrl(const std::string &cmd, AL_PARAM param) override {
        if (cmd == "get_transcript") {
            std::lock_guard<std::mutex> lk(m_mtx);
            std::string full;
            for (auto &l : m_transcript)
                full += (l.spk > 0 ? "讲话人" + std::to_string(l.spk) + "：" : "") + l.text + "\n";
            if (param.data && param.size > 0)
                return al::alCopyOut(param, full);   /* 转写文本可能很长，必须查截断 */
            return E_AL_OK;
        }
        if (cmd == "new_meeting") {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_speakers.clear();
            m_transcript.clear();
            return E_AL_OK;
        }
        if (cmd == "get_speaker_stats") {
            std::lock_guard<std::mutex> lk(m_mtx);
            std::vector<int> cnt(m_speakers.size() + 1, 0);
            for (auto &l : m_transcript)
                if (l.spk > 0 && l.spk <= (int)m_speakers.size()) cnt[l.spk]++;
            std::ostringstream oss;
            oss << "{\"speakers\":[";
            for (size_t i = 1; i < cnt.size(); i++)
                oss << (i > 1 ? "," : "") << "{\"id\":" << i << ",\"lines\":" << cnt[i] << "}";
            oss << "]}";
            if (param.data && param.size > 0)
                return al::alCopyOut(param, oss.str());
            return E_AL_OK;
        }
        /*---- 专用配置命令（数值统一 double 协议：AL_PARAM{&dVal, sizeof(double)}）----
         * 校验失败 setLastError（含期望范围与实际值）+ E_AL_PARAM_ERR，成员不变；
         * 生效后写回内存配置树（失败仅告警，运行期修改仍有效，只是不随 save_params 持久化） */
        if (cmd == "set_threshold") {
            std::lock_guard<std::mutex> lk(m_mtx);
            if ((NULL == param.data) || (param.size < (int)sizeof(double))) {
                mng()->setLastError("diarize.set_threshold: expect AL_PARAM{&double, sizeof(double)}");
                return E_AL_PARAM_ERR;
            }
            double dVal = *(const double *)param.data;
            if ((!std::isfinite(dVal)) || (dVal <= 0) || (dVal >= 1)) {
                mng()->setLastError("diarize.set_threshold: expect (0,1), got " +
                                    std::to_string(dVal));
                return E_AL_PARAM_ERR;
            }
            m_threshold = (float)dVal;
            if (!mng()->UpdateCfgNum("spk", "threshold", dVal))
                mng()->logWarn("diarize.set_threshold: cfg writeback failed, not persisted");
            return E_AL_OK;
        }
        if (cmd == "set_max_speakers") {
            std::lock_guard<std::mutex> lk(m_mtx);
            if ((NULL == param.data) || (param.size < (int)sizeof(double))) {
                mng()->setLastError("diarize.set_max_speakers: expect AL_PARAM{&double, sizeof(double)}");
                return E_AL_PARAM_ERR;
            }
            double dVal = *(const double *)param.data;
            if ((!std::isfinite(dVal)) || (dVal < 1) || (dVal != std::floor(dVal))) {
                mng()->setLastError("diarize.set_max_speakers: expect int >= 1, got " +
                                    std::to_string(dVal));
                return E_AL_PARAM_ERR;
            }
            /* 下调小于已建质心数：仅告警不收缩（新段不再新建，质心继续滑动更新） */
            if ((int)dVal < (int)m_speakers.size())
                mng()->logWarn("diarize.set_max_speakers: " + std::to_string((int)dVal) +
                               " < built speakers " + std::to_string(m_speakers.size()) +
                               ", centroids kept (no shrink)");
            m_max_speakers = (int)dVal;
            if (!mng()->UpdateCfgNum("spk", "max_speakers", dVal))
                mng()->logWarn("diarize.set_max_speakers: cfg writeback failed, not persisted");
            return E_AL_OK;
        }
        if (cmd == "get_params") {
            std::lock_guard<std::mutex> lk(m_mtx);
            if (param.data && param.size > 0)
                return al::alFormatOut(param,
                         "{\"threshold\":%.4f,\"max_speakers\":%d}",
                         (double)m_threshold, m_max_speakers);
            return E_AL_OK;
        }
        return E_AL_PARAM_ERR;
    }

private:
    struct Spk { std::vector<float> centroid; int utt_cnt; };
    struct TLine { int spk; int start_ms, end_ms; std::string text; };

    int assignSpeaker(const float *emb, int dim) {
        std::lock_guard<std::mutex> lk(m_mtx);
        int best = -1;
        float best_cos = -2.0f;
        for (size_t i = 0; i < m_speakers.size(); i++) {
            float c = cosine(emb, m_speakers[i].centroid.data(), dim);
            if (c > best_cos) { best_cos = c; best = (int)i; }
        }
        if (best >= 0 && best_cos >= m_threshold) {
            /* 滑动均值更新质心：c = (c*cnt + emb)/(cnt+1) */
            Spk &s = m_speakers[best];
            for (int k = 0; k < dim; k++)
                s.centroid[k] = (s.centroid[k] * s.utt_cnt + emb[k]) / (s.utt_cnt + 1);
            s.utt_cnt++;
            return best + 1;
        }
        if ((int)m_speakers.size() >= m_max_speakers)
            return best + 1;   /* 超限：强制归入最相似者 */
        m_speakers.push_back({std::vector<float>(emb, emb + dim), 1});
        return (int)m_speakers.size();   /* 新讲话人 */
    }

    static float cosine(const float *a, const float *b, int dim) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < dim; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-6f);
    }

    void appendTranscript(int spk, int start_ms, int end_ms, const std::string &text) {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_transcript.push_back({spk, start_ms, end_ms, text});
    }

    float m_threshold = 0.6f;
    int m_max_speakers = 8;
    std::vector<Spk> m_speakers;
    std::vector<TLine> m_transcript;
    std::string m_line;      /* process 期间暂存格式化行（AL_RESULT.line 指向它） */
    std::mutex m_mtx;        /* Ctrl 查询与任务线程并发 */
};

} /* namespace al */
#endif /* AL_DIARIZE_LINK_H */
