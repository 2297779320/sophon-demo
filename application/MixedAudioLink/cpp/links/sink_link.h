//===----------------------------------------------------------------------===//
//
// sink_link.h - 结果汇聚 → AL_RESULT → 用户回调
//
//===----------------------------------------------------------------------===//
#ifndef AL_SINK_LINK_H
#define AL_SINK_LINK_H

#include "../link_mng.h"

namespace al {

class CSinkLink : public CBaseLink {
public:
    CSinkLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_SINK) {}

    int create(const JsonValue *cfg, AL_PARAM param) override {
        (void)cfg;
        /* param 可携带 { AL_RESULT_CB cb; void* user; } 创建即注册 */
        struct SINK_PARAM { AL_RESULT_CB cb; void *user; };
        auto *p = (SINK_PARAM *)param.data;
        if (p && param.size >= (int)sizeof(SINK_PARAM)) {
            m_cb = p->cb;
            m_user = p->user;
        }
        return E_AL_OK;
    }

    /** 结果转发（已由 diarize 完成说话人归属；is_final 语义由段队列侧消费） */
    void emit(const AL_RESULT &r) {
        if (!m_cb) return;
        m_cb(&r, m_user);
    }

    int ctrl(const std::string &cmd, AL_PARAM param) override {
        if (cmd == "set_result_cb") {
            struct SINK_PARAM { AL_RESULT_CB cb; void *user; };
            auto *p = (SINK_PARAM *)param.data;
            if (!p || param.size < (int)sizeof(SINK_PARAM)) return E_AL_PARAM_ERR;
            m_cb = p->cb;
            m_user = p->user;
            return E_AL_OK;
        }
        return E_AL_PARAM_ERR;
    }

private:
    AL_RESULT_CB m_cb = nullptr;
    void *m_user = nullptr;
};

} /* namespace al */
#endif /* AL_SINK_LINK_H */
