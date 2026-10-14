//===----------------------------------------------------------------------===//
//
// spk_link.h - CAM++ 声纹 embedding（SYSIO 安全 launch，参照 sample/CAM++/cpp）
// 输出向量交上层做 1:N 比对（决策已确认：框架内不建声纹库）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_SPK_LINK_H
#define AL_SPK_LINK_H

#include "../link_mng.h"

namespace al {

class CSpkLink : public CBaseLink {
public:
    CSpkLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_SPK) {}
    ~CSpkLink() override = default;

    int create(const JsonValue *cfg, AL_PARAM param) override {
        (void)cfg; (void)param;
        CModelHub &hub = mng()->hub();
        /* 模型名唯一来源：按角色 speaker 反查，无角色配置时退回键名 "spk" */
        m_model = mng()->modelNameByRole("speaker", "spk");
        if (m_model.empty()) {
            mng()->logErr("spk model not found: neither a model with role 'speaker' nor an 'spk' key");
            return E_AL_NET_NOT_MATCH;
        }
        m_net = hub.netByRole(m_model, "speaker");
        if (m_net.empty()) m_net = hub.model(m_model)->net_names[0]; /* 单网模型取第一个 */
        m_info = hub.netInfo(m_model, m_net);
        if (!m_info) return E_AL_NET_NOT_MATCH;
        /* spk.emb_dim 与子网输出维度校验（不符仅告警，输出仍按模型实际维度取） */
        auto spk_cfg = mng()->section("spk");
        int emb_dim = spk_cfg ? (int)spk_cfg->getNum("emb_dim", 0) : 0;
        auto &osh = m_info->stages[0].output_shapes[0];
        int out_dim = (osh.num_dims >= 1) ? (int)osh.dims[osh.num_dims - 1] : 0;
        if (emb_dim > 0 && out_dim > 0 && emb_dim != out_dim)
            mng()->log("warn: spk.emb_dim " + std::to_string(emb_dim) +
                       " != net output dim " + std::to_string(out_dim));
        m_max_batch = m_info->stages[0].input_shapes[0].dims[0];
        mng()->log("spk ready: net=" + m_net + " batch=" + std::to_string(m_max_batch));
        return E_AL_OK;
    }

    /** fbank[rows, cols] → embedding 向量
     *  shape 策略：dynamic bmodel 用实际 rows；静态 bmodel 按 net_info dims
     *  补零/截断对齐（rows > T_max 截断保前段，rows < T_max 右补零） */
    int processSegment(const std::vector<float> &fbank, std::vector<float> &emb) {
        int rows = m_last_rows, cols = m_last_cols;
        if (rows <= 0 || cols <= 0 || fbank.size() != (size_t)rows * cols)
            return E_AL_PARAM_ERR;

        bm_handle_t h = (bm_handle_t)bmrt_get_bm_handle(mng()->hub().model(m_model)->bmrt);
        void *bmrt = mng()->hub().model(m_model)->bmrt;

        /* 输入 shape：dynamic=实际 rows；静态=net_info dims（补零/截断对齐） */
        std::vector<float> feat_in(fbank);
        auto &ish = m_info->stages[0].input_shapes[0];
        int rows_in = rows, batch_in = m_max_batch;
        if (!m_info->is_dynamic && (3 == ish.num_dims)) {
            batch_in = (int)ish.dims[0];
            rows_in = (int)ish.dims[1];
            if (rows_in != rows) feat_in.resize((size_t)rows_in * cols, 0.0f);
        }

        /* 输入张量：[batch, rows_in, cols] float32（campplus.hpp 同款） */
        bm_tensor_t in, out;
        if (BM_SUCCESS != bmrt_tensor(&in, bmrt, BM_FLOAT32, {3, {batch_in, rows_in, cols}}))
            return E_AL_INTERNAL_ERR;
        size_t real_bytes = feat_in.size() * sizeof(float);
        if (real_bytes > (size_t)m_info->max_input_bytes[0]) {
            bm_free_device(h, in.device_mem);
            return E_AL_NET_NOT_MATCH; /* 超过模型允许的最大输入 */
        }
        if (BM_SUCCESS != bm_memcpy_s2d_partial(h, in.device_mem, (void *)feat_in.data(), real_bytes)) {
            bm_free_device(h, in.device_mem);
            return E_AL_LAUNCH_FAIL;
        }
        if (BM_SUCCESS != bm_malloc_device_byte(h, &out.device_mem, m_info->max_output_bytes[0])) {
            bm_free_device(h, in.device_mem);
            return E_AL_INTERNAL_ERR;
        }

        bool ok = bmrt_launch_tensor_ex(bmrt, m_net.c_str(), &in, 1, &out, 1, true, false);
        if (ok) bm_thread_sync(h);

        if (ok) {
            /* 实际 embedding 维度取输出 shape 最后一维（避免拷出 padding） */
            int dim = 0;
            auto &osh = m_info->stages[0].output_shapes[0];
            if (osh.num_dims >= 2) dim = (int)osh.dims[osh.num_dims - 1];
            if (dim <= 0) dim = (int)(m_info->max_output_bytes[0] / sizeof(float));
            emb.resize(dim);
            if (BM_SUCCESS != bm_memcpy_d2s_partial(h, (void *)emb.data(), out.device_mem,
                                                    emb.size() * sizeof(float)))
                ok = false;
        }
        bm_free_device(h, in.device_mem);
        bm_free_device(h, out.device_mem);
        return ok ? E_AL_OK : E_AL_LAUNCH_FAIL;
    }

    /** feat_link 回填实际行列数 */
    void setFeatShape(int rows, int cols) { m_last_rows = rows; m_last_cols = cols; }

private:
    std::string m_model, m_net;
    const bm_net_info_t *m_info = nullptr;
    int m_max_batch = 1;
    int m_last_rows = 0, m_last_cols = 0;
};

} /* namespace al */
#endif /* AL_SPK_LINK_H */
