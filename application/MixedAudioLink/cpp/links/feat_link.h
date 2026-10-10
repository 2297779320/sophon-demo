//===----------------------------------------------------------------------===//
//
// feat_link.h - 特征提取多支路
//   mel128 (asr)   → asr_link（asr_mel.h 实现）
//   fbank80 (spk)  → spk_link（本文件，CAM++ 风格：fbank + utterance 均值减）
//   fbank+cmvn(vad)→ vad_link 内部使用 al_feature.h，不经此 Link
//
//===----------------------------------------------------------------------===//
#ifndef AL_FEAT_LINK_H
#define AL_FEAT_LINK_H

#include <functional>
#include <memory>

#include "../al_feature.h"
#include "../link_mng.h"

namespace al {

class CFeatLink : public CBaseLink {
public:
    CFeatLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_FEAT) {}

    int create(const JsonValue *cfg, AL_PARAM param) override {
        (void)cfg; (void)param;
        FbankConfig fc;   /* CAM++ 惯例：80 bins / 25 / 10ms */
        m_spk_fbank.reset(new FbankComputer(fc));
        return E_AL_OK;
    }

    /** mel128 支路：实际计算在 asr_mel.h（CAsrLink 持有），此处透传 pcm 原样即可 */
    int computeMel128(const std::vector<int16_t> &pcm, std::vector<float> &mel) {
        if (!m_mel_cb) return E_AL_NO_FRAME;   /* 回调未注入（init 绑定缺失/asr 不可用），
                                                * mel 保持空，asr.processSegment 按 NO_FRAME 跳过 */
        m_mel_cb(pcm, mel);
        return E_AL_OK;
    }

    /** fbank80 支路：CAM++ 风格 + utterance 均值减 */
    int computeFbank80(const std::vector<int16_t> &pcm, std::vector<float> &feat) {
        std::vector<int16_t> buf = pcm;
        std::vector<float> raw;
        int rows, cols;
        m_spk_fbank->compute(buf, raw, rows, cols);
        if (rows <= 0) return E_AL_NO_FRAME;
        FbankComputer::subtractFeatureMean(raw, cols);
        feat = raw;
        m_rows = rows;
        m_cols = cols;
        return E_AL_OK;
    }

    int lastRows() const { return m_rows; }
    int lastCols() const { return m_cols; }

    /* asr_link 注入 mel128 计算回调（避免 feat→asr 反向依赖） */
    void setMelComputer(std::function<void(const std::vector<int16_t> &, std::vector<float> &)> cb) {
        m_mel_cb = std::move(cb);
    }

private:
    std::unique_ptr<FbankComputer> m_spk_fbank;
    std::function<void(const std::vector<int16_t> &, std::vector<float> &)> m_mel_cb;
    int m_rows = 0, m_cols = 0;
};

} /* namespace al */
#endif /* AL_FEAT_LINK_H */
