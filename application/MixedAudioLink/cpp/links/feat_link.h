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
        /* spk fbank：严格对齐 CAM++/3D-Speaker 官方前端（sample/CAM++/cpp/main.cpp
         * 的 speakerlab::FbankOptions 默认 + 运行时覆盖）：
         *   80 bins / 16k / 25ms / 10ms / preemph 0.97 / 逐帧去直流
         *   window=povey（(0.5-0.5cos)^0.85）/ high_freq=0（→Nyquist 全带宽）
         *   dither=0（官方运行时把默认 1.0 覆盖为 0）/ power 谱 / log
         *   sample_scale=1/32767（官方 WavReader 先归一化到 [-1,1] 再算 fbank）
         * 与 VAD 的 funasr 前端（hamming + high_freq=-400 + int16 量纲）区分开，故此处显式给出 */
        FbankConfig fc;
        fc.num_mel_bins = 80;
        fc.sample_rate   = 16000;
        fc.frame_ms      = 25;
        fc.shift_ms      = 10;
        fc.preemph       = 0.97f;
        fc.low_freq      = 20.0f;
        fc.high_freq     = 0.0f;          /* kaldi 约定：<=0 → Nyquist + high_freq = 8000 */
        fc.remove_dc     = true;
        fc.use_log       = true;
        fc.dither        = 0.0f;
        fc.sample_scale  = 1.0f / 32767.0f;
        fc.window        = FBANK_WIN_POVEY;
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
