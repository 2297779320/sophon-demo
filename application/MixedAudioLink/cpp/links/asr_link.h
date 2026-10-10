//===----------------------------------------------------------------------===//
//
// asr_link.h - qwen3_asr 推理 Link
// LLM 公共路径在 CTextLLM（text_llm.h）；本类叠加 audio 编码分支、
// mel 前处理、tokenizer 与 prompt 构建。支持模型卸载/重载
// （onModelUnload/onModelLoad，配合 Ctrl "unload_model"/"load_model"）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_ASR_LINK_H
#define AL_ASR_LINK_H

#include <map>
#include <mutex>
#include <vector>

#include "../asr_mel.h"
#include "../asr_tokenizer.h"
#include "../link_mng.h"
#include "../text_llm.h"

namespace al {

class CAsrLink : public CBaseLink {
public:
    CAsrLink(CLinkMng *mng, std::string name) : CBaseLink(mng, name, AL_LINK_ASR) {}
    ~CAsrLink() override = default;

    int create(const JsonValue *cfg, AL_PARAM param) override;
    int ctrl(const std::string &cmd, AL_PARAM param) override;

    /** 任务线程入口：mel[n_times,128,ALEN]（feat 计算的展平结果） → 文本 */
    int processSegment(const std::vector<float> &mel, bool is_final, std::string &out_text);

    /** 模型已卸载（unload_model 后） */
    bool modelReady() const { return m_llm.ready(); }

    /** Ctrl unload_model/load_model 联动（link_mng 调用） */
    int onModelUnload() { m_llm.deinit(); return E_AL_OK; }
    int onModelLoad() { return m_llm.reload(mng()->hub(), m_model); }

    /** mel 计算回调（注册给 feat_link） */
    void computeMel(const std::vector<int16_t> &pcm, std::vector<float> &mel) {
        if (m_llm.ready()) m_mel.compute(pcm, mel);
    }

private:
    int buildInputIds(int n_times, std::vector<int> &ids, int &audio_bos_pos);
    int forwardAudio(const float *chunks, int n_times, const std::vector<int> &offsets);
    int tokensPerChunk();   /* audio 子网每块输出 token 数（output 推导，回退 13） */

    std::string m_model;                 /* JSON models 键 */
    std::string m_audio_net;             /* audio 编码子网名（角色映射结果） */
    CTextLLM m_llm;

    /* 前处理 */
    AsrMel m_mel;
    QwenTokenizer m_tok;
    std::string m_system_prompt;
    std::string m_language;              /* 空=自动 */
    /* 字符串参数锁：set_system_prompt/set_language（ctrl 线程写）与 buildInputIds
     * （任务线程读 std::string，重分配可致读侧 UAF）互斥 */
    std::mutex mtxStr;
    std::atomic<bool> m_enable_history{false};   /* ctrl 线程写 / 任务线程读 */
    int m_max_new_tokens = 64;
    int m_tokens_per_sec = 13;

    /* 特殊 token id */
    int m_id_im_end = -1, m_id_audio_bos = -1;

    /* 增量回调 */
    AL_PARTIAL_CB m_partial_cb = nullptr;
    void *m_partial_user = nullptr;
};

} /* namespace al */
#endif /* AL_ASR_LINK_H */
