//===----------------------------------------------------------------------===//
//
// audio_link_mng.h - CAudioLinkMng 音频工作流管理器
// 框架通用化分层：基座 CLinkMng（link_mng.h）持有通用骨架（配置解析/对象池/
// 段队列/线程生命周期/通用 Ctrl/统计），本类经虚钩子落地音频工作流专属逻辑：
//   默认链路创建（pcm→vad + feat mel 支路）/ Link 工厂分支 / VAD+任务双线程 /
//   pushPcm 数据口 / 音频 Ctrl（flush/new_meeting/unload_model/load_model）/
//   音频配置键校验 / 退出快照音频段（说话人统计+转写摘录）。
//
//===----------------------------------------------------------------------===//
#ifndef AL_AUDIO_LINK_MNG_H
#define AL_AUDIO_LINK_MNG_H

#include <atomic>
#include <string>

#include "link_mng.h"

namespace al {

class CAudioLinkMng : public CLinkMng {
public:
    CAudioLinkMng() = default;
    ~CAudioLinkMng() override;   /* 派生析构内先 stop：保证线程 join 先于线程对象销毁 */

    CAudioLinkMng(const CAudioLinkMng &) = delete;
    CAudioLinkMng &operator=(const CAudioLinkMng &) = delete;

    /* 数据口（AudioLinkPushPcm/PushPcmEx 桥接经基座虚调用至此） */
    int pushPcm(const void *pcm, int bytes) override;
    int pushPcmEx(const void *pcm, int bytes, int rate, int ch, int bits) override;

protected:
    /*---- 基座虚钩子落地（音频工作流） ----*/
    std::string validateAudio(bool after_hub) override;
    int createLinks() override;
    int startWorkers() override;
    void stopWorkers() override;
    int onCtrl(const std::string &name, const std::string &cmd, AL_PARAM param) override;
    void appendStats(std::string &js) override;
    void appendStatusMd(std::string &md) override;
    CBaseLink *createLinkImpl(const std::string &type, const std::string &name) override;

private:
    void vadThreadFunc();
    void taskThreadFunc();

    std::string strAsrModel;              /* asr 模型名（JSON models 键，按 asr_llm 角色定位） */
    std::atomic<int> iAsrFailStreak{0};   /* asr 连续失败计数（成功清零，达 3 升级 logErr） */
};

} /* namespace al */
#endif /* AL_AUDIO_LINK_MNG_H */
