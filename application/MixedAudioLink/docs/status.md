# MixedAudioLink 商用库状态

> 更新日期：2026-10-09　|　版本：0.1.0（`AudioLinkGetVersion()` 可查）
> 定位：嵌入式单路音频设备（会议主机/录音设备内嵌模块）
> 优先级：稳定性 > 资源占用 > 易集成

## 一、功能完成度

| 模块 | 状态 | 说明 |
|---|---|---|
| 音频流推理链（C API） | ✅ 完成 | PCM 推入 → VAD 切段 → ASR 整句识别 + CAM++ 声纹 → 说话人归属回调 |
| Link 框架 | ✅ 完成 | 字符串寻址 / 拉模型 / 对象池背压 / 双线程调度（VAD 线程 + 任务线程） |
| 框架通用化（CAudioLinkMng 子类化） | ✅ 完成 | `CLinkMng` 拆分为通用基座 + 音频子类：基座持配置/对象池/段队列/通用 Ctrl，音频专属逻辑（默认链路/工厂分支/双线程/pushPcm/音频 Ctrl/音频校验/快照音频段）下沉 `CAudioLinkMng`，经 8 个 protected 虚钩子回接；行为零变化、C ABI 只增不改 |
| 混合模型管理 | ✅ 完成 | **多文件分离**（每 bmodel 独立 bmrt 上下文，**无融合模型**）、子网动态枚举、JSON 角色映射、静态 shape 维度推导 |
| 按模型卸载/重载 | ✅ 完成 | 卸载闸门 + 2s 空闲等待，v1 仅 asr（显存换入换出，会议总结时序） |
| 说话人归属（diarize） | ✅ 完成 | 在线质心聚类"讲话人N：xxx"、转写缓冲、`get_transcript`/`new_meeting` |
| MeetingSummary 总结模块 | ✅ 完成 | 独立模块（非 link 节点）、text-only LLM、map-reduce 长转写、流式回调 |
| 总结模块三件套外置 | ✅ 完成 | `prompt_file` 外置模板（config/summary/summary_prompt.txt，自动剥离 UTF-8 BOM）+ `MeetingSummaryRunFile` 文件输入 + sample 落盘 `summary_output.txt`；独立总结模式 `-t` 失败返回码可感知 |
| 日志系统 | ✅ 完成 | 四级 + 文件重定向/轮转 + 线程安全（`al_log.h`），Ctrl 热更 |
| 版本 API | ✅ 完成 | `AudioLinkGetVersion()`，CMake 注入 git hash/构建日期 |
| 退出状态快照 | ✅ 完成 | `Ctrl "set_status_file"`：Stop 退出前（链路存活时）自动写 Markdown 运行状态（版本/时长/段数/丢弃/转写统计） |
| 静音处理 | ✅ 完成 | 纯静音零下游开销；`max_seg_s` 强制切段（防连续噪音内存无界）；能量前置门（`energy_gate`，默认关闭）静音期跳过 VAD 推理 |
| 纯 C 接入验证 | ✅ 完成 | `examples/sample_c.c`（纯 C99，仅新增不改已有 ABI） |
| 内置重采样 | ✅ 完成 | `al_resample.h` 多相 FIR（48k/44.1k/8k→16k + 多声道混缩）；新 API `AudioLinkPushPcmEx`；16k 直推零开销路径逐字节兼容；格式首推锁定 |
| 链路级自愈 | ✅ 完成 | VAD launch 失败 `resetCache` 后原地重试一次（只复位设备 cache，不动端点状态机 / 滚动缓冲——音频已在缓冲中，段连续性不受影响）；ASR 失败 clear_history 重试；3 连败升级上报；`get_stats` 含 self_heal_count |
| 配置启动校验强化 | ✅ 完成 | `ValidateConfig` 两段式（必需键/数值范围/文件存在 + roles 命中）；`strict` 开关（默认宽松）；报错含完整 JSON 路径 |
| 动态配置接口 | ✅ 完成 | 各 Link 专用 Ctrl 命令（13 个：vad 7 / asr 3 / diarize 2 / pcm 1，数值统一 double 协议）+ get_params + save_params 持久化，立即生效、tmp+rename 原子写回 |
| KV 免拷贝（io_alone） | ✅ 完成 | init 双重检测（addr_mode==1 + block_cache_0 KV 输入/输出地址运行时比对）；io_alone=true：prefill KV 原地写（免 2×NUM_LAYERS d2d）+ decode 偏移原地写；false：BASIC 回退逐层 d2d（⚠️ 双分支行为待板端实测） |
| 转写落盘 / 热词 / 统计导出 | ⬜ P2 | 未实施（P2 候选清单已经 LLM-TPU 全量 demo 扫描补齐，见 commercial_roadmap.md：高价值 6 项 / 中价值 8 项，含参照文件:行） |

## 二、质量状态

| 项 | 状态 |
|---|---|
| 正式评审 | 3 轮 correctness 专项（修复 12 必修 + 14 建议）+ 1 轮五轴评审（5 必修全修 + 死代码清零）+ 1 轮独立交叉核验（PASS）+ meeting_summary 官方例程对照研究（P1 优化 5 项落地）+ P0 改动独立核验（PASS，4 建议全修）+ 三件套外置独立核验（PASS，2 建议 + BOM 剥离已落地）+ P1 三项独立核验（PASS，1 必修 status 同步 + 8 建议落地）+ 专用命令/风格整改独立核验（PASS，0 必修）+ io_alone 独立核验（PASS，1 建议层间 I/O 校验已落地） |
| 静态核验 | 符号/类型/逻辑逐条人工级核查；死代码已清零；退出钩子机制按决策移除（状态快照改由 Stop 直接触发）；近期新增代码块已按 rk_libs 风格整改（专用命令重写 + al_resample/al_json/link_mng 新增段） |
| 编码规范 | rk_libs 风格编码规范已入 CODEBUDDY.md（第 7 条 C++ 映射：T_ 类/类型前缀成员/大驼峰/禁异常/al_log；细则：override 跟随基类、调用存量方法保现名、vec/mtx 前缀约定）；**适用范围：新写与大改代码严格执行**，存量不主动重构 |
| **编译验证** | 🟡 就绪工作已完成（CMakeLists 补 asr_link.cpp/SDK 规范化/fail-fast、build_soc.sh 修复、tokenizer 资产已拷入、vad CMVN 键移除），**待在 BM1688 SDK（≥23.x）环境执行首次交叉编译**；⚠️ mel_filters.npy 为资产缺口（whisper npz 转换或模型方提供） |
| **板端实测** | ⚠️ **未做**（模型加载/VAD 子网 I/O 约定/数值精度均待实机确认） |
| 单元测试 / CI | ⚠️ 无（决策暂缓，见 commercial_roadmap.md 风险节） |
| 已知遗留 | al_json 递归深度上限、npy header 上限、VAD rename 失败重试、错误码语义统一（均低风险，记录于 code_review_report.md） |

## 三、模型与依赖

| 项 | 状态 |
|---|---|
| bmodel | **外部提供**（框架只约定 chat 格式子网命名与 I/O 约定，见 dataflow.md）；预算表模板见 model_budget.md（显存列待实测） |
| 选型建议 | 8GB 级以下：asr 0.6B int8 + summary 0.5B 级 int8（精度回归 CER 劣化 <2pp） |
| 依赖 | bmlib / bmruntime（sophon-sdk）；零第三方 C++ 依赖（无 langchain/boost） |
| 许可 | fsmn_vad / CAM++：Apache-2.0 ✅；qwen3_asr / summary LLM：⚠️ 逐模型核对 ModelScope LICENSE |

## 四、交付物清单

```
application/MixedAudioLink/
├── cpp/                    # 源码：audio_link.h（C ABI 入口）/ meeting_summary.h（总结模块）
│                           # + link_mng（通用基座）/ audio_link_mng（音频子类）/ model_hub
│                           # + al_log / al_json（Set/Dump）/ al_resample / al_feature
│                           # + text_llm / asr_mel / asr_tokenizer
│                           # + links/{pcm,vad,feat,asr,spk,diarize,sink}
├── examples/sample_c.c     # 纯 C 调用示例（C99，验证库接入形态；含运行状态快照演示）
├── config/                 # audio_link.json / meeting_summary.json / summary_prompt.txt
│                           # + vocab/merges（asr+summary 四份 tokenizer 资产）
├── docs/                   # README / API / dataflow / model_budget / status（本文件）
│                           # + code_review_report / meeting_summary_reference_report
│                           # + review_report / commercial_roadmap
├── scripts/build_soc.sh    # 板端构建脚本
└── CMakeLists.txt          # 产物：libaudiolink.a + audiolink_demo + sample_c
```

## 五、下一步（按 commercial_roadmap.md）

1. **P0-1 编译验证**（板端/SDK 容器人工交叉编译，半天）——唯一硬阻断；
2. **P0-2 模型落地**（外部 bmodel 到位后：装载 → 预算表实测回填 → CER 回归）；
3. 交付前补最小回归集（3-5 个 wav + 期望 diff 脚本）；
4. P2 按候选清单排期（高价值 6 项：转写持久化+SRT、热词注入、段级时间戳、share_prompt KV 快照、stop-string 截断、io_alone_reuse 重载保 KV——详见 commercial_roadmap.md）。
