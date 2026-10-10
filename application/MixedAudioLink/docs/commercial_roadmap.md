# MixedAudioLink 商业化优化路线图

定位（已确认）：**嵌入式单路音频设备**（会议主机/录音设备内嵌模块），
优先级：稳定性 > 资源占用 > 易集成。

## P0 商用阻断项

| # | 项 | 说明 | 工作量 |
|---|---|---|---|
| 0-1 | **编译验证** | 🟡 就绪工作已完成（2026-10-09）：CMakeLists 补 asr_link.cpp/SDK 规范化/fail-fast、build_soc.sh 修复、tokenizer 资产拷入、vad CMVN 键移除。待在 BM1688 SDK（≥23.x）环境执行首次交叉编译；⚠️ 资产缺口 mel_filters.npy（whisper npz 转换或模型方提供） | 半天 |
| 0-2 | **模型选型落地** | asr 换 0.6B / INT8 量化 bmodel（显存 8GB 级以下目标板）；summary 同步换小模型；配套：识别精度回归（WER 对比 bf16 基线）+ **显存预算表**（各 bmodel 实测占用、unload/load 切换峰值）写入 docs | 2-3 天 |
| 0-3 | **日志系统** ✅（2026-10-09） | 四级 + 文件重定向/轮转 + 线程安全（al_log.h）+ Ctrl 热更 + 模型加载进度日志 | 1 天 |
| 0-4 | **版本 API** ✅（2026-10-09） | `AudioLinkGetVersion()`（major.minor.patch + git hash + 构建日期，CMake 注入） | 半天 |

## P1 短期（首发版本应含）——✅ 已全部实施（2026-09-22，含独立核验）

| # | 项 | 说明 | 工作量 |
|---|---|---|---|
| 1-1 | **链路级自愈** ✅ | launch 失败/VAD cache 异常时自动重置子模块（cache 归零→重试 1 次），连续 N(=3) 次失败才上报；卸载/重载路径复用现有 Ctrl 机制 | 1-2 天 |
| 1-2 | **内置重采样** ✅ | 目标：摆脱"外部必须 16k mono"假设。多相滤波重采样（48k/44.1k→16k）+ 多声道混缩；`AudioLinkPushPcmEx` 显式格式 + JSON `pcm.input_fmt` 默认声明，格式首推锁定；16k 直推零开销 | 2 天 |
| 1-3 | **配置校验强化** ✅ | 启动时全量校验 JSON（键存在性/数值范围/模型文件存在/子网角色命中数），失败打印具体 JSON 路径与原因；`strict` 开关（默认宽松兼容现状） | 1 天 |

## P2 中期（按客户需求排期）——候选清单已经 LLM-TPU 全量 demo 扫描补齐（2026-10-09）

### 已完成
| # | 项 | 说明 |
|---|---|---|
| 2-0 | **io_alone KV 免拷贝** ✅（2026-10-09） | init 双重检测（`addr_mode==1` 声明 + block_cache_0 KV 输入/输出设备地址运行时比对，逐层不一致 logErr 取首层）；io_alone=true：forwardFirst outs[1]/[2] 重绑 past 缓存（prefill 免 2×NUM_LAYERS d2d）+ forwardNext KV 偏移原地写；io_alone=false：BASIC 回退（stage KV 输出 launch 后 d2d 回写 past+offset）。⚠️ addr_mode 双分支行为待板端实测（实现见 text_llm.h，说明见 dataflow.md §4.1） |

### 高价值候选（零/低依赖，优先排期）
| # | 项 | 参照（LLM-TPU / sophon-demo） | 说明 |
|---|---|---|---|
| 2-1 | 转写持久化 + SRT/TXT 落盘 | sophon-demo `sample/Whisper/python/bmwhisper/utils.py:282-437`（WriteTXT/SRT/VTT/JSON 工厂 + format_timestamp 断行） | 按天落盘 append + 可选 SRT 字幕导出；总结可从文件读入（MeetingSummaryRunFile 已就绪） |
| 2-2 | 热词/领域词注入 | Qwen3_ASR `pipeline.py:44-53`（system prompt 用户注入）；Whisper `transcribe.py:163-165`（initial_prompt=custom vocabularies） | `Ctrl "asr" "set_system_prompt"` 已就绪，热词表拼接即用；LLM 式 ASR 对 prompt 领域词敏感 |
| 2-3 | 段级时间戳转写行 | Whisper `transcribe.py:286-293`；VAD 段时钟自产 | 转写行格式升级 `[mm:ss] 讲话人N：文本`，纯格式化 |
| 2-4 | share_prompt KV 快照/恢复 | `Qwen3_5/cpp_demo_share_prompt/chat.cpp`（save_share_prompt/restore_share_prompt） | 固定 system prompt 只 prefill 一次，KV 快照备份/回拷；MeetingSummary 多问题同模板场景省重复 prefill。依赖额外 KV 备份显存 |
| 2-5 | stop-string 流式截断 | `Qwen3/cpp_demo/chat.cpp` check_stop | decode 循环内 ends_with 提前截断；summary 可配置截断词 |
| 2-6 | io_alone_reuse 重载保 KV | `legacy/Qwen1_5/share_cache_demo/chat.cpp:321-354` | 卸载/重载换轮时 tmp 中转 d2d 保留旧 KV。依赖层间地址连续性断言 |

### 中价值候选
| # | 项 | 参照 | 说明 |
|---|---|---|---|
| 2-7 | memory_prealloc 分段预分配 | `legacy/Qwen1_5/share_cache_demo/chat.cpp:257-261,454-472`（bmrt_set_flags BM_RUNTIME_SHARE_MEM + load_bmodel_with_mem_v2） | 按段预分配/跨实例权重复用/动态归还；双模型卸载收益大 |
| 2-8 | 统计导出扩展 | — | get_stats 扩展：段延迟分位/自愈次数趋势——远程运维数据基础 |
| 2-9 | 跨段上下文 | Qwen3_ASR `pipeline.py:164-169`（enable_history 满则清） | 上段文本抗断裂；SEQLEN 余量管理 |
| 2-10 | 自动语种检测 | Whisper `transcribe.py:126-148`（前 30s 检测锁定） | set_language 已有，自动检测为增强 |
| 2-11 | 识别失败回退重解码 | Whisper `transcribe.py:171-211`（temperature fallback） | Qwen3 无 no_speech 头，仅空结果重试 |
| 2-12 | is_dynamic 探测口子 | `legacy/Qwen1_5/share_cache_demo/chat.cpp:163,316` | 全静态够用，留探测+按需改 shape 口子 |
| 2-13 | bmodel AES 加密加载 | `legacy/Qwen1_5/share_cache_demo/chat.cpp:405-452` | 模型资产保护（商用交付可选） |
| 2-14 | 72h 老化测试 | — | 长跑脚本 + 内存/显存基线（重点验证 bmrt 重载无碎片） |

### 低价值/不建议
- speculative decoding（`legacy/Qwen1_5/speculative_sample_demo/chat_speculative.cpp`）：双模型工程量大，音频场景瓶颈在 prefill；
- 独立标点恢复模型 / 词级时间戳 / 真流式 ASR（funasr wss、Seamless）：Qwen3_ASR 原生带标点、非流式、无注意力权重输出；
- Silero VAD 备选：已有 fsmn_vad 同源实现，仅作调参基准（silero_vad.py:100-103：thres 0.5/min_speech 250/min_sil 100/pad 30）。

### 扫描确认的无缺口项
重采样/声道处理（官方全假设 16k 单声道，al_resample.h 已覆盖且更完整）、language 强制
（与 Qwen3_ASR pipeline.py:52-53 同源同法）、标点（LLM 解码原生输出）、VAD 流式协议
（与 fsmn-vad 官方用法同源）、多卡（嵌入式单路场景不需要）。

## 已就绪项（无需投入）

对象池上限与背压、按模型卸载/重载（显存换入换出）、map-reduce 长文本总结、
在线说话人聚类、静态 shape 维度推导、纯 C ABI（`examples/sample_c.c` 已验证接入形态）。

工程规范：已确立 rk_libs 风格 C 编码规范（详见 CODEBUDDY.md；T_xxx/pt 前缀/
大驼峰 static/常量左置/SysLog/分段注释块）——**新增独立纯 C 文件严格套用**，
既有 C++ 代码保持现有风格不混排；后续 P1/P2 新增纯 C 模块按此执行。

## 风险与合规

| 项 | 说明 |
|---|---|
| ⚠️ 质量工程暂缓 | 用户决策。至少 P0-1 编译验证必须做；建议交付前补最小回归集（3-5 个 wav + 期望 diff 脚本） |
| 模型许可证 | qwen3_asr / summary LLM：核对 ModelScope 各模型 LICENSE（Qwen 系列部分为 Tongyi Qianwen 许可，商用条款需逐模型确认）；fsmn-vad（Apache-2.0）/ CAM++（Apache-2.0）无风险 |
| 数据合规 | 全本地推理（无云端上传）是卖点，转写落盘属用户数据，交付文档需声明存储位置与清理方式 |
| 供应链 | tpu-mlir / sophon-sdk 版本锁定写入发布说明（bmodel 与 SDK 版本强相关） |
