# MixedAudioLink 双 Agent 评审报告

评审方式：**reviewer（只读评审）+ fixer（修改）双 agent 迭代**，共 3 轮
"评审 → 修复 → 重检"，另有 1 次主导者定向修复收尾。评审范围：本模块全部
.h/.cpp/config/docs（约 3600 行）。判定标准：连续一轮零"必须修"缺陷。

## 轮次记录

### 第 1 轮（全量评审）
发现 10 必修 + 9 建议，全部由 fixer 修复。关键缺陷：

| # | 缺陷 | 后果（若带出） |
|---|---|---|
| 1 | CMakeLists 漏 `vad_link.cpp` | 链接失败 |
| 2 | createLink 工厂缺 diarize 分支 | AudioLinkInit 必定失败 |
| 3 | Qwen BPE 字节表：高区可打印字节未按 UTF-8 两字节映射 | 全部中文无法编解码 |
| 4 | mel 前处理 n_fft=400 用 radix-2 FFT | mel 特征全错 |
| 5 | asr prefill 未设置 m_max_posid | 解码位置错乱 |
| 6 | forwardNext 各层绑定自己的 pos/mask mem（未写入） | 第 1 层起拿到垃圾输入 |
| 7 | VAD preroll 每帧重复插块 + onset 后音频丢失 | 段音频重复/缺失 |
| 8 | tokens_per_chunk=13*ALEN/100 公式错 | audio embedding 写入偏移全错 |
| 9 | summary fail 路径 UAF（hub 先于 llm 释放） | 崩溃 |
| 10 | get_last_error 无生产者 | 错误查询不可达 |

建议级 9 项（`<new>` include、center pad、loadOne 泄漏、对象池上限 64、
empty_net、负错误码链路、dtype 校验、JSON 键对齐、task_busy 空闲查询）全部采纳。

### 第 2 轮（回归评审）
上轮 19 项修复核验通过（含 Bluestein 数学、字节表双射性、KV mem 共享时序）。
新发现 2 必修 + 5 建议，全部修复：
- `setLastError` 仍无生产者（第 1 轮修复只有管道）→ 全失败路径补写；
- VAD 帧计时用 chunk 均分（11ms/帧）而非 fbank 帧移（10ms）→ 分钟级时间戳漂移；
- VAD 概率 shape 未校验、E_AL_TIMEOUT 紧循环、unload TOCTOU、netLaunch 返回值
  被忽略、computeMel128 空回调 terminate——均修复。

### 第 3 轮（最终评审）
第 2 轮修复核验通过（含锁序无死锁、负错误码全链语义一致）。发现
1 必修 + 4 建议，由主导者定向修复收尾：

| 项 | 内容 | 处理 |
|---|---|---|
| 必修 | VAD onset 记录的是确认帧而非首个语音帧，段首系统性丢 ≤250ms 语音（吞首字） | 新增 `m_onset_frame/m_onset_chunk_beg`，确认进入语音时从原始语音起点截入当前块 |
| 建议 | flush 后未复位段起点/累计，可能污染下段 | flush 收尾补复位 |
| 建议 | unload 判定空闲与新段入队仍有竞态窗口 | 新增 `m_task_pause` 卸载闸门（置位后任务线程不取新段），完成后清除 |
| 建议 | mel_filters.npy 转置布局 [201,128] 被按行主序误读 | 支持转置读取 |
| 建议 | al_json.h "插入序"注释与 map 字典序实现不符 | 修正注释并说明语义 |

**最终状态：零必修缺陷遗留。** 4 项建议级全部同步修复。

## 遗留与验证边界（重要）

1. **静态评审 ≠ 编译验证**：全部结论来自人工级静态核查（符号/类型/逻辑/语义
   对照），上板后第一件事是交叉编译，链接期问题理论上已排查但以实际编译为准。
2. **数值对齐待实测**：mel/fbank 简化实现与训练前端的数值一致性需用同一 wav
   对比 Python 侧输出（docs/dataflow.md §3）。
3. **运行时行为待验证**：VAD 子网 I/O 约定（inputs[0]=fbank、其余 cache）与
   概率布局（[T]/[T,2]）基于 funasr 惯例推导，实机不符时改 vad_link.cpp 顶部
   下标与 shape 校验逻辑。
