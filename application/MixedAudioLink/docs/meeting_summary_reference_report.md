# meeting_summary 官方例程对照研究报告

研究方式：子 agent 深读官方例程 `application/meeting_summary/`（Qwen1.5-7B +
sail.EngineLLM + langchain map-reduce），逐行对照 MixedAudioLink 实现。
行号以当前代码为准。（⚠️ 历史快照：本文行号对应更名前的 audio_summary.cpp
时代，现文件已更名 meeting_summary.cpp，行号可能有漂移）

## A. 官方例程机制清单

| # | 机制 | 位置 |
|---|---|---|
| 1 | 多卡：dev_ids 逐卡 Handle + KV 按卡分配，forward_next 多卡 CPU 回读 token 广播 | utils/qwen.py:21-23, 81-90, 209-214 |
| 2 | addr_mode 0/1 张量管理自适应；io_alone 时 past_k/v 原地引用免拷贝 | qwen.py:29-38, 84-90 |
| 3 | 动态 shape 检测，first 阶段按实际 token 长度构造输入 | qwen.py:43, 152-157 |
| 4 | bfloat16 mask 特判 0xC61C / f16 -10000 | qwen.py:50-52 |
| 5 | 模型内采样子网检测（greedy_head / penalty_sample_head） | qwen.py:54-59, 195-199 |
| 6 | 张量复用：一次性预分配最大尺寸输入/输出 | qwen.py:93-102 |
| 7 | 流式输出：逐词 decode + 替换符攒 token + FTL/TPS 统计 | qwen.py:281-305 |
| 8 | 结构化 API：finish_reason=stop/length；输入超 SEQLEN-5 预检提示 | qwen.py:307-350, 284-285 |
| 9 | 分块：RecursiveCharacterTextSplitter（8300 字符，句号边界修正） | main.py:15-23, 81 |
| 10 | 字段化纪要 prompt（27 行模板，system+user 双注入） | prompt.txt:1-27; main.py:36-42, 70-79 |
| 11 | CLI 与落盘：--bmodel/--dev_id/--text_path，output.txt + 耗时 | main.py:112-133 |

## B. 优化点

### P1 建议采纳（小改动高收益）

| 编号 | 优化点 | 现状 vs 官方 | 改动位置 | 工作量 |
|---|---|---|---|---|
| P1-1 | **字段化纪要模板**：缺会议信息/议程/行动项完成时间等字段；官方 prompt.txt 全字段且 system+user 双注入 | 三段式 vs 全字段 | audio_summary.cpp:31-34 默认模板 + config/audio_summary.json | 小 |
| P1-2 | **修复硬切 UTF-8 截断 bug**：单行超预算按"字节×token 比例" substr，可能切断多字节汉字导致 tokenizer 乱码 | 官方按分隔符切无此问题 | audio_summary.cpp:130-132，按 UTF-8 边界对齐 | 小 |
| P1-3 | **流式回调透传**：AudioSummaryRun 传 cb=nullptr，而 CTextLLM::generate 已支持逐词回调；加 Ctrl "set_partial_cb" 即可实时上屏 | 官方逐 token yield | audio_summary.cpp:103-104 + Ctrl | 小 |
| P1-4 | **推理性能统计**：官方打印 FTL/TPS，我们全程无计时 | 无 vs 有 | text_llm.h::generate 计时 → Ctrl "get_model_info" 扩展 | 小 |
| P1-5 | **budget 耗尽兜底**：模板开销吞掉预算时 splitByBudget 整文入块，runOnce 必报错；应硬切并告警 | — | audio_summary.cpp:113 | 小 |

### P2 可选

| 编号 | 优化点 | 说明 | 工作量 |
|---|---|---|---|
| P2-1 | penalty/top_p 采样子网路径 | 总结场景 greedy 够用；CTextLLM 泛化到对话/创作时再补 | 中 |
| P2-2 | io_alone 免拷贝（addr_mode=1 时 KV 原地引用） | 当前每 token 每层 2 次 d2d 回写（text_llm.h:226-227），预估 5-10% TPS 收益 | 中 |
| P2-3 | finish_reason 语义（stop/length） | 调用方无法得知截断还是自然结束 | 小 |
| P2-4 | demo CLI/落盘体验（--text_path 输出保存 + 耗时） | main.cpp:138-149 仅 printf | 小 |

### P3 不建议采纳

| 项 | 理由 |
|---|---|
| 多卡并行（qwen.py:21-23, 81-90） | 板端单芯片场景用不上；全链单 dev_id 设计，改造大收益零 |
| langchain + transformers 依赖 | 我们 C++ 零依赖、按 token 精确预算分块 + 迭代 reduce 上限 4 轮，比官方固定 8300 字符更强 |
| 动态 shape / addr_mode 0 兼容 | 全静态 shape 是既定架构决策，为兼容旧模型不值 |

### 已等价或更优（无需动）

bf16 mask 值（text_llm.h:186）、逐词替换符检测（text_llm.h:141）、双 token
去重叠（text_llm.h:142-147）、KV 偏移写法、SEQLEN 预算预留
（audio_summary.cpp:67-73）；reduce 拼接带 "\n\n" 分隔优于官方无分隔直拼
（main.py:105-107）。

## C. 结论

官方例程对我们真正的增量只有三件事：**字段化纪要 prompt、UTF-8 硬切修复、
流式回调透传**（均小改动高收益）；其余调度层我们已等价或更优，不建议引入
其 langchain 依赖与多卡/动态 shape 复杂度。
