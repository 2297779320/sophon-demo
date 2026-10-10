# 模型预算表与选型指引

本框架**不转换、不内置任何 bmodel**：模型文件全部由外部提供
（LLM-TPU / tpu-mlir 等转换产物），框架只消费。框架侧对模型的全部约定为：

1. **多文件分离 + JSON 映射**：每个 bmodel 一个文件，`config/audio_link.json`
   的 `models` 节按"模型名 → path + roles 关键字"映射（角色关键字匹配子网名，
   未匹配子网仅登记暴露，见 `docs/README.md` 模型组织一节）；
2. **chat 格式子网命名与 I/O 约定**：LLM 类模型（asr/summary）须为
   LLM-TPU chat.cpp 导出格式（embedding / block_i / block_cache_i /
   audio / lm_head / greedy_head 等子网，SEQLEN 等维度由子网 stages[0]
   推导），详细 I/O 契约见 `docs/dataflow.md`；
3. **全部静态 shape**：框架按静态 shape 补齐 padding 后整块拷入，不做动态 shape 适配。

转换命令不在本文档范围，qwen3_asr 系列直接使用
**LLM-TPU 仓库 `models/Qwen3_ASR` 的 README**（含各芯片量化/编译命令与
config 产物清单）；CAM++ / fsmn_vad 走 tpu-mlir 常规 onnx→bmodel 流程。

## 1. 多文件分离架构与显存策略

> ⚠️ **无融合模型**：所有模型均为独立 bmodel 文件、独立 bmrt 上下文；
> 不支持/不规划单文件多子网混合 bmodel。显存占用 = Σ(常驻模型) + 切换窗口峰值。

```
常驻（音频流期间）：        按需（会议总结窗口）：
  fsmn_vad（小，常驻）        ── 卸载 asr ──▶ 加载 summary LLM
  campplus（小，常驻）        ── 总结生成 ──▶ MeetingSummaryDeinit
  qwen3_asr（大，常驻）       ── 重载 asr ──▶ 恢复音频流
```

- **asr 常驻**：qwen3_asr 是音频流的大头，Init 即加载、常驻显存；
- **summary 按需加载**：总结是低频非流式操作，text-only LLM 仅在总结窗口加载；
- **unload-load 切换时序**：显存受限板（两套大模型无法同时驻留）按
  `卸 asr → SummaryInit → Run → Deinit → 重载 asr` 手动时序驱动
  （`docs/API.md` 会议总结一节）；卸载期间 VAD/声纹/聚类照常
  （宿主内存状态不受影响），段无文本不追加转写行。

**显存峰值模型**（估算用）：

- 稳态峰值 = max(常驻集合)（fsmn_vad + campplus + asr）；
- 切换窗口峰值 = max(常驻集合 − asr + summary LLM, 稳态峰值)，
  即切换窗口内"卸载已完成 + 新模型已加载"的共存瞬态；
- 另计入所有模型共用的设备运行时开销（bmlib/bmruntime 上下文，实测校准）。

## 2. 预算表（模板）

> "显存占用"列**待板端实测**：bmlib 设备内存接口，或部署前后 `bm-smi`
> 显存读数对比；量化格式为 LLM-TPU 转换产物常见档位，以实际转换配置为准。
>
> 配置资产依赖（运行期非显存项）：`mel_filters.npy`（128×201）两仓均无现成文件，
> 需 whisper mel_filters.npz 转换或模型方提供（见 README 资产就绪清单）；
> vad CMVN npy 为可选能力，当前配置已移除引用（fsmn_vad 官方不用 CMVN）。

| 模型文件 | 角色 | 量化格式 | 显存占用 | 来源 | 许可 | 备注 |
|---|---|---|---|---|---|---|
| qwen3_asr_0.6b_bf16.bmodel | asr_llm + asr_audio | bf16 | 待板端实测 | LLM-TPU models/Qwen3_ASR | Apache-2.0（Qwen3 系列同源） | 0.6B 基线，精度优先 |
| qwen3_asr_0.6b_int8.bmodel | asr_llm + asr_audio | int8 | 待板端实测 | LLM-TPU models/Qwen3_ASR | Apache-2.0 | 8GB 级以下首选 |
| qwen3_asr_1.7b_bf16.bmodel | asr_llm + asr_audio | bf16 | 待板端实测 | LLM-TPU models/Qwen3_ASR | Apache-2.0 | 大显存板精度升级档 |
| qwen3_asr_1.7b_int8.bmodel | asr_llm + asr_audio | int8 | 待板端实测 | LLM-TPU models/Qwen3_ASR | Apache-2.0 | 大显存板速度/显存折中 |
| qwen3_0.5b_summary_int8.bmodel | summary（text-only LLM） | int8 | 待板端实测 | LLM-TPU Qwen3 系列转换 | Apache-2.0 | chat 格式、无 audio 子网，0.5B 级够用 |
| campplus.bmodel | speaker | int8/bf16 | 待板端实测 | sophon-demo sample/CAM++ 同源 | 见上游 | 声纹 embedding，192 维 |
| fsmn_vad.bmodel | vad | int8/bf16 | 待板端实测 | funasr speech_fsmn_vad_zh-cn-16k-common bmodel 化 | 见上游（Modellicense） | 流式协议与 funasr 一致 |

注意：`config/audio_link.json` 的 `models` 键名（asr/spk/vad 等）可自行改名，
但 roles 关键字与子网名的对应关系必须成立；asr 与 summary 两个模型
不能共用一个模型键（各自独立 hub 实例、独立 bm_handle）。

## 3. 选型指引

**8GB 级以下目标板**（BM1684X SoC 典型可用显存扣除系统占用后有限）：

- asr 选 **0.6B int8**：识别延迟与显存的最优折中，留足 vad/spk 与切换窗口余量；
- summary 选 **0.5B 级 int8**：纪要模板任务（结构化抽取+归纳）对模型容量
  敏感度低，0.5B 级 int8 实测足够；转写文本经 map-reduce 分块，单次输入
  预算受 MAX_INPUT_LENGTH 约束与模型档位无关；
- 16GB 级及以上再考虑 asr 1.7B（int8 起步）。

**通用原则**：

- 先 int8 后 bf16：int8 不满足精度要求再回 bf16（显存翻倍）；
- 切换窗口共存峰值 ≤ 板端可用显存，且预留 ≥ 20% 余量给运行时碎片；
- summary 模型档位跟随 asr 降档，不与 asr 同档（两者永不同时驻留）。

## 4. 精度回归方法

模型换档（0.6B↔1.7B、int8↔bf16、换 summary 模型）后必须做回归：

1. **基线**：固定一批 wav（建议覆盖：安静/噪声/多人重叠/方言口音，
   单条 ≥ 30s 与 ≤ 3s 两类），用 bf16 档先跑出基线转写与总结文本；
2. **ASR 指标**：同批 wav 推新档位，计算 **WER / 中文字错率 CER**
   （`python -m jieba` 分词后字级对齐，或直接用 jiwer/sclite）；
   经验阈值：CER 相对基线劣化 < 2 个百分点可接受，超过需回退 bf16；
3. **声纹/聚类**：同一批音频的说话人划分（diarize 输出）与质心档位无关
   （campplus 不换），换 campplus 时检查同说话人余弦相似度分布不塌缩；
4. **总结指标**：固定转写输入对比新旧总结文本，人工核对
   【会议摘要】【行动项】字段完整性（纪要无标准答案，字段级抽查即可）；
5. **性能口径**：同批统计 `last_tps`（Ctrl "get_model_info"）与整段
   端到端延迟，量化换档通常伴随吞吐变化，一并记录进上表"备注"列。
