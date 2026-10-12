# 模型转换需求清单（BM1684X，用户自行转换）

> 唯一事实来源：框架代码的 I/O 期望（cpp/links/*、model_hub deriveAsrDims）。
> 通用要求：全部 **BM1684X** 编译（tpu-mlir `--chip bm1684x` 或默认）；**静态 shape**
> （唯一例外：spk 保持官方 dynamic，见 §2）；
> 产出 bmodel 统一放 `models/`（文件名与 config `models.*.path` 一致，路径相对启动目录）；
> 子网命名必须命中 config roles 关键字（不命中启动即报错，见 ValidateConfig）。
>
> **来源、逐条转换命令、参数与精度回归数据集**：完整版见
> [`docs/model_provenance.md`](model_provenance.md)。本文件是「框架对模型的 I/O 需求」视角。

## 1. vad——fsmn_vad（已验证签名，与框架适配完成）

| 项 | 要求 |
|---|---|
| 源模型 | ModelScope `iic/speech_fsmn_vad_zh-cn-16k-common-onnx` 的 `model_quant.onnx` |
| ⚠️ 前置 | 官方 `model_quant.onnx` 是 **ORT 动态量化**（`DynamicQuantizeLinear`/`MatMulInteger`），tpu-mlir **不支持**这两个算子。须先用 `modelSource/dequant_onnx.py` 反量化为等价 float ONNX（已验证：logits cos=0.9994、逐帧判决 100% 一致） |
| 实测签名 | 输入 `speech` [1,**20**,400]（80 fbank × LFR5）+ `in_cache0..3` [1,128,19,1]；输出 `logits_Softmax_f32` [1,**20**,248]（pdf，sil_pdf_ids=[0]）+ `out_cache0..3` [1,128,19,1] |
| 转换（静态 T=20） | 见 `modelSource/fsmn_T20/convert_T20.sh`：`model_transform.py --model_def fsmn_vad_float.onnx --input_shapes [[1,20,400],[1,128,19,1]×4] --output_names logits,out_cache0..3` → `model_deploy.py --quantize F16 --processor BM1684X`（**勿加 --dynamic**） |
| T 的取值 | **T = `vad.chunk_ms`/10 = 20 帧（200ms）**。T 必须与 `chunk_ms` 一致；不一致不会崩（FIFO 会跨块凑组）但稳态多攒一段延迟，启动时 WARN |
| 配套资产 | 同 repo `am.mvn` → `config/audio/am.mvn`（JSON `vad.cmvn_file` 已引用） |
| 验收 | `model_tool --info`：`static`，input [1,20,400] / output [1,20,248]；板端跑 `audiolink_demo` 出段且 `get_transcript` 有文本 |

## 2. spk——CAM++ 声纹（**保持官方 dynamic**）

| 项 | 要求 |
|---|---|
| 源模型 | ModelScope `iic/speech_campplus_sv_zh-cn_16k-common`（经 3D-Speaker 导出 ONNX） |
| 官方配方 | `sophon-demo/sample/CAM++/scripts/gen_fp32bmodel_mlir.sh`：`--input_shapes [[1,600,80]] --dynamic --quantize F32 --chip bm1684x --disable_layer_group` |
| 框架期望 | input `[B,T,80]` float32 fbank（**dynamic 变长**）；output 末维 = embedding（192） |
| 状态 | 现有 `campplus.bmodel`（v1.11、`[max:1,max:600,max:80]` dynamic）**即官方产物，无需重转** |
| config | `models.spk.path` + roles `speaker: "campplus\|speaker"` |
| 说明 | 声纹输入是按说话人段的 fbank，段长天然可变——官方即用 dynamic；框架 launch 用**实际 rows**，动态子网原生支持。详细导出步骤见 `model_provenance.md` §3 |

## 3. asr——qwen3_asr（chat 格式 + audio 子网）

| 项 | 要求 |
|---|---|
| 源模型 | ModelScope `Qwen/Qwen3-ASR-1.7B`（**当前档位 1.7B**；0.6B 为降档候选） |
| 现成产物 | 官方 bmodel `qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_*.bmodel`（3.81GB，**当前采用**） |
| 自转命令 | LLM-TPU `models/Qwen3_ASR/README.md`：`llm_convert.py -m <模型> -s 512 --max_input_length 256 --quantize bf16 -c bm1684x --out_dir qwen3_asr --qwen_asr` |
| 子网命名（必须命中 roles） | `embedding`、`block_0..N`、`block_cache_0..N`、**`audio`**（mel 编码器）、`lm_head`、`greedy_head`（可选）。实测 60 子网、28 层、hidden=2048 |
| audio 子网 I/O | input `[1,1,128,100]`（mel128）；每块输出 13 token embed（1 秒音频 ≈ 13 token） |
| 维度 | SEQLEN/MAX_INPUT_LENGTH/HIDDEN_SIZE/NUM_LAYERS/AUDIO_LENGTH 由 net_info **自动推导**（JSON dims 保持 auto） |
| config | `models.asr.path` + roles `asr_llm: "embedding\|block_\|lm_head\|greedy_head"`、`asr_audio: "audio"` |
| 验收 | `Ctrl "asr" "get_model_info"` 返回推导维度；`Ctrl "asr" "list_nets"` 各子网角色齐全 |

## 4. summary——text-only LLM（chat 格式，无 audio 子网）

| 项 | 要求 |
|---|---|
| 源模型 | **Qwen1.5-7B int4 6k**（官方 `sophon-demo/application/meeting_summary` 默认档） |
| 现成产物 | `qwen1.5-7b_int4_6k_1dev.bmodel`（4.94GB，**当前采用**）；tokenizer 用官方 `token_config.zip`（与项目 `config/summary/summary_vocab.json` **逐字节一致**） |
| 子网命名 | `embedding`、`block_0..N`、`block_cache_0..N`、`lm_head`、`greedy_head`（可选） |
| SEQLEN | 6144（6k 档）；需满足 `prefill + max_new_tokens ≤ SEQLEN−128`（4k 档则 ≤3456） |
| config | `models.summary.path` + roles `asr_llm`（复用关键字映射） |
| 验收 | `MeetingSummaryInit` 成功 + 日志 `summary init: llm ok (io_alone=?, layers=N)` |

## 常见转换坑

1. **勿加 --dynamic**（spk 例外）：框架按静态 shape 适配（vad 固定 T=20，前端攒满 20 帧才 launch、
   尾部补零凑组；asr/summary prefill 补齐 MAX_INPUT_LENGTH）；**唯一例外是 spk**——官方 CAM++ 配方
   即为 `--dynamic`，保持不动。
2. **子网名必须保留**：转换脚本若重命名/合并子网，roles 映射会 miss（启动报
   `no net matched role`）；
3. **greedy_head 可选**：无此子网时 `lmhead_with_topk=false`，decode 走 lm_head
   全词表 + C++ argmax（已支持）；
4. **io_alone**：新工具链默认 addr_mode=1（KV 原地引用，prefill 免 d2d）——init 日志
   会打印 `io_alone detect`，addr_mode=0 模型自动走 BASIC 回退；
5. **INT8 量化**：8GB 级显存建议 asr/summary 用 int8 档位；转换后须做 CER 回归
   （对比 bf16 基线，劣化 <2pp），方法见 `docs/model_budget.md`。

## 产出核对清单（models/ 目录）

```
models/
├── fsmn_vad.bmodel                     # [1,20,400]+cache → [1,20,248]（静态 T=20）6.5MB
├── campplus.bmodel                     # [max:1,max:600,max:80] → embedding[1,192]（dynamic）32MB
├── qwen3_asr.bmodel                    # chat 格式 + audio 子网（Qwen3-ASR-1.7B）3.81GB
└── qwen1.5-7b_int4_6k_1dev.bmodel      # chat 格式 text-only（SEQLEN=6144）4.94GB
config/audio/am.mvn                           # vad MVN 资产
models/vad_config.yaml                  # funasr 运行时参数契约（参考，非转换配置）
```
