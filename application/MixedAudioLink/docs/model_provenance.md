# 模型溯源与转换手册（BM1684X / MixedAudioLink）

> 本文档回答四个问题：**模型从哪来**、**怎么转成 bmodel**（逐条命令与参数）、
> **参数如何落到配置**、**要做精度调整需要什么数据集与指标**。
> 所有转换均在 x86 主机 + `sophgo/tpuc_dev` 容器内完成；SoC 只消费 `.bmodel`。
>
> 核对时间：2026-10-10。bmodel 元信息均由 `model_tool --info` 实测复核。

---

## 0. 总览：四个模型一览

| 角色 | 文件（`models/`） | 档位 / 量化 | 静态 I/O 关键 shape | 来源 | 转换方式 | 工具链版本 |
|---|---|---|---|---|---|---|
| **vad** | `fsmn_vad.bmodel` | fsmn_vad（小），F16 | in `speech[1,20,400]` + `in_cache0..3[1,128,19,1]`；out `logits[1,20,248]` + `out_cache0..3` | ModelScope `iic/speech_fsmn_vad_zh-cn-16k-common-onnx` | **本仓自行转换**（需先反量化，见 §2） | B.2.2+v1.30.beta.0 |
| **spk** | `campplus.bmodel` | CAM++，F32 | in `feature[max:1,max:600,max:80]`（**dynamic**）；out `embedding[1,192]` | ModelScope `iic/speech_campplus_sv_zh-cn_16k-common`（经 3D-Speaker 导出 ONNX） | **官方现成产物 / 官方脚本**（`--dynamic`，见 §3） | B.2.2+v1.11.beta.0 |
| **asr** | `qwen3_asr.bmodel` | **Qwen3-ASR-1.7B**，bf16，seq512 | 60 子网：`audio` / `embedding` / `embedding_cache` / `block_cache_0..27` / `lm_head` / `block_0..27` | ModelScope `Qwen/Qwen3-ASR-1.7B` | **官方现成 bmodel 下载**（如需自转见 §4） | B.2.2+v1.27.beta.0 |
| **summary** | `qwen1.5-7b_int4_6k_1dev.bmodel` | Qwen1.5-7B，int4，SEQLEN=6144 | 69 子网 / 32 层：`embedding`/`embedding_cache`/`lm_head`/`greedy_head`(dynamic)/`block_0..31`/`block_cache_0..31` | `sophon-demo/application/meeting_summary` | **官方现成 bmodel 下载**（见 §5） | B.2.2+v1.9.beta.0 |

> ⚠️ **纠正一处文档偏差**：`docs/model_budget.md` 把 asr 记为「0.6B 档位优先」，但
> **实物是 1.7B**——实测 `lm_head` 输入 `hidden_states[1,2048]`（0.6B 的 hidden=1024）、
> 28 层、seq512、3.81 GiB，与官方产物
> `qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_20260408_094600.bmodel` 尺寸一致。
> 本地 `modelSource/Qwen3-ASR-0.6B/` 是另一路（BM1688 seq256）源，**当前主链用的是 1.7B**。
> 显存预算需按 1.7B 重新评估（见 §7.4）。

### 角色关键字匹配（`config/*.json` 的 `roles`）

`CModelHub::matchKeywords` 用**子串匹配**（`net_name.find(kw) != npos`，`|` 分隔任一命中）：

| 模型 | roles | 命中的子网名 |
|---|---|---|
| asr | `asr_llm: "embedding\|block_\|lm_head\|greedy_head"`、`asr_audio: "audio"` | `embedding`/`embedding_cache`/`block_0..27`/`block_cache_0..27`/`lm_head`/`audio` |
| spk | `speaker: "campplus\|speaker"` | `campplus` |
| vad | `vad: "fsmn_vad\|speech_fsmn"` | `speech_fsmn_vad_T20` |

⇒ 转换时**不要重命名子网**，否则启动直接报 `no net matched role`（`ValidateConfig` 拒绝）。

---

## 1. 转换环境准备（x86 主机）

```bash
# 1) 拉取官方开发镜像（tpu-mlir 与镜像有绑定关系）
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name tpuc -v $PWD:/workspace -it sophgo/tpuc_dev:latest

# 2) 镜像内激活 tpu-mlir 环境（本仓实测路径；官方 release 包为 source ./envsetup.sh）
export TPUC_ROOT=/workspace/tpu-mlir/install
export PATH=$TPUC_ROOT/bin:/workspace/tpu-mlir/python/tools:/workspace/tpu-mlir/python/utils:$PATH
export PYTHONPATH=$TPUC_ROOT/python:/workspace/tpu-mlir/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TPUC_ROOT/lib:/workspace/tpu-mlir/capi/lib:$LD_LIBRARY_PATH
# 缺 libortools.so.9 时就是因为没设 LD_LIBRARY_PATH
python3 -c "import pymlir; print('pymlir OK')"   # 自检
model_transform.py --help | head -3              # 自检
```

**版本说明**：本仓现有 bmodel 版本混用（campplus v1.11 / 旧 vad v1.13 / asr v1.27 / 新 vad v1.30），
但 bmodel 格式均为 `B.2.2`，运行期（bmruntime）兼容，**不强制统一工具链版本**。
若需与某模型严格对齐，官方发布名里已标注版本（如 `tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414`）。

---

## 2. vad —— fsmn_vad（流式 VAD，静态 T=20）

### 2.1 来源

| 项 | 值 |
|---|---|
| 模型主页 | https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx |
| 下载文件 | `model_quant.onnx`（+ `config.yaml` / `am.mvn` / `test.sh`） |
| 官方契约 | frontend `fs=16000 / hamming / n_mels=80 / frame_length=25 / frame_shift=10 / dither=0 / lfr_m=5 / lfr_n=1`；encoder `input_dim=400 / output_dim=248` |

### 2.2 ⚠️ 关键坑：官方 `model_quant.onnx` 无法被 tpu-mlir 直接转换

`model_quant.onnx` 是 **onnxruntime 动态量化**产物（`producer: onnx.quantize`），图里有
12 处 `DynamicQuantizeLinear + MatMulInteger`，tpu-mlir 报：

```
RuntimeError: Op not support: {'MatMulInteger', 'DynamicQuantizeLinear'}
```

**解法：图手术反量化 → 干净 float ONNX**（脚本 `modelSource/dequant_onnx.py`）。
固定模式（每处 MatMul 一组）：

```
DynamicQuantizeLinear(x) -> x_q, x_scale, x_zp
Mul(x_scale, W_scale)    -> combined
MatMulInteger(x_q, W_q, x_zp, W_zp) -> acc
Cast(acc) -> acc_f
Mul(acc_f, combined)     -> y
Add(bias, y)             -> out
        ↓ 还原为
MatMul(x, W_fp32)        # W_fp32 = (W_q - W_zp) * W_scale，沿用 m2 输出名（下游 Add 不用改）
```

```bash
python3 modelSource/dequant_onnx.py \
    speech_fsmn_vad_zh-cn-16k-common-onnx/model_quant.onnx \
    fsmn_vad_float.onnx
```

> 手工塞入新 initializer 的**拓扑序坑**：新 `W_fp32` 必须手动 append 进 `graph.initializer`，
> 否则 checker 报 `input ... is not output of any previous nodes`。
>
> **正确性验证**（`modelSource/verify_dequant.py`，onnxruntime 同输入对拍）：
> logits `cos=0.99937`、`out_cache0..3 cos=0.9993~0.9999`、**逐帧判决 100% 一致**——
> 残差即 int8 量化误差，手术正确。

### 2.3 转换命令（静态 T=20）

脚本：`modelSource/fsmn_T20/convert_T20.sh`（`T=${T:-20}` 可覆盖）。

```bash
# [1/2] ONNX -> MLIR（静态，无 --dynamic）
model_transform.py \
    --model_name speech_fsmn_vad_T20 \
    --model_def  fsmn_vad_float.onnx \
    --input_shapes [[1,20,400],[1,128,19,1],[1,128,19,1],[1,128,19,1],[1,128,19,1]] \
    --input_types float32,float32,float32,float32,float32 \
    --output_names logits,out_cache0,out_cache1,out_cache2,out_cache3 \
    --mlir fsmn_vad_T20.mlir

# [2/2] MLIR -> bmodel（F16，带精度校验）
model_deploy.py \
    --mlir fsmn_vad_T20.mlir \
    --quantize F16 \
    --processor BM1684X \
    --test_input   fsmn_T20_in.npz \
    --test_reference fsmn_vad_T20_top_outputs.npz \
    --tolerance 0.99,0.90 \
    --model fsmn_vad_T20_bm1684x_f16.bmodel
```

**参数说明**：

| 参数 | 取值 | 原因 |
|---|---|---|
| `--input_shapes` | `[[1,20,400],[1,128,19,1]×4]` | `speech` 是 [1,T,400]（80 fbank × LFR5）；4 路 cache 为 FSMN 记忆，shape 固定 |
| `--output_names` | `logits,out_cache0..3` | 必须显式指定，否则输出名带 Slice/Softmax 后缀，roles 虽按子串仍可命中但下游列名要对齐 |
| `--quantize` | `F16` | FP16 有指数位无需校准；8GB 板可选 INT8（需校准集，见 §7.5） |
| `--processor` | `BM1684X` | 目标芯片 |
| `--dynamic` | **不要加** | 全流程静态（见下） |
| `--test_input` 位置 | 放 **deploy 阶段** | 放 transform 阶段会触发 `args.test_result` 断言失败 |

**T 的选择依据**：`T = vad.chunk_ms / 10 = 20`（200ms）。上游 `pcm_link` 每 tick 固定拉 3200 样本
（200ms），前端 LFR 每块产出 14/20/20… 帧（首块因 carry-over 少），VAD 内部用**特征 FIFO 攒满 T 行才 launch**。
T 与 `chunk_ms` 不一致不会崩（FIFO 跨块凑组），但稳态会多攒一段延迟，启动时 WARN。

**为什么静态**：BM1684X SoC 上动态 bmodel 在**流式因果 FSMN** 上有一个隐患——
若运行时忽略 T 按满维跑，首 T 行输出仍正确（因果），但 `out_cache` 会被输入缓冲尾部陈旧帧污染，
导致下一块判决漂移。静态 T 每次前进量恒定，无此风险。

### 2.4 配套资产

| 资产 | 来源 | 目标 | 引用 |
|---|---|---|---|
| `am.mvn`（CMVN） | 同 repo（8040 B） | `config/audio/am.mvn` | `config/audio/audio_link.json` → `vad.cmvn_file` |
| `config.yaml`（运行时参数契约） | 同 repo | 参考 | 参数已落到 `vad` 节（见 §6） |

> 注：`models/vad_config.yaml` 是 funasr 的**运行时模型配置**，不是 tpu-mlir 转换配置——
> 转换参数在 `test.sh` / `convert_T20.sh`。fsmn_vad 官方**不使用 CMVN**，`cmvn_file` 为可选能力。

### 2.5 验收

```bash
model_tool --info models/fsmn_vad.bmodel | head -12
# 期望：net 0 [speech_fsmn_vad_T20]  static
#       input  speech [1,20,400] + in_cache0..3 [1,128,19,1]
#       output logits_Softmax_f32 [1,20,248] + out_cache0..3 [1,128,19,1]
```

---

## 3. spk —— CAM++ 声纹（dynamic，官方配方）

### 3.1 官方导出指南（`sophon-demo/sample/CAM++/docs/Campplus_Export_Guide.md`）

依赖官方仓库 [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker)，**在 x86 主机编译**。

```bash
# TPU-MLIR（BM1684X）
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/campplus/mlir_1024/tpu-mlir_bm1684x.tar.gz
# docker
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
cd /workspace/tpu-mlir_vx.y.z-<hash>-<date> && source ./envsetup.sh
```

### 3.2 导出 ONNX

```bash
git clone https://github.com/modelscope/3D-Speaker
```

**必须修改源码**（否则导出 ONNX 与 PyTorch 结果不对齐）——`speakerlab/models/campplus/layers.py` 的 `seg_pooling`：

```diff
     def seg_pooling(self, x, seg_len=100, stype='avg'):
         if stype == 'avg':
-            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
+            seg_ori = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
+            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True, count_include_pad=False)
+            assert((seg_ori == seg).all())
```

```bash
pip3 install -r tools/requirements.txt
python speakerlab/bin/export_speaker_embedding_onnx.py \
    --experiment_path . \
    --model_id iic/speech_campplus_sv_zh-cn_16k-common \
    --target_onnx_file campplus.onnx
```

### 3.3 编译 bmodel（**官方脚本本身即 `--dynamic`**）

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x      # 或 cv84x6
```

脚本内容等价于：

```bash
model_transform.py --model_name campplus \
    --model_def ../models/onnx/campplus.onnx \
    --input_shapes [[1,600,80]] --dynamic \
    --mlir campplus_1b.mlir

model_deploy.py --mlir campplus_1b.mlir \
    --quantize F32 --chip bm1684x --dynamic \
    --disable_layer_group \
    --model campplus_bm1684x_fp32_1b.bmodel
```

**参数说明**：

| 参数 | 取值 | 原因 |
|---|---|---|
| `--input_shapes` | `[[1,600,80]]` | 600 帧 ≈ 6s 段（`seg_pooling` 的 seg_len=100，`ceil_mode`）；80 维 fbank |
| `--dynamic` | **有** | 声纹输入是**按说话人段**的 fbank，段长天然可变；官方配方即动态 |
| `--quantize F32` | FP32 | 声纹对数值敏感（要算余弦相似度），官方用 FP32 不做量化 |
| `--disable_layer_group` | 有 | 官方脚本固定项 |

> **决策记录**：项目**保持官方 dynamic 做法，不改成静态**。现有 `campplus.bmodel`
> （v1.11、`[max:1,max:600,max:80]`）与官方脚本产物一致，无需重转。
> 框架侧 launch 用**实际 rows**，动态子网原生支持。

### 3.4 官方现成产物

```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/CAM++/...   # 见 sample/CAM++/scripts/download.sh
```
（BM1684X 现成可下载；BM1688 需重转）

### 3.5 验收

```bash
model_tool --info models/campplus.bmodel | head -12
# 期望：net 0 [campplus]  dynamic
#       input  feature [max:1,max:600,max:80]
#       output embedding_BatchNormalization [1,192]
```

---

## 4. asr —— Qwen3-ASR-1.7B（chat 格式 + audio 子网）

### 4.1 来源

| 项 | 值 |
|---|---|
| 模型主页 | https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B |
| 本地已存源 | `modelSource/Qwen3-ASR-0.6B/`（另档；`model.safetensors` 1.88GB、`config.json`） |
| 官方现成 bmodel | `/ext_model_information/LLM/LLM-TPU/qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_20260408_094600.bmodel`（3.81GB，**与项目文件同尺寸**） |
| 转换工具链 | LLM-TPU `tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414.tar.gz`（支持 Qwen3-ASR） |

### 4.2 路径 A：官方现成 bmodel（当前采用）

```bash
# 用本仓 SFTP 工具（容器内跑，见 §8）
python3 dfss_fetch.py get \
  /ext_model_information/LLM/LLM-TPU/qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_20260408_094600.bmodel \
  /workspace/MixedAudioLink/models/qwen3_asr.bmodel
```

### 4.3 路径 B：自行编译（LLM-TPU `models/Qwen3_ASR/README.md`）

```bash
# 1) ModelScope 下载源模型
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir Qwen3-ASR-1.7B
#    0.6B 档： modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir Qwen3-ASR-0.6B

# 2) 容器（同上）
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest

# 3) 拉支持 Qwen3-ASR 的 tpu-mlir 发布包并激活
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Qwen/qwen3_asr/tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414.tar.gz
tar zxf tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414.tar.gz
cd tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414 && source ./envsetup.sh

# 4) 编译（-s=SEQLEN，--max_input_length=音频 token 预算，--qwen_asr 启用 ASR 分支）
pip3 install torch==2.4.1 transformers==4.57.6 qwen_asr -U   # 如提示版本问题
llm_convert.py -m /workspace/Qwen3-ASR-1.7B -s 512 --max_input_length 256 \
               --quantize bf16 -c bm1684x --out_dir qwen3_asr --qwen_asr
# 产出 qwen3_asr/qwen3-asr-xxx.bmodel 与 config/
```

**参数说明**：

| 参数 | 取值 | 说明 |
|---|---|---|
| `-s` | `512` | SEQLEN；声明式上限，决定 KV cache 尺寸与 attention_mask 长度（实测 mask `[1,1,1,513]` = 512+1） |
| `--max_input_length` | `256` | 音频 token 预算（不指定则默认等于 `-s`）；**1 秒音频 ≈ 13 token**（官方口径） |
| `--quantize` | `bf16` / `int8` | 8GB 级板优先 int8（需校准，见 §7.5） |
| `-c` | `bm1684x` | 目标芯片（`bm1688` 则使用 2core 产物） |
| `--qwen_asr` | 有 | 启用 Qwen3-ASR 专用导出（含 audio 子网、thinking 分支） |

### 4.4 子网约定（必须命中 roles）

实测 60 个子网：

```
audio                 in  input_states [1,1,128,100]  ->  [1,13,2048]（mel 编码器，13 token/block）
embedding             in  input_ids [1,512]           ->  [1,512,2048]
embedding_cache       in  input_ids [1,1]             ->  [1,1,2048]
lm_head               in  hidden_states [1,2048]      ->  token_id [1,1]（含采样子网，subnet=2）
block_0..27           prefill（28 层）
block_cache_0..27     decode（KV cache：history_k/v [1,512,8,128]，attention_mask [1,1,1,513]）
```

> `greedy_head` **可选**：无此子网时 `lmhead_with_topk=false`，decode 走 lm_head 全词表 + C++ argmax（已支持）。
> 维度（SEQLEN / MAX_INPUT_LENGTH / HIDDEN_SIZE / NUM_LAYERS / AUDIO_LENGTH）由 `net_info` **自动推导**
> （`config/*.json` 的 `dims.*` 保持 `auto`），故 `model_budget.md` 里 asr 档位的 dimension 不必手工填。

### 4.5 验收

```bash
# 一次列全：子网数、audio/embedding/lm_head/block 角色是否齐
model_tool --info models/qwen3_asr.bmodel | grep -E "^net [0-9]+:" | wc -l   # 期望 60
# 端到端：启动后 Ctrl "asr" "get_model_info" 返回推导维度；Ctrl "asr" "list_nets" 角色齐全
```

---

## 5. summary —— Qwen1.5-7B int4 6k（chat 格式，text-only）

### 5.1 来源（`sophon-demo/application/meeting_summary`）

官方例程**开箱用现成 bmodel**，不做转换：

```bash
# 工程目录
#  ├── models/qwen1.5-7b_int4_6k_1dev.bmodel
#  └── scripts/download_model.sh  |  download_tokenizer.sh

chmod -R +x scripts/
./scripts/download_tokenizer.sh     # token_config.zip（2.91MB）→ python/utils/
./scripts/download_model.sh         # 4.94GB bmodel
```

两条脚本的底层 URL（可直接用本仓 SFTP 工具）：

```bash
# 模型（4.94GB）
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Qwen1_5/qwen1.5-7b_int4_6k_1dev.bmodel
# tokenizer（2.91MB）
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Qwen1_5/token_config.zip
```

### 5.2 ⭐ tokenizer 与项目词表逐字节相同（关键结论）

解压 `token_config.zip` 得 `vocab.json` / `tokenizer.json` / `tokenizer_config.json`。
与项目 `config/summary/summary_vocab.json` 对比：

| 对比项 | 官方 token_config | 项目 config | 结论 |
|---|---|---|---|
| vocab 项数 | 151643 | 151643 | 一致 |
| max id | 151642 | 151642 | 一致 |
| 逐项相等 | `a == b → True`；共有 token 中 id 不同数 = **0** | | **完全一致** |

- `tokenizer_class = Qwen2Tokenizer`、`model_max_length = 32768`；
- `added_tokens = <|endoftext|> / <|im_start|> / <|im_end|>`，与
  `cpp/meeting_summary.cpp` 硬编码的 `<|im_end|>=151645`、`<|im_start|>=151644` 对上。

⇒ **官方 Qwen1_5 目录下任何档位的 bmodel 都可直接使用，tokenizer 一个字节都不用换。**

### 5.3 可选档位（`/sophon-demo/Qwen1_5/`）

| 文件 | 大小 | SEQLEN | 适配 |
|---|---|---|---|
| `qwen1.5-1.8b_int4_4k.bmodel` | 1.41 GB | 4096 | 显存最省，刚好满足 ≥4096 |
| `qwen1.5-4b_int4.bmodel` | 2.55 GB | — | 折中 |
| **`qwen1.5-7b_int4_6k_1dev.bmodel`** | **4.94 GB** | **6144** | **官方默认，纪要质量最高（当前选用）** |
| `qwen1.5-14b_int4.bmodel` | 8.29 GB | — | 大显存 |
| `qwen2-1.5b_int4_seq512.bmodel` | 1.23 GB | 512 | ✗ seqlen 远小于 4096，不适用 |

**SEQLEN 约束**（`meeting_summary.cpp`）：`max_new_tokens=512`，`prefill + 512 ≤ SEQLEN − 128`
⇒ 4k 档 prefill ≤ 3456、6k 档 prefill ≤ 5504。

### 5.4 验收（实测已通过）

```bash
model_tool --info models/qwen1.5-7b_int4_6k_1dev.bmodel | head -6
# bmodel version: B.2.2+v1.9.beta.0    chip: BM1684X
# 共 69 子网、32 层：
#   embedding / embedding_cache / lm_head / greedy_head(dynamic) / penalty_sample_head
#   block_0..31 / block_cache_0..31
# 文件大小 5,305,080,860 B（4.94 GiB）——与远端声明一致
```

> roles `asr_llm: "embedding\|block_\|lm_head\|greedy_head"` 全部命中；
> `penalty_sample_head` 未命中关键字（仅登记暴露，不影响运行）。
> `greedy_head` 为 **dynamic** 子网，框架按需使用（有则 `lmhead_with_topk=true`）。
>
> ⚠️ **显存提示**：4.94GiB 比 asr（3.81GiB）还大。按 `model_budget.md` 的切换时序，
> 加载 summary 前必须先 `unload asr`；8GB 级板这是主要风险点，需板端实测确认。

---

## 6. 参数配置映射（转换参数 ↔ 运行时 JSON）

### 6.1 vad（`config/audio/audio_link.json` → `vad` 节）

键名对齐 funasr 官方 `config.yaml`（时间统一 `_ms` 后缀）：

| JSON 键 | 官方对应 | 默认 | 作用 |
|---|---|---|---|
| `chunk_ms` | — | 200 | 上游块长；**决定静态 T（T = chunk_ms/10 = 20）** |
| `window_size_ms` | `window_size_ms` | 200 | 判决滑窗（20 帧环形投票） |
| `sil_to_speech_time_ms` | `sil_to_speech_time_thres` | 150 | 起段确认时长 |
| `speech_to_sil_time_ms` | `speech_to_sil_time_thres` | 150 | 停段确认时长 |
| `max_end_silence_ms` | `max_end_silence_time` | 800 | 结束静音上限 |
| `max_start_silence_ms` | `max_start_silence_time` | 3000 | 起始静音上限 |
| `max_seg_s` | `max_single_segment_time/1000` | 20 | 强制切段上限 |
| `do_extend` | `do_extend` | 1 | 端点回看/前瞻补偿 |
| `lookback_time_start_point_ms` | `lookback_time_start_point` | 200 | 起段回看 |
| `lookahead_time_end_point_ms` | `lookahead_time_end_point` | 100 | 末端前瞻 |
| `detect_mode` | `detect_mode` | 1 | 判决模式 |
| `speech_noise_thres` | `speech_noise_thres` | 0.6 | 判决阈值 |
| `speech_2_noise_ratio` | `speech_2_noise_ratio` | 1.0 | 判决指数 |
| `sil_pdf_id` | `sil_pdf_ids[0]` | 0 | 静音 pdf 列 |
| `cmvn_file` | — | `config/audio/am.mvn` | CMVN（fsmn_vad 官方不用，可选） |
| `lfr_m` / `lfr_n` | `lfr_m` / `lfr_n` | 5 / 1 | LFR 拼接 |
| `fbank.*` | frontend | 80/25ms/10ms | fbank 参数 |

> `vad.energy_gate`（默认关闭）：非语音期低能量块跳过 TPU 推理以省功耗；
> 语义是「标注 FIFO 行」，整组全 gated 才不发网络，前处理照常推进（帧栅格不漂移）。

### 6.2 spk / asr / summary

| 文件 | 键 | 指向 |
|---|---|---|
| `audio_link.json` → `models.spk` | `path` / `roles.speaker` | `models/campplus.bmodel`；`campplus\|speaker` |
| `audio_link.json` → `models.asr` | `path` / `roles.asr_llm,asr_audio` | `models/qwen3_asr.bmodel` |
| `audio_link.json` → `asr` | `mel_bins=128` / `mel_filters` / `vocab` / `merges` | `config/audio/mel_filters.npy` / `config/audio/vocab.json` / `config/audio/merges.txt` |
| `audio_link.json` → `spk` | `emb_dim=192` / `threshold=0.6` / `max_speakers=8` | 声纹聚类参数 |
| `meeting_summary.json` → `models.summary` | `path` / `roles.asr_llm` | `models/qwen1.5-7b_int4_6k_1dev.bmodel` |
| `meeting_summary.json` → `summary` | `vocab` / `merges` / `system_prompt` / `prompt_file` / `max_new_tokens=512` | `config/summary_*` |

### 6.3 运行期资产清单

| 资产 | 状态 | 来源 |
|---|---|---|
| `config/audio/vocab.json` / `merges.txt` | ✅ 已拷入 | LLM-TPU `models/Qwen3_ASR/config/` |
| `config/summary/summary_vocab.json` / `summary_merges.txt` | ✅ 已拷入 | 官方 `token_config.zip`（逐字节一致，见 §5.2） |
| `config/audio/mel_filters.npy` | ✅ 已入仓（128×201 float32） | whisper 式 mel filter，`asr_mel.h` 消费 |
| `config/summary/summary_prompt.txt` | ✅ 本仓 | 会议纪要模板 |
| `config/audio/am.mvn` | ✅ | fsmn_vad 官方 repo |
| `test16k.wav` | ⚠️ 自备 | 16k / mono / pcm16 |

---

## 7. 精度调整：数据集与指标

> 触发场景：模型换档（0.6B↔1.7B、int4/int8↔bf16）、重转 bmodel（改 T / 量化格式）、
> 或调 VAD 端点参数。**换档必须回归**，方法分四块。

### 7.1 ASR 回归（WER / CER）

| 项 | 建议 |
|---|---|
| **数据来源** | 公开：AISHELL-1/2/3、WenetSpeech、KeSpeech（中文）；THCHS-30；短语音用 AISHELL-1 test |
| | 自采：会议录音（覆盖 安静 / 噪声 / 多人重叠 / 方言口音），**单条 ≥30s 与 ≤3s 各若干** |
| **格式** | 16k / mono / pcm16 wav + 对应文本标签（UTF-8） |
| **指标** | **CER（中文字错率）**、WER；`jiwer` 或 `sclite` 或 `python -m jieba` 分词后字级对齐 |
| **阈值** | 相对 bf16 基线劣化 **< 2 个百分点**可接受，否则回退 bf16 |
| **口径** | 固定同一批 wav、同一 VAD 切段、同一 `max_new_tokens`，避免变量漂移 |

> 用法：先 `model_tool --info` 确认新旧 bmodel 子网/维度一致，再跑同一批 wav 出转写文本，
> 与基线做 CER。若换 asr 档位导致 SEQLEN 变化，`--max_input_length` 需同步（见 §4.3）。

### 7.2 VAD 回归（端点一致性）

| 项 | 建议 |
|---|---|
| **数据来源** | 带人工标注的语音/静音分段（如 AISHELL 句子边界；或自采音频手工标注端点） |
| **指标** | 段边界误差（ms）、段级 IoU、**帧级判决准确率** |
| **方法** | ① **离线状态机对拍**：`tools/vad_seg_ref.py`（funasr Python 参考实现）vs C++ `vad_seg_test`——同一判决序列 + 段边界 + PCM FNV-1a 哈希，要求 **bit-exact**（当前 97/97 通过）；② 换 bmodel 后跑同一音频，比段边界与 PCM 长度 |
| **阈值** | 段边界漂移 ≤ 1 帧（10ms）；判决序列与参考实现逐帧一致 |

### 7.3 声纹 / 聚类回归

| 项 | 建议 |
|---|---|
| **数据来源** | 3D-Speaker 数据集、CN-Celeb、VoxCeleb；或自采多说话人音频 |
| **指标** | 同说话人余弦相似度分布（不塌缩）；换 campplus 时加 **EER** |
| **方法** | campplus 不换则聚类质心稳定，只需抽查；换 campplus 必做 EER + 相似度直方图对比 |

### 7.4 summary 回归 + 显存预算

| 项 | 建议 |
|---|---|
| **数据来源** | 公开：官方例程自带 `python/meeting.txt`（会议文本示例）；自建：真实会议转写文本 |
| **指标** | 纪要无标准答案 → **字段级人工抽查**（【会议摘要】【行动项】完整性）；可加 BERTScore 辅助 |
| **方法** | 固定转写输入，对比新旧总结文本 |
| **显存预算** | ⚠️ **asr 是 1.7B 不是 0.6B**（§0 更正）：加载 summary（4.94GB）前必须先 `unload asr`（3.81GB）；切换窗口共存峰值 ≤ 板端可用显存 + 预留 ≥20%。**板上 `bm-smi` 实测填 `docs/model_budget.md`** |

### 7.5 INT8 量化校准集（改量化格式时必需）

FP16/BF16 有指数位、动态范围与 FP32 一致，**无需校准**；INT8 是定点
（`real = scale × (q − zero_point)`），scale 由「该层激活实际能到多大」决定，**必须喂真实样本跑一遍**：

- **vad**：N 份真实 20 帧特征 npz（`[1,20,400]` + 4×cache），覆盖安静/噪声/语音；
- **asr/summary**：LLM-TPU 的 `llm_convert.py` 内置量化流程，用真实 wav/文本样本；
- 校准集规模经验：**100~500 条**（tpu-mlir 默认 `--cali` 流程）；
- 校准后**必须做 §7.1 / §7.4 的精度回归**，劣化超阈回退 bf16/F16。

---

## 8. 复现速查（命令清单）

```bash
# ---------- 环境 ----------
docker run --privileged --name tpuc -v $PWD:/workspace -it sophgo/tpuc_dev:latest
export TPUC_ROOT=/workspace/tpu-mlir/install
export PATH=$TPUC_ROOT/bin:/workspace/tpu-mlir/python/tools:$PATH
export PYTHONPATH=$TPUC_ROOT/python:/workspace/tpu-mlir/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TPUC_ROOT/lib:/workspace/tpu-mlir/capi/lib:$LD_LIBRARY_PATH

# ---------- vad（自行转换）----------
python3 modelSource/dequant_onnx.py \
    modelSource/speech_fsmn_vad_zh-cn-16k-common-onnx/model_quant.onnx \
    modelSource/fsmn_vad_float.onnx
bash modelSource/fsmn_T20/convert_T20.sh          # T=20 静态 F16
cp modelSource/fsmn_T20/fsmn_vad_T20_bm1684x_f16.bmodel models/fsmn_vad.bmodel

# ---------- spk（官方现成，或不转）----------
model_tool --info models/campplus.bmodel          # 验 dynamic [max:1,max:600,max:80]→[1,192]

# ---------- asr（官方现成 bmodel）----------
cd /d/sophgo && cp dfss_fetch.py /d/sophgo/ 2>/dev/null
docker exec soc-cross bash -lc 'cd /workspace && python3 dfss_fetch.py get \
  /ext_model_information/LLM/LLM-TPU/qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_20260408_094600.bmodel \
  /workspace/MixedAudioLink/models/qwen3_asr.bmodel'

# ---------- summary（官方现成 bmodel + tokenizer）----------
docker exec soc-cross bash -lc 'cd /workspace && \
  python3 dfss_fetch.py get /sophon-demo/Qwen1_5/qwen1.5-7b_int4_6k_1dev.bmodel \
    /workspace/MixedAudioLink/models/qwen1.5-7b_int4_6k_1dev.bmodel && \
  python3 dfss_fetch.py get /sophon-demo/Qwen1_5/token_config.zip /workspace/token_config.zip'

# ---------- 全部验收 ----------
for f in models/*.bmodel; do echo "=== $f ==="; \
  model_tool --info "$f" 2>/dev/null | grep -E "^net [0-9]+:|^bmodel version|^chip"; done
```

> **下载环境注意**：本机 Python38 / managed 3.13 **均无 paramiko**（`pip install` 被网络挡住）。
> 用 `soc-cross` 容器（paramiko 3.3.1 / py3.8.10）或 `tpu-mlir` 容器（paramiko 3.2.0 / py3.10.12）内跑
> `dfss_fetch.py`（soc-cross 挂载 `D:\sophgo → /workspace`）。`dfss` 官方客户端会卡在测速，勿用。

---

## 附：常见坑速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Op not support: {MatMulInteger, DynamicQuantizeLinear}` | 官方 ONNX 是 ORT 动态量化 | 先跑 `dequant_onnx.py` 反量化（§2.2） |
| `input ... is not output of any previous nodes` | 新 initializer 未塞进 graph | 手术脚本手动 append（§2.2） |
| `no net matched role` | 子网重命名 / 角色关键字未命中 | 保留子网名，或改 config `roles` |
| transform 阶段 `args.test_result` 断言失败 | `--test_input` 放错阶段 | 挪到 `model_deploy.py`（§2.3） |
| `import pymlir` 缺 `libortools.so.9` | 未设 LD_LIBRARY_PATH | 按 §1 导出环境变量 |
| 动态 bmodel 流式 FSMN cache 漂移 | 运行时按满维跑污染 out_cache | 改用静态 T（§2.3「为什么静态」） |
| 显存不足 / 加载 summary 失败 | asr 是 1.7B（3.81GB），非 0.6B | 先 `unload asr`；板端实测预算（§7.4） |
