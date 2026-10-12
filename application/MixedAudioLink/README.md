# MixedAudioLink 例程

## 目录

- [MixedAudioLink 例程](#mixedaudiolink-例程)
  - [简介](#简介)
  - [特性](#特性)
  - [1. 工程目录](#1-工程目录)
  - [2. 准备模型与数据](#2-准备模型与数据)
  - [3. 编译](#3-编译)
  - [4. 运行](#4-运行)
  - [5. 文档](#5-文档)

## 简介

MixedAudioLink 例程是面向算能 **BM1684X** 的会议音频理解框架：外部持续推入
16k / mono / int16 PCM，框架内部串联完成

**流式 VAD 切段（fsmn_vad）→ 整句语音识别（Qwen3-ASR）＋ 声纹提取（CAM++）→ 会议纪要生成（Qwen1.5-7B）**，

结果经 C 回调交付。架构形态仿 rk_libs Link 框架（字符串名字寻址 + 拉模型 + 统一状态码），
内部 C++ 实现、对外导出**纯 C ABI**（`audio_link.h` / `meeting_summary.h`），C 程序可直接链接。

流水线：

```
PCM(16k int16) ─▶ VAD 切段 ─┬─▶ ASR 整句识别 ─┐
                            └─▶ CAM++ 声纹     ┴─▶ "讲话人N：xxx" ─▶ 会议总结
```

## 特性

* 支持 BM1684X（PCIE / SoC）——当前主链；BM1688 需按同配方重转 bmodel
* **全部 bmodel 静态 shape**，运行期从各子网 I/O 自动推导维度（JSON `dims` 保持 `auto`）
* **流式 VAD**：fsmn_vad 静态 T=20 + 4 组 cache，FunASR 官方端点状态机，切段与官方 Python 参考 **bit-exact**
* 多模型各自独立 bmrt 上下文，支持 `unload_model` / `load_model` 按需切换（显存受限场景）
* 纯 C ABI 静态库，零第三方 C++ 依赖（仅 bmlib / bmruntime）

## 1. 工程目录

```bash
MixedAudioLink/
├── cpp/                        # 框架源码（内部 C++ 实现）
│   ├── audio_link.h/.cpp       # 音频流模块 C API（对外入口）
│   ├── meeting_summary.h/.cpp  # 会议总结模块 C API
│   ├── link_mng.h/.cpp         # Link 框架通用基座（对象池/段任务队列/线程/通用 Ctrl）
│   ├── audio_link_mng.h/.cpp   # 音频工作流子类（默认链路/双线程/音频 Ctrl）
│   ├── model_hub.h/.cpp        # 多 bmodel 管理 + 维度推导 + 卸载/重载
│   ├── text_llm.h              # text-only LLM 推理公共类（asr / summary 共用）
│   ├── asr_mel.h / asr_tokenizer.h
│   ├── al_feature.h            # fbank80 / CMVN / npy（vad / spk 支路共用）
│   ├── al_log.h / al_json.h / al_resample.h / al_path.h
│   ├── links/                  # pcm / vad / feat / asr / spk / diarize / sink
│   └── main.cpp                # C++ demo
├── examples/sample_c.c         # 纯 C 调用示例
├── config/audio/               # audio_link.json + vocab/merges/mel_filters.npy/am.mvn
├── config/summary/             # meeting_summary.json + vocab/merges/prompt
├── models/                     # bmodel 落位目录（大文件不入仓，见 .gitkeep）
├── scripts/                    # build_soc.sh / download_model.sh / convert_vad.sh
├── tools/                      # 对拍与回归脚本 + 单测
├── docs/                       # 使用者文档（见 §5）
└── CMakeLists.txt
```

## 2. 准备模型与数据

### 2.1 拉取官方模型与分词器

```bash
bash scripts/download_model.sh          # = models + assets + datasets
#   models    qwen3_asr / campplus / qwen1.5-7b_int4_6k
#   assets    Qwen 分词器 → config/audio 与 config/summary
#   datasets  一段 16k/mono wav → test16k.wav（仓库不收录音频）
```

下载器优先用官方 `dfss`；若其内置测速失效（报 `No available server found`），
脚本会自动改用本仓 `tools/dfss_fetch.py` 直连官方 SFTP 兜底（需 `paramiko`）。

### 2.2 自转 VAD bmodel（静态 T=20）

`models/fsmn_vad.bmodel` **不在下载脚本内**，它由官方 ONNX 自行转换：

```bash
# 1) 取官方 ONNX（ModelScope iic/speech_fsmn_vad_zh-cn-16k-common-onnx）
#      model_quant.onnx  —— ORT 动态量化，tpu-mlir 不支持，需先反量化
#      am.mvn            —— Kaldi CMVN → config/audio/am.mvn
# 2) 反量化
python3 tools/dequant_onnx.py model_quant.onnx fsmn_vad_float.onnx
# 3) 转 bmodel（在 tpu-mlir 镜像容器内执行）
ONNX=/path/fsmn_vad_float.onnx bash scripts/convert_vad.sh
```

> ⚠️ VAD bmodel 冻结为静态 **T=20**（= `vad.chunk_ms` / 10）。改 `vad.chunk_ms` 必须重转 bmodel，
> 框架启动时会校验并告警。逐条命令与精度数据集见 [`docs/model_provenance.md`](docs/model_provenance.md)。

### 2.3 模型清单

| 角色 | 文件 | 档位 / 大小 | 获取 |
|---|---|---|---|
| vad | `models/fsmn_vad.bmodel` | 静态 T=20 / F16 / 6.5MB | 自转（§2.2） |
| asr | `models/qwen3_asr.bmodel` | Qwen3-ASR-1.7B / bf16 seq512 / 3.81GB | 官方现成下载 |
| spk | `models/campplus.bmodel` | CAM++ / F32 dynamic / 32MB | 官方现成下载 |
| summary | `models/qwen1.5-7b_int4_6k_1dev.bmodel` | Qwen1.5-7B / int4 6k / 4.94GB | 官方现成下载 |

## 3. 编译

需要 sophon-sdk（**≥ 23.x**，含 `bmlib` / `bmruntime`）。SoC 交叉编译：

```bash
export SDK=/path/to/sophon-sdk
SDK=$SDK ./scripts/build_soc.sh      # 产出 build/libaudiolink.a、build/audiolink_demo、build/sample_c
```

或手工：

```bash
export SDK=/path/to/sophon-sdk
export CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
cmake -B build -DSDK="$SDK" && cmake --build build -j
```

编译本身**不需要任何 bmodel**。

## 4. 运行

```bash
# 从应用根启动（模型的相对路径以 CWD 为基准；配置资产以 JSON 所在目录为基准）
./build/audiolink_demo -c config/audio/audio_link.json -d 0 -i test16k.wav
```

纯 C 调用（`examples/sample_c.c`，`-std=c99` 即可）：

```bash
gcc -std=c99 -O2 -I cpp -I "$SDK/include" \
    examples/sample_c.c -L build -L "$SDK/lib" \
    -laudiolink -lbmlib -lbmruntime -lpthread -o sample_c
./sample_c -c config/audio/audio_link.json -s config/summary/meeting_summary.json \
           -d 0 -i test16k.wav
```

> **显存**：asr（3.81GB）与 summary（4.94GB）不能共存。会议总结前须先
> `Ctrl "asr" "unload_model"` 释放显存，总结完再 `load_model` 恢复。

### 路径基准（双基准）

| 类别 | 键 | 解析基准 |
|---|---|---|
| 模型文件 | `models.*.path` | **应用根（启动 CWD）** |
| 配置资产 | `vocab` / `merges` / `mel_filters` / `cmvn_file` / `prompt_file` | **启动 JSON 所在目录** |

## 5. 文档

| 文档 | 内容 |
|---|---|
| [`docs/README.md`](docs/README.md) | 框架总览：架构、数据流、模型组织、静态 shape 适配、目录树 |
| [`docs/API.md`](docs/API.md) | C API 全量参考（函数 / 控制命令 / 回调 / 参数结构） |
| [`docs/dataflow.md`](docs/dataflow.md) | 各 Link 的数据流与子网 I/O 协议 |
| [`docs/status.md`](docs/status.md) | 功能状态与已知限制 |
| [`docs/model_provenance.md`](docs/model_provenance.md) | **模型溯源手册**：来源 / 逐条转换命令 / 参数 / 精度回归数据集 |
| [`docs/model_conversion.md`](docs/model_conversion.md) | 模型转换流程与产物布局 |
| [`docs/model_budget.md`](docs/model_budget.md) | 显存预算与降档路线 |
