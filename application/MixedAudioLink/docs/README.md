# MixedAudioLink — 混合模型音频推理框架（C API + Link 链路）

面向算能 BM1684X/BM1688 的音频流 AI 推理框架：外部推入 16k/mono/int16 PCM，
框架完成 **VAD 切段（fsmn_vad）→ qwen3_asr 整句识别 + CAM++ 声纹提取**，
结果经 C 回调交付。架构形态仿 rk_libs Link 框架（字符串名字寻址 + 拉模型 +
统一状态码），内部 C++ 实现、对外导出纯 C ABI（`audio_link.h`）。

## 架构与数据流

```
AudioLinkPushPcm(16k int16)
   │  非阻塞，环形缓冲（默认 5s），满丢最旧并计数
   ▼
CPcmLink ──拉模型──▶ CVadLink(fsmn_vad 子网, 200ms 块 + cache, 切段状态机)
   │                    │ VAD 线程（独占 fsmn_vad bmrt 上下文）
   │                    ▼ 段任务队列
   │              CFeatLink ──mel128──▶ CAsrLink(qwen3_asr)  ┐
   │                    │         └─fbank80─▶ CSpkLink(CAM++) ┤ 任务线程（串行）
   │                    ▼                                    ▼
   │              CDiarizeLink（在线说话人聚类 + 转写缓冲 "讲话人N：xxx"）
   │                                                    ▼
   └─────────────────────────────────────▶ CSinkLink → 用户回调
```

- **双线程模型**：VAD 线程持续跑保证不丢音频；ASR 长推理期间切段照常进行。
- **拉模型**（上半段）：`CPcmLink::getFrame` 被下游持有，帧走对象池"借用-归还"，
  池满返回 `E_AL_TIMEOUT`（背压语义）。
- **段任务驱动**（下半段）：段为批次粒度，任务线程串行 ASR→SPK→Diarize。
- **框架通用化分层**：`CLinkMng`（link_mng.h）为通用管理器基座——配置解析/
  对象池/段任务队列/线程生命周期/通用 Ctrl（get_stats/save_params/set_log 等）；
  音频工作流专属逻辑下沉至派生类 `CAudioLinkMng`（audio_link_mng.h）：默认链路
  创建、Link 工厂分支、VAD+任务双线程、`pushPcm` 数据口、音频 Ctrl
  （flush/new_meeting/unload_model/load_model）、音频配置键校验、退出快照音频段。
  两层经 protected 虚钩子（validateAudio/createLinks/startWorkers/stopWorkers/
  onCtrl/appendStats/appendStatusMd/createLinkImpl）回接，新增工作流类型时
  派生基座并落地钩子即可复用全部框架设施。
- **模型加载进度日志**：Init 全程经 al_log 输出（默认 INFO 可见、可重定向文件）——
  每模型 `load begin（文件大小 MB）→ loaded（耗时 ms + 子网数）→ ready（角色映射）`、
  逐 Link created、`init done（总耗时 + links/models 数）`；MeetingSummary 各步骤
  同构（hub/llm 含 io_alone 与层数/tokenizer）。首编冒烟时此日志流即加载排障主线索。

## 模型组织（多文件分离，无融合模型）

| 模型名（JSON models 键） | 内容 | 角色关键字映射 |
|---|---|---|
| `asr` | qwen3_asr（embedding / block_i / block_cache_i / audio / lm_head / greedy_head） | `asr_llm` / `asr_audio` |
| `spk` | CAM++ 单网 | `speaker` |
| `vad` | fsmn_vad（speech_fsmn_vad_zh-cn-16k-common） | `vad` |

> ⚠️ **所有模型均为独立 bmodel 文件**：每个 JSON models 键对应一个独立
> bmrt 上下文（独立加载/卸载），**不支持融合模型**（单文件多子网混合
> bmodel）。同一 bmodel 内的多子网（如 asr 的 LLM 分解子网）是转换产物
> 自身的子网划分；跨模型之间无任何共享。显存占用 = Σ(常驻模型)。

每个 bmodel 一个 bmrt 上下文（`CModelHub`）。子网**动态枚举**
（`bmrt_get_network_names`），JSON `roles` 按名字关键字匹配角色；
**未匹配子网仅登记暴露**（`Ctrl "list_nets"` 可查，`Ctrl "asr.block_0" "disable"`
可禁用），未来新增模型/子网只改 JSON 即接入。

## 静态 shape 适配

全部 bmodel 为静态 shape。框架内置**维度推导器**：加载后从各子网
`stages[0]` 的 input/output shapes 自动推导 SEQLEN / MAX_INPUT_LENGTH /
HIDDEN_SIZE / NUM_LAYERS / AUDIO_LENGTH / KV_BYTES / mask_value 等
（对齐 LLM-TPU chat.cpp `init_by_names`），JSON `dims` 节保持 `"auto"`；
手工数值仅作推导失败回退，且与推导值不符时告警。所有输入
（input_ids / position_ids / mask / fbank）按静态 shape **补齐 padding**
后整块拷入，`Ctrl "asr" "get_model_info"` 可查询推导结果。

## 快速开始

```bash
# 板端（SoC 模式，BM1688/BM1684X；SDK 需 >= 23.x——io_alone 检测依赖 addr_mode 字段）
export SDK=/path/to/sophon-sdk
export CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
cd application/MixedAudioLink
cmake -B build -DSDK="$SDK" && cmake --build build
./build/audiolink_demo -c config/audio/audio_link.json -d 0 -i test16k.wav
```

### 运行期资产就绪清单

| 资产 | 引用位置 | 状态 | 来源 |
|---|---|---|---|
| `config/audio/vocab.json` / `merges.txt` | asr 节 | ✅ 已拷入 | LLM-TPU `models/Qwen3_ASR/config/` |
| `config/summary/summary_vocab.json` / `summary_merges.txt` | summary 节 | ✅ 已拷入（同源重命名） | 同上 |
| `config/summary/summary_prompt.txt` | summary 节 prompt_file | ✅ 已入仓 | 本仓库 |
| `config/audio/mel_filters.npy` | asr 节（128×201 float32 npy） | ✅ 已入仓 | whisper 式 mel filter（`asr_mel.h` 消费） |
| `config/audio/am.mvn` | vad 节 `cmvn_file` | ✅ 已入仓 | funasr `speech_fsmn_vad_zh-cn-16k-common` Kaldi CMVN（8040 B） |
| `test16k.wav` | demo 输入 | ⚠️ 自备（`scripts/download_model.sh datasets` 可取） | 16k/mono/pcm16 |

### bmodel 就绪清单（芯片无关，代码不区分芯片；现成度按芯片不同）

| 模型 | BM1684X 现成度 | BM1688 现成度 | 获取方式 |
|---|---|---|---|
| spk（CAM++） | ✅ **现成可下载** | 🟡 需重转 | `sophon-demo/sample/CAM++/scripts/download.sh`（BM1684X，官方配方即 `--dynamic F32`） |
| asr（qwen3_asr） | ✅ **现成可下载** | 🟡 需重转 | 官方 bmodel `qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_*.bmodel`（或 LLM-TPU `models/Qwen3_ASR/README.md` 自转） |
| vad（fsmn_vad） | 🟡 自行转换 | 🟡 自行转换 | funasr `speech_fsmn_vad_zh-cn-16k-common`：ORT 量化 onnx → **反量化** → tpu-mlir → bmodel（见 `model_provenance.md` §2） |
| summary（text-only LLM） | ✅ **现成可下载** | 🟡 需重转 | 官方 `sophon-demo/application/meeting_summary`：`qwen1.5-7b_int4_6k_1dev.bmodel` |

> 逐条转换命令、参数与精度回归数据集：见 [`docs/model_provenance.md`](model_provenance.md)。

转换指引要点：
- asr/summary 均为 **LLM-TPU chat 格式**（embedding/block_i/block_cache_i/lm_head[/greedy_head]），
  asr 另含 audio 编码子网——子网命名与 I/O 约定见 `docs/dataflow.md`；
- vad 为**单网静态 shape**（[1,T,400] LFR 特征输入 + 4 路 cache 输入 + [1,T,248] 概率输出），
  **T 必须与 `vad.chunk_ms` 匹配（200ms → T=20 帧）**；源 `model_quant.onnx` 是 ORT 动态量化，
  须先反量化为 float ONNX 再转（见 `docs/model_conversion.md` §1）；
- **冒烟建议顺序**：spk（现成）→ vad（转换后）→ asr → summary（转换耗时最长）；
  编译验证本身不需要任何 bmodel。

依赖：bmlib / bmruntime（sophon-sdk **≥ 23.x**——io_alone 检测依赖 addr_mode 字段与
bm_mem_get_device_addr）；零第三方 C++ 依赖。

## 工程目录

```
application/MixedAudioLink/
├── cpp/
│   ├── audio_link.h/.cpp       # C API（音频流模块，对外唯一入口）
│   ├── meeting_summary.h/.cpp    # MeetingSummary 会议总结模块（独立 C API）
│   ├── link_mng.h/.cpp         # Link 框架通用基座：CBaseLink/CLinkMng/对象池/虚钩子
│   ├── audio_link_mng.h/.cpp   # 音频工作流子类 CAudioLinkMng（默认链路/双线程/音频 Ctrl）
│   ├── model_hub.h/.cpp        # 多 bmodel 管理 + 维度推导 + 卸载/重载
│   ├── text_llm.h              # text-only LLM 公共推理类（asr/summary 共用）
│   ├── asr_mel.h               # 128 维 mel（whisper 式 + Bluestein FFT）
│   ├── asr_tokenizer.h         # Qwen BPE（vocab.json + merges.txt）
│   ├── al_feature.h            # fbank80/CMVN/npy（vad/spk 支路共用）
│   ├── al_log.h                # 全局统一日志（级别/文件重定向/轮转，header-only）
│   ├── al_json.h               # 极简 JSON 解析 + Set/Dump 序列化（save_params 用）
│   ├── al_resample.h           # 多相 FIR 重采样（48k/44.1k/8k→16k + 混缩，header-only）
│   ├── links/                  # pcm / vad / feat / asr / spk / diarize / sink
│   └── main.cpp                # C++ demo
├── examples/sample_c.c         # ★ 纯 C 程序调用示例（见下节）
├── config/audio/               # 音频流：audio_link.json + vocab/merges/mel_filters.npy/am.mvn
├── config/summary/             # 总结：meeting_summary.json + vocab/merges/prompt
├── models/                     # bmodel 落位（大文件不入仓，.gitkeep + scripts 获取）
├── docs/                       # 使用者文档：README / API / dataflow / status / model_*
├── scripts/                    # build_soc.sh / download_model.sh / convert_vad.sh / pack_board_test.sh
├── tools/                      # 对拍与回归 + 单测（随仓）
└── CMakeLists.txt
```

### 路径基准（双基准）

运行期路径分两类，容易踩坑，务必区分：

| 类别 | 键 | 解析基准 |
|---|---|---|
| **模型文件** | `models.*.path` | **应用根 = 启动时 CWD**（`resolveModelPath`，只做分隔符归一） |
| **配置资产** | `vocab` / `merges` / `mel_filters` / `cmvn_file` / `prompt_file` | **启动 JSON 所在目录**（`resolveAsset`，绝对路径原样） |

即：`models/qwen3_asr.bmodel` 随**应用部署位置**解析；`vocab.json` 随**配置文件位置**解析。
所以从**应用根**直接 `./build/audiolink_demo -c config/audio/audio_link.json` 可寻址全部资产——
`config/audio/` 下的相对引用以该目录为基准。若把 JSON 挪到别处，只需同步挪走同目录资产。

## C 程序如何调用编译好的库

库产物为 **静态库 `libaudiolink.a`**（含音频流模块 + MeetingSummary 模块），
头文件 `cpp/audio_link.h` 与 `cpp/meeting_summary.h` 均为纯 C ABI
（`extern "C"` + 无 C++ 类型），C 程序直接 include + 链接即可。

最小调用序列（完整可编译示例见 `examples/sample_c.c`）：

```c
#include "audio_link.h"
#include "meeting_summary.h"

/* 1. 结果回调（每识别完一段触发，line = "讲话人N：xxx"） */
static void on_result(const AL_RESULT *r, void *user) {
    printf("%s\n", r->line ? r->line : r->text);
}

int main(void) {
    /* 2. 初始化：加载 JSON 声明的全部模型 */
    AL_HANDLE mgr = AudioLinkInit("config/audio/audio_link.json", 0);

    /* 3. 注册回调 */
    AL_RESULT_CB cb = on_result;
    AL_PARAM p = { &cb, sizeof(cb) };
    AudioLinkCtrl(mgr, "sink", "set_result_cb", p);

    /* 4. 启动链路，然后从采集线程循环推 PCM（16k/mono/int16，非阻塞） */
    AudioLinkStart(mgr);
    AudioLinkPushPcm(mgr, chunk, chunk_bytes);

    /* 5. 结束：冲刷尾段 → 取转写 → 会议总结（卸 asr 释放显存） → 清理 */
    AudioLinkCtrl(mgr, NULL, "flush", (AL_PARAM){0});
    AudioLinkCtrl(mgr, "asr", "unload_model", (AL_PARAM){0});
    AL_HANDLE sm = MeetingSummaryInit("config/summary/meeting_summary.json", 0);
    MeetingSummaryRun(sm, transcript, (AL_PARAM){out, sizeof(out)});
    MeetingSummaryDeinit(sm);
    AudioLinkCtrl(mgr, "asr", "load_model", (AL_PARAM){0});
    AudioLinkCtrl(mgr, NULL, "new_meeting", (AL_PARAM){0});
    AudioLinkStop(mgr);
    AudioLinkDeinit(mgr);
    return 0;
}
```

编译（`-std=c99` 即可，无 C++ 运行时依赖）：

```bash
gcc -std=c99 -O2 \
    -I cpp -I $SDK/include \
    examples/sample_c.c \
    -L build -L $SDK/lib \
    -laudiolink -lbmlib -lbmruntime -lpthread \
    -o sample_c
./sample_c -c config/audio/audio_link.json -s config/summary/meeting_summary.json \
           -d 0 -i test16k.wav
```

要点：
- 两个头文件可同时 include（符号经 `extern "C"` 导出，无名字冲突）；
- 所有句柄（`AL_HANDLE`）以 `NULL`/`0` 判无效；错误经 `get_last_error` 查询
  （`AudioLinkInit` 失败后句柄为 NULL，也可查）；
- `AudioLinkPushPcm` 可在任意线程调用（内部加锁）；回调在框架线程触发，
  勿在回调内长时间阻塞或反向调用音频流接口；
- **非 16k 输入用 `AudioLinkPushPcmEx`**（显式传 sample_rate/channels/bits，
  bits 仅支持 16；首次推入锁存格式，框架内部重采样到 16k/mono）；
- 模型卸载/重载期间（unload_model ~ load_model 之间）不要推入新 PCM。

## 关于 FunASR 框架

**FunASR（PyTorch 工具包）不能直接运行在算能 TPU 上**，正确路线是模型转换：
`torch → onnx → tpu-mlir → bmodel`。本框架只消费已转换的 bmodel：
- fsmn_vad：funasr `speech_fsmn_vad_zh-cn-16k-common` 的 bmodel 化版本，
  流式协议与 funasr Python 版一致（200ms 块 + cache + `[beg,-1]` 事件语义）；
- qwen3_asr：LLM-TPU `models/Qwen3_ASR` 的 C++ 实现（chat.cpp）移植；
- CAM++：sophon-demo `sample/CAM++` 同源。

## 说话人归属与会议转写

`CDiarizeLink` 对每段结果做**在线自动聚类**（无预注册）：
- 首段 → 讲话人1；后续段与已有质心算余弦相似度，≥ `spk.threshold`(0.6)
  归入该说话人并滑动更新质心，否则新建 讲话人2/3/…（上限 `spk.max_speakers`=8）；
- 回调 `AL_RESULT` 带 `speaker_id` 与格式化行 `line`（"讲话人N：xxx"）；
- 转写缓冲经 `Ctrl("diarize","get_transcript")` 一次性取全量文本；
  `Ctrl("new_meeting")` 清空开新会。

## 动态配置（运行期热调参数）

VAD / ASR / 说话人聚类 / PCM 参数支持运行期经各 Link **专用 Ctrl 命令**热调
（数值统一 double 协议 `AL_PARAM{&dVal, sizeof(dVal)}`，立即生效），
`save_params` 把修改原子写回启动 JSON（13 个命令清单详见 `docs/API.md` 动态配置节）：

```c
double dVal = 0.3;
AL_PARAM p = { &dVal, sizeof(dVal) };
AudioLinkCtrl(mgr, "vad", "set_noise_thres", p);   /* 立即生效（后续块用新阈值） */
AudioLinkCtrl(mgr, NULL, "save_params", (AL_PARAM){0});   /* 原子写回启动 JSON */
```

## 会议总结（MeetingSummary 模块，非 Link 节点）

会议总结是对**文字集合的非流式处理**，设计为独立模块而非链上节点
（`meeting_summary.h`）：内部为 text-only LLM（LLM-TPU chat 格式 bmodel，
Qwen3 系列通用，无 audio 子网），`MeetingSummaryRun(转写文本)` 生成总结。
长转写自动 **map-reduce**（分块摘要→合并终总结，机制参考本仓库
`application/meeting_summary` 官方例程的 RecursiveCharacterTextSplitter +
分段摘要 + 合并流程，C++ 侧手写实现、无 langchain 依赖）。

### 两种驻留时序

**全常驻（默认推荐，64TOPS 级 / 显存 ≥8GB 设备）**：所有模型同时驻留，
启动即 `MeetingSummaryInit`，总结零加载延迟，句柄可复用：

```c
AL_HANDLE mgr = AudioLinkInit(...);
AL_HANDLE sm  = MeetingSummaryInit("config/summary/meeting_summary.json", dev);  /* 同时驻留 */
/* ... 音频流运行，随时 MeetingSummaryRun(sm, transcript, out) ... */
MeetingSummaryDeinit(sm);   /* 退出统一释放 */
AudioLinkDeinit(mgr);
```

**换入换出（显存受限设备备选，sample 加 `--swap` 演示）**：总结前卸 asr 释放
显存，总结后重载：

```c
AudioLinkCtrl(mgr, "asr", "unload_model", {0});   /* 释放 qwen3_asr 显存 */
AL_HANDLE sm = MeetingSummaryInit("config/summary/meeting_summary.json", dev);
MeetingSummaryRun(sm, transcript, out);              /* transcript ← get_transcript */
MeetingSummaryDeinit(sm);                            /* 释放总结模型显存 */
AudioLinkCtrl(mgr, "asr", "load_model", {0});      /* 恢复音频流识别 */
AudioLinkCtrl(mgr, nullptr, "new_meeting", {0});   /* 开新会 */
```

卸载期间：VAD/声纹/聚类照常运行（宿主内存状态不受影响），段无文本
不追加转写行。

### 独立总结模式（MeetingSummaryRunFile，对齐官方 meeting_summary 例程）

`meeting_summary.h` 新增 `MeetingSummaryRunFile(h, text_path, out)`：**直接从文本文件
读入转写内容生成总结**，不依赖音频流模块（无需 `AudioLinkInit`/`Start`/`PushPcm`，
不加载 asr/vad/spk 模型，显存只需总结 LLM 一份）。对齐官方
`application/meeting_summary` 例程的三件套用法：

| 官方 meeting_summary 三件套 | 本模块对应 |
|---|---|
| `prompt.txt`（字段化纪要模板） | `config/summary/summary_prompt.txt`（含 `{transcript}` 占位，经 `meeting_summary.json` 的 `prompt_file` 指向） |
| `meeting.txt`（转写输入文件） | 用户自备转写文本（"讲话人N：xxx\n…" 或纯文本，UTF-8） |
| `output.txt`（总结输出文件） | `summary_output.txt`（sample_c 自动落盘） |

用法（`examples/sample_c.c` 已内置 `-t` 参数支持）：

```bash
./sample_c -s config/summary/meeting_summary.json -t meeting.txt
# -d 0 可省略（默认 dev 0）；终端打印总结，同时落盘 summary_output.txt
```

典型场景：转写已由其他系统产出，只需在板端跑总结 LLM；或调试
`summary_prompt.txt` 模板效果（改文件即可，无需重编/改配置）。

## 退出状态快照

商用运维场景：配置快照路径后，`Stop` 退出前（链路仍存活、统计最完整时）
自动写入 Markdown 运行状态（版本/运行时长/段数/丢弃/转写统计）：

```c
AudioLinkCtrl(mgr, NULL, "set_status_file", (AL_PARAM){"runtime_status.md", 18});
```

示例见 `examples/sample_c.c`（运行后在 CWD 生成 `runtime_status.md`）。
进程级收尾仍由上层负责（框架内不做 atexit）。

## 参考

- `docs/API.md` — C API 逐函数说明
- `docs/dataflow.md` — 维度推导、前处理细节与已知的数值对齐注意事项
- **`docs/model_provenance.md` — 模型溯源与转换手册（来源 / 逐条转换命令 / 参数 / 精度回归数据集）**
- `docs/model_conversion.md` — 框架对模型的 I/O 需求清单（转换视角）
- `docs/model_budget.md` — 模型预算表、显存峰值模型与选型/精度回归指引
- `docs/code_review_report.md` — 五轴代码评审报告
- `docs/meeting_summary_reference_report.md` — 官方 meeting_summary 例程对照研究
- `docs/review_report.md` — 双 agent 评审报告（更名与三件套外置轮次）
- `docs/status.md` — **商用库当前状态**（功能完成度/质量/模型/交付物/下一步）
- `docs/commercial_roadmap.md` — 商业化优化路线图
