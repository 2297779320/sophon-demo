# C API 参考（audio_link.h）

统一返回 `E_AL_STATE`（0 = 成功；`E_AL_TIMEOUT` 为背压语义非错误）。
所有名字寻址的接口，名字为 Link 名（pcm/vad/feat/asr/spk/sink 或自定义）。

## 版本

### AudioLinkGetVersion
```c
const char *AudioLinkGetVersion(void);
```
返回静态版本字符串 `"MixedAudioLink <major>.<minor>.<patch> (<git_hash> <build_date>)"`
（进程生命周期内有效，勿释放）。版本号见 audio_link.h 顶部宏
`AL_VERSION_MAJOR/MINOR/PATCH`；git_hash/build_date 可经构建注入
（CMake `-DAL_GIT_HASH=... -DAL_BUILD_DATE=...`，默认 "unknown"）。

## 生命周期

### AudioLinkInit
```c
AL_HANDLE AudioLinkInit(const char *json_cfg, int dev_id);
```
读配置、请求设备、加载全部模型、创建默认七条 Link（pcm/vad/feat/asr/spk/diarize/sink）
并绑定 `pcm→vad`。
失败返回 NULL；错误经 `AudioLinkCtrl(NULL, "get_last_error", ...)` 查询
（Init 失败句柄已释放，NULL 句柄下仅 `get_last_error` 可查，内部取静态最近错误）。

**启动校验**（Init 内两段执行）：
- 第一段（模型加载前）：必需键存在（`models.*.path`、`asr.mel_filters/vocab/merges`）、
  数值范围（`pcm.ring_ms≥1`、`vad.chunk_ms∈[50,1000]`、`vad.speech_noise_thres∈[0,1]`、
  `vad.max_seg_s≥1`、`spk.threshold∈(0,1)`、`asr.max_new_tokens>0`、`log.level∈[-1,3]`）、
  文件存在（上述路径键 + `vad.cmvn_mean/std` 若填写）；
- 第二段（模型加载后）：各模型 `roles` 逐角色命中子网数 > 0。
顶层 `"strict"` 开关（默认 `false`）：`true` 时校验失败拒绝启动（返回 NULL，
`get_last_error` 给出含完整 JSON 路径的首个错误，如
`vad.speech_noise_thres: expect [0.000000,1.000000], got 3.500000`）；
`false` 时仅 WARN 告警，沿用默认/钳制值继续启动（越界值仍被既有钳制点兜底并打 WARN）。

### AudioLinkCreate
```c
int AudioLinkCreate(AL_HANDLE mgr, const char *name, const char *type, AL_PARAM param);
```
创建额外 Link（默认七条已自动创建）。`type` 取
`AL_LINK_PCM/VAD/FEAT/ASR/SPK/DIARIZE/SINK`。
`sink` 的 `param` 可传 `{ AL_RESULT_CB cb; void* user; }` 结构注册结果回调。

### AudioLinkBind / Start / Stop / Deinit
- `Bind(src, dst)`：拉模型连线（Start 前调用）。
- `Start`：启动 VAD 线程 + 任务线程。
- `Stop`：停止；未冲刷的语音段丢弃（冲刷先 Ctrl `"flush"`）。
- `Deinit`：释放全部资源，句柄作废。

## 数据

### AudioLinkPushPcm
```c
int AudioLinkPushPcm(AL_HANDLE mgr, const void *pcm, int bytes);
```
推入 PCM，**非阻塞**；环形缓冲（`pcm.ring_ms`，默认 5000ms）满时丢最旧并累计
告警，`Ctrl "get_stats"` 可查丢弃字节数。输入格式按 JSON `pcm.input_fmt`
（`{sample_rate, channels, bits}`，默认 16k/1/16，此时行为与历史版本逐字节一致）
解释；非 16k 输入请用 `AudioLinkPushPcmEx`。

### AudioLinkPushPcmEx
```c
int AudioLinkPushPcmEx(AL_HANDLE mgr, const void *pcm, int bytes,
                       int sample_rate, int channels, int bits);
```
显式格式推入（非阻塞，环形缓冲/丢最旧语义同上）。`bits` 仅支持 16；
`sample_rate/channels` 任意——首次成功推入即**锁存**输入格式，此后不一致返回
`E_AL_PARAM_ERR`（防误配混流）；非 16k/mono 输入内部经多相 FIR 重采样到
16k/mono（多声道均值混缩）后进链。`AudioLinkPushPcm` 等价于按
`pcm.input_fmt` 调用本接口。

### AudioLinkCtrl（万能控制口）

| name | strCmd | param | 说明 |
|---|---|---|---|
| `NULL` | `get_last_error` | `char[256]` 出参 | 最近错误描述（Init 失败返回 NULL 后仍可查） |
| `NULL` | `get_stats` | `char[1024]` 出参 | 统计 JSON（PCM 丢弃计数、`task_busy` 任务线程忙闲、`self_heal_count` 自愈成功次数等） |
| `NULL` | `list_nets` | `char[4096]` 出参 | 全部模型/子网/角色/启停状态 |
| `NULL` | `flush` | — | 冲刷 VAD 残留段（is_final） |
| `NULL` | `new_meeting` | — | 清说话人质心与转写缓冲，开新会 |
| `NULL` | `set_status_file` | `const char*` 路径 | Stop 退出前自动写入 Markdown 运行状态快照（版本/时长/段数/丢弃/转写统计）；NULL 关闭 |
| `NULL` | `set_log` | `int` | 日志级别：-1=关闭，0..3=ERROR/WARN/INFO/DEBUG（作用域：仅音频流管理器日志，见 audio_link.h 注释） |
| `NULL` | `set_log_file` | `const char*`（AL_PARAM{path,字节数}） | 日志重定向到文件（单文件超 `max_bytes` 轮转为 `.1`，保留 1 份历史）；NULL=回退 stderr |
| `NULL` | `save_params` | — | 运行期专用配置命令修改写回启动 JSON：先写 `<cfg>.tmp` 再 `rename` 原子替换；dump 为**字典序键 + 2 空格缩进**标准 JSON（原注释/键顺序不保留，`null` 值键去除，值语义不变） |
| `"vad"` | `set_max_end_silence_ms` 等 7 个专用命令 | `AL_PARAM{&double, sizeof(double)}` | 热调 VAD 参数（清单见"动态配置"节），立即生效 |
| `"vad"` | `get_params` | `char[512]` 出参 | 当前 VAD 参数 JSON |
| `"asr"` | `set_language` | `const char*` | 强制语言（"Chinese" 等，NULL/空=自动） |
| `"asr"` | `clear_history` | — | 清 KV history（`enable_history=true` 时有效） |
| `"asr"` | `set_max_new_tokens` / `set_system_prompt` / `set_enable_history` | 数值 `AL_PARAM{&double,...}`；字符串仅 `set_system_prompt` | 热调（`set_enable_history` 切换时自动清 KV history；空串 prompt = 清空） |
| `"asr"` | `get_params` | `char[2048]` 出参 | 当前 asr 参数 JSON（含 language） |
| `"asr"` | `set_partial_cb` | `{AL_PARTIAL_CB cb; void* user;}` | 增量文本回调，NULL 取消 |
| `"asr"` | `get_model_info` | `char[2048]` 出参 | 维度推导结果 JSON（含 loaded 状态） |
| `"asr"` | `unload_model` | — | **真实卸载** asr bmodel 释放显存（总结前调用）；任务队列未清空时先轮询等待（最多 2s，超时返回 `E_AL_BUSY` 拒绝卸载） |
| `"asr"` | `load_model` | — | 重新加载 asr（重绑子网/维度推导/重建 dev_buffer） |
| `"diarize"` | `get_transcript` | `char[N]` 出参 | 全量转写 "讲话人N：xxx\n…"（总结模块输入） |
| `"diarize"` | `new_meeting` | — | 清质心+转写（等价管理器级 new_meeting） |
| `"diarize"` | `get_speaker_stats` | `char[1024]` 出参 | 各说话人段数统计 JSON |
| `"diarize"` | `set_threshold` / `set_max_speakers` | `AL_PARAM{&double, sizeof(double)}` | 热调（`set_max_speakers` 下调不收缩已建质心，仅告警） |
| `"diarize"` | `get_params` | `char[256]` 出参 | 当前聚类参数 JSON |
| `"pcm"` | `set_ring_ms` | `AL_PARAM{&double, sizeof(double)}` | 热调 ring_ms（int [1,60000]，m_mtx 下重建容量，缩容裁最旧保最新音频） |
| `"pcm"` | `get_params` | `char[256]` 出参 | 当前 PCM 参数 JSON（ring_ms + input_fmt 三键） |
| `"sink"` | `set_result_cb` | `{AL_RESULT_CB cb; void* user;}` | 注册整段结果回调 |
| `"<model>.<net>"` | `enable` / `disable` | — | 子网启停**标记**（当前仅登记状态供 list_nets 查询，不改变调度；能力级开关用各 Link Ctrl） |

## 错误码（E_AL_STATE）

| 错误码 | 含义 | 典型触发场景 |
|---|---|---|
| `E_AL_OK` | 成功 | — |
| `E_AL_PARAM_ERR` | 参数非法 | Ctrl 命令/参数越界、PushPcmEx 格式不符、name=value 解析失败 |
| `E_AL_NOT_INIT` | 未初始化 | 句柄 NULL 时调用、save_params 未加载配置 |
| `E_AL_LINK_NOT_FOUND` | Link 未注册 | Ctrl name 拼写错误 |
| `E_AL_MODEL_LOAD_FAIL` | bmodel 加载失败 | 文件缺失/损坏、芯片不匹配 |
| `E_AL_NET_NOT_MATCH` | 子网缺失/shape 不符 | roles 未命中、fbank 行数与 VAD 静态 T 失配、strict 校验 roles 段 |
| `E_AL_LAUNCH_FAIL` | 推理 launch 失败 | 设备异常（自愈机制会重试，3 连败上报） |
| `E_AL_NO_FRAME` | 拉模型无数据 | 缓冲不足（非错误，轮询语义） |
| `E_AL_TIMEOUT` | 对象池满 | 背压语义（pcm getFrame，非错误） |
| `E_AL_BUSY` | 资源忙 | unload_model 等待任务空闲超时（2s） |
| `E_AL_INTERNAL_ERR` | 内部错误 | save_params 写盘失败、自愈重试后仍失败 |

## 动态配置（专用命令 / get_params / save_params）

第一批 13 个专用命令支持运行期热调，协议：

- **数值参数统一 double 协议**：`param = AL_PARAM{&dVal, sizeof(dVal)}`（调用方
  `double dVal = ...;` 后取地址传入，类型安全无字符串解析歧义）；
  字符串参数仅 asr 的 `set_system_prompt`（`AL_PARAM{串, 字节数}`，空串表示清空提示词）；
- **立即生效**：校验通过即改运行成员（当段/当前块起用新值，不重启链路），
  并同步写回内存配置树；配置树写回失败（对应节缺失）仅 WARN 告警，运行期修改仍有效；
- 非法值（越界 / 非整数 / NaN / inf）：返回 `E_AL_PARAM_ERR` 并置 `get_last_error`
  （错误描述含期望范围与实际值），**原值不变**；
- **`save_params`**（管理器级）：把内存配置树写回启动 JSON——先清残留
  `<cfg>.tmp` → 写 tmp → `rename` 原子替换（中途断电不留半个配置文件）；
  dump 结果为**字典序键 + 2 空格缩进**的标准 JSON，原文件的注释与键顺序不保留
  （`null` 值键被去除），值语义不变，下次 Init 直接生效；
- **`get_params`**：输出当前生效值 JSON（数值型；`gate_enable`/`enable_history` 输出 0/1）。

专用命令清单（13 个，范围校验标准与启动校验一致）：

| Link | 命令 | 值类型/范围 | 写回位置 | 说明 |
|---|---|---|---|---|
| vad | set_max_end_silence_ms | double，int ≥ 1 | `vad.max_end_silence_ms` | 判停静音阈值（= funasr `max_end_silence_time`）；判停帧数 =(该值−`speech_to_sil_time_ms`)/10ms |
| vad | set_sil_to_speech_time_ms | double，int ≥ 1 | `vad.sil_to_speech_time_ms` | 静音→语音确认时长（= funasr `sil_to_speech_time_thres`） |
| vad | set_noise_thres | double [0,1] | `vad.speech_noise_thres` | funasr 加性噪声阈值：speech iff (1−P_sil) ≥ P_sil^ratio + thres |
| vad | set_max_seg_s | double，int [1,2147] | `vad.max_seg_s` | 强制切段上限（秒，= funasr `max_single_segment_time`/1000） |
| vad | set_gate_enable | double 0/1 | `vad.energy_gate.enable` | 能量前置门开关（置 1 即时武装清挂起期） |
| vad | set_gate_rms | double，int > 0 | `vad.energy_gate.rms_thres` | 门控 RMS 阈值 |
| vad | set_gate_hold_ms | double，int > 0 | `vad.energy_gate.hold_ms` | 门控挂起期 |
| asr | set_max_new_tokens | double，int > 0 | `asr.max_new_tokens` | 单段最大生成 token |
| asr | set_system_prompt | 字符串 `AL_PARAM{str, len}` | `asr.system_prompt` | 系统提示词（空串=清空；下一识别段生效） |
| asr | set_enable_history | double 0/1 | `asr.enable_history` | 跨段上下文开关（切换自动清 KV history） |
| diarize | set_threshold | double (0,1) | `spk.threshold` | 聚类归属阈值 |
| diarize | set_max_speakers | double，int ≥ 1 | `spk.max_speakers` | 说话人上限（小于已建数仅告警不收缩质心） |
| pcm | set_ring_ms | double，int [1,60000] | `pcm.ring_ms` | 环形缓冲时长（缩容裁最旧保最新音频，扩容保留现有内容） |

调用示例（数值 double 协议）：

```c
double dVal = 0.3;
AL_PARAM p = { &dVal, sizeof(dVal) };
AudioLinkCtrl(mgr, "vad", "set_noise_thres", p);   /* 立即生效（后续块用新阈值） */
AudioLinkCtrl(mgr, NULL, "save_params", (AL_PARAM){0});   /* 原子写回启动 JSON */
```

`get_params` 输出示例（vad）：

```json
{"max_end_silence_ms":800,"sil_to_speech_time_ms":150,"speech_to_sil_time_ms":150,"max_single_segment_ms":20000,"max_seg_s":20,"speech_noise_thres":0.6000,"gate_enable":0,"gate_rms_thres":300,"gate_hold_ms":600}
```

写回兼容性说明：`gate_enable` / `enable_history` 原为 JSON bool，热调写回经数值
通道存为 `0/1` 数值，Init 读回时**两种类型均兼容**，语义不变。

## 日志（al_log）

统一格式 `[YYYY-MM-DD HH:MM:SS.mmm][LEVEL][tag] msg`，级别
`ERROR(0)/WARN(1)/INFO(2)/DEBUG(3)`，默认 INFO 输出到 stderr。来源：
`config/audio/audio_link.json` 的 `"log"` 节（`level`/`file`/`max_bytes`，
Init 时应用）、运行期 Ctrl `set_log` / `set_log_file`。

## 回调

```c
typedef void (*AL_RESULT_CB)(const AL_RESULT *result, void *user);
```
- `start_ms/end_ms`：段起止（相对音频流起点，含 padding）。
- `text`：UTF-8 文本（ASR 未启用/失败为 NULL）。
- `spk_emb/spk_dim`：CAM++ embedding（未启用为 NULL）。回调返回后指针失效，需保留请拷贝。
- `speaker_id`：说话人编号（1 起始，diarize 在线聚类；0=未归属）。
- `line`：格式化行 "讲话人N：xxx"（无归属时为原文；可为 NULL）。

## MeetingSummary 会议总结模块（meeting_summary.h）

会议总结为**非流式的文字集合处理**，是独立模块（不做成链上节点）：

```c
AL_HANDLE MeetingSummaryInit(const char *json_cfg, int dev_id); /* 加载 text-only LLM */
int MeetingSummaryRun(AL_HANDLE h, const char *transcript, AL_PARAM out); /* out=char 缓冲 */
int MeetingSummaryRunFile(AL_HANDLE h, const char *text_path, AL_PARAM out); /* 从文本文件读入转写并总结 */
int MeetingSummaryCtrl(AL_HANDLE h, const char *cmd, AL_PARAM param);
/*   "set_prompt"（模板须含 {transcript}）/"set_max_tokens"（int）
     "set_partial_cb"（{AL_PARTIAL_CB cb; void* user;}，总结逐词流式回调，NULL 取消）
     "get_last_error"（char[]）/"get_model_info"（char[]，含 last_ftl_ms/last_tps/last_tok_num 性能统计） */
int MeetingSummaryDeinit(AL_HANDLE h);                          /* 释放总结模型显存 */
```

`MeetingSummaryRunFile`：独立总结模式，从 UTF-8 文本文件读取转写内容生成总结
（对齐官方 `application/meeting_summary` 例程的 meeting.txt 用法），不依赖音频流模块
（无需先 `AudioLinkInit`）。文本可为 "讲话人N：xxx\n…" 行集合或纯文本，
内部转调 `MeetingSummaryRun`（map-reduce 长转写处理全部继承）。

`MeetingSummaryRun` 内置 **map-reduce**（参考 `application/meeting_summary` 例程）：
转写文本单次放不下输入预算（`MAX_INPUT_LENGTH` 与 `SEQLEN-128-max_new` 取小）时，
自动按讲话人行边界分块 → 逐块生成分段摘要 → 合并终总结（摘要仍超预算则迭代
压缩，上限 4 轮）；短转写走单次生成原路径。

显存受限标准时序（上层手动驱动）：

```c
char transcript[16384] = {0};
AL_PARAM tp{transcript, sizeof(transcript)};
AudioLinkCtrl(mgr, "diarize", "get_transcript", tp);   /* 取转写文本集合 */

AudioLinkCtrl(mgr, "asr", "unload_model", {0});        /* 释放 qwen3_asr 显存 */
AL_HANDLE sm = MeetingSummaryInit("config/summary/meeting_summary.json", 0);
AL_PARAM op{out_buf, sizeof(out_buf)};
MeetingSummaryRun(sm, transcript, op);                   /* greedy 生成总结 */
MeetingSummaryDeinit(sm);                                /* 释放总结模型 */
AudioLinkCtrl(mgr, "asr", "load_model", {0});          /* 恢复音频流识别 */
AudioLinkCtrl(mgr, NULL, "new_meeting", {0});          /* 开新会 */
```

卸载期间：VAD/声纹/聚类照常（宿主内存），段无文本不追加转写行。

prompt 模板加载顺序（`config/summary/meeting_summary.json` 的 `summary` 节）：
**`prompt_file`（外置模板文件，须含 `{transcript}` 占位）> `prompt_template`
（JSON 内联模板）> 内置默认**（字段化纪要模板）。推荐用 `prompt_file`
外置模板（默认指向 `config/summary/summary_prompt.txt`），便于不改代码/配置热换模板。

```c
typedef void (*AL_PARTIAL_CB)(const char *partial_text, void *user);
```
ASR 逐词增量文本（打字机效果用），在任务线程上下文调用。

## 典型时序

```c
AL_HANDLE m = AudioLinkInit("config/audio/audio_link.json", 0);
struct { AL_RESULT_CB cb; void *user; } cb = {onResult, NULL};
AL_PARAM p{&cb, sizeof(cb)};
AudioLinkCtrl(m, "sink", "set_result_cb", p);
AudioLinkStart(m);
/* 采集线程循环： */
AudioLinkPushPcm(m, chunk, chunk_bytes);
/* 结束时： */
AudioLinkCtrl(m, NULL, "flush", {0});
AudioLinkStop(m);
AudioLinkDeinit(m);
```
