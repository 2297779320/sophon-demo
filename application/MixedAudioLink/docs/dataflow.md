# 数据流与实现细节

## 1. PCM 通路

```
AudioLinkPushPcm(int16) ──▶ CPcmLink 环形缓冲（pcm.ring_ms，默认 5000ms）
   满 → 丢最旧 + m_dropped_bytes 累计（Ctrl "get_stats" 查询）
CVadLink::getFrame 每次拉 chunk_ms（默认 200ms = 3200 样本）
```

输入格式（P1①）：`AudioLinkPushPcm` 按 JSON `pcm.input_fmt`（默认 16k/1/16）
解释；`AudioLinkPushPcmEx` 显式传格式（首次推入锁存，此后不一致拒绝）。
非 16k/mono 输入经内置多相 FIR 重采样（`al_resample.h`：L/M 有理数比分解 +
每相位 32 tap Blackman 加窗低通 + 多声道均值混缩 + int16 饱和）转 16k/mono
后入环；16k/mono 走 bypass 直推路径（逐字节不变）。

启动时 `CLinkMng::ValidateConfig` 两段式全量校验（通用必需键/文件存在 + 音频键
`validateAudio` 钩子 → roles 命中子网），顶层 `"strict": true` 时失败拒绝启动，
否则 WARN 后沿用默认/钳制值（钳制点：`pcm.ring_ms`、`vad.chunk_ms`、
`vad.max_seg_s`）。

参数生效语义：运行期各 Link **专用 Ctrl 命令**（vad 7 / asr 3 / diarize 2 / pcm 1，
数值统一 double 协议）热调**立即生效**（后续块/段即用新值，不重启链路）并同步
内存配置树，`save_params` 原子写回启动 JSON（协议与命令清单见
docs/API.md 动态配置节）。

## 2. FSMN VAD（vad_link）

子网 I/O 约定（JSON 可覆盖，默认按 LLM-TPU 惯例）：
- `inputs[0]`：fbank `[1, T, 80]`，静态 T（T = chunk 对应帧数）；
- `inputs[1..n-1]`：in_cache（块间状态，静态 shape）；
- `outputs[0]`：每帧概率，`[T,2]` 取 `[:,1]` 为语音概率，`[T]` 直读；
- `outputs[1..n-1]`：out_cache，`bm_memcpy_d2d_byte` 回写 in_cache
  （与 LLM block_cache 相同的 d2d 模式）。

前处理（funasr 官方默认，JSON `vad.fbank` 可调）：
去直流 → kaldi 风格 fbank（80 bins / 25ms / 10ms / 预加重 0.97）→ CMVN
（`vad.cmvn_mean/std` npy 文件，加载校验维度，缺省禁用）。

切段状态机（对齐 funasr `speech_noise_thres` 语义 `p = p_speech - thres*p_sil`）：
- 帧计时：真实帧间隔 = fbank 帧移（`frameShift()`，默认 160 样本 = 10ms），
  转写时间戳 = `chunk_beg + 帧号 × 帧移ms`，onset 样本偏移 = `onset帧号 × 帧移样本数`，
  分钟级无累计漂移；
- launch 前按概率输出 shape 推 T_out/通道数（`[T]`/`[T,2]`/`[1,T(,2)]`），
  fbank rows != T_out 时记日志并返回 `E_AL_NET_NOT_MATCH` 跳过本块（不做静默截断）；
- 空闲：`p>0` 累计，达 `min_speech_ms`（250）确认起点（回退 padding 300ms），
  预卷缓冲入段（预卷缓冲每块追加/裁剪一次；语音起点落在块中间时，
  从 onset 样本偏移起把该块剩余部分接入段）；
- 说话中：每块整段追加进当前段，静音连续达 `min_silence_ms`（500）→
  段尾 = 最后语音帧 + padding → 出段并 cache 归零（funasr `cache={}` 语义）；
- `Ctrl "flush"`：强制收尾 `is_final=true`；
- **强制切段上限**（`vad.max_seg_s`，默认 30s）：连续噪音/音乐下 VAD 一直判
  in_speech 时，段长达到上限即强制切段下发（is_final=true，日志标记），防止
  `m_seg_pcm` 无界增长与 ASR 输入超预算；
- **能量前置门**（`vad.energy_gate`，默认关闭）：非语音 + 块 RMS 低于
  `rms_thres` 时跳过 VAD 推理与 fbank 计算（省 TPU/功耗，7x24 设备静音期
  近零开销）；`hold_ms`（默认 600ms）挂起期保证语音结束后状态机正常收尾；
  块级判定不丢音频——本块 RMS 超阈即恢复全速。

## 3. 特征三支路（feat_link / asr_mel）

| 支路 | 消费者 | 特征 | 备注 |
|---|---|---|---|
| mel128 | CAsrLink | WhisperFeatureExtractor 复现：n_fft=400（非 2 的幂，FFT 走 Bluestein/chirp-z，频点与标准 DFT 一致）、hop=160、hann、中心对齐 pad=n_fft/2（whisper center=True）、128 mel、log10(clamp 1e-10)、clamp(max-8)、(x+4)/4 | 整秒对齐 → 按 AUDIO_LENGTH 分块；mel 滤波器优先 `asr.mel_filters` npy（transformers 自带 mel_filters.npy 拷入即可，201 列），缺省用 slaney 惯例现算（**数值可能有微小差异**） |
| fbank80 | CSpkLink | kaldi fbank 80 + utterance 均值减 | 与 sample/CAM++ 的 feature_fbank 等价简化 |
| fbank+CMVN | CVadLink | 见上节 | |

数值对齐提示：mel 滤波器组与 STFT 的窗/中心对齐方式与 transformers/librosa
存在实现自由度，量产前建议用同一 wav 对比 Python 侧
`WhisperFeatureExtractor` 输出（最大绝对误差 < 1e-3 即可）。

## 4. qwen3_asr 推理序列（asr_link，LLM-TPU chat.cpp 移植）

1. **prompt**：`chat_template.json` 语义硬编码渲染
   `<|im_start|>system … <|im_start|>user <|audio_start|>{audio_pad×N}<|audio_end|>…`
   ，`N = n_times × tokens_per_chunk`，`tokens_per_chunk` 由 audio 子网输出 mem
   字节数推导（`out_bytes / (HIDDEN_SIZE × sizeof(uint16_t))`，每块 13 token，
   推导失败回退 13）；可选 `language xx<asr_text>` 后缀；
2. **forward_embed**：input_ids 补零到 MAX_INPUT_LENGTH，输出 embed 暂存 dev_buffer
   （子网 launch 失败返回错误码，生成链立即终止；`netLaunch` 返回 bool，
   forwardFirst/forwardNext 失败以负 token 码上传，调用方按 `token<0` 判错）；
3. **forward_audio**：逐块（[128, AUDIO_LENGTH]）launch audio 子网，
   输出 embed 按 `audio_bos_pos+1+i*tpc` 的 token 偏移 d2d 写入 dev_buffer；
4. **forward_first**：mrope 三路 position_ids（文本退化为三路相同 arange +
   `history_max_posid` 偏移）补齐 3×MAX，静态因果 mask（bf16=0xC61C），
   `m_max_posid = 末位 position_id`（对齐 pipeline.py `position_ids.max()`），
   逐层 block_i（block_0 输入先 empty_net 清零），KV 写入 past_key/value
   （= block_cache_i stages mems），lm_head 取末位 → greedy_head；
5. **forward_next** 循环：embedding_cache → block_cache_i（KV 偏移
   `(history_length-1)*KV_BYTES`；pos/mask 所有层复用 block_cache_0 的 mem）→
   lm_head，至 `<|im_end|>` 或 `max_new_tokens`。逐词 decode（双 token 技巧
   去重叠前缀，对齐 pipeline.py），`<asr_text>` 清空重计。

### 4.1 io_alone（addr_mode）双重检测与 KV 写入分支（text_llm.h）

- **检测**（init 内，m_ready 前完成；reload 走 deinit→init 自动重检，无状态残留）：
  ① 声明初判：`block_cache_0->addr_mode == 1`；
  ② 运行时实证：block_cache_0 的 KV 输入(input_mems[3]/[4]) 与 KV 输出
  (output_mems[1]/[2]) 设备地址一致（`bm_mem_get_device_addr` 比对，参照
  Qwen3 chat.cpp `is_same_addr` 思路）；两者结合置 `m_io_alone`；
  各层 addr_mode 不一致时 logErr 并取首层判定；检测结果打日志
  `[model] io_alone detect: addr_mode=x, io_alone=y`；
- **io_alone=true**：forward_first 逐层 outs[1]/[2] 直接重绑 past_key/value
  （offset 0，shape 用 block 的 output_shapes[1]/[2]），prefill KV 原地写入
  past 缓存，省去尾部 2×NUM_LAYERS 次 KV d2d 回拷；forward_next 维持 KV 偏移
  `(history_length-1)*KV_BYTES` 原地写（行为同改前）；
- **io_alone=false（BASIC 回退）**：forward_first 走原 d2d 回拷路径（保留）；
  forward_next 的 outs[1]/[2] 绑 stage 独立输出 mem，launch 后每层 d2d 回写
  past+offset（对齐 ChatGLM3 chat.cpp decode 逐层回拷路径）；
- ⚠️ 板端验证待办：addr_mode=1 与 =0 两分支各跑一次整段识别，逐 token 对比
  改前输出（要求完全一致）。

## 5. 线程与并发约束

| 线程 | 独占资源 | 节奏 |
|---|---|---|
| VAD 线程（CAudioLinkMng） | vad 模型 bmrt 上下文 | 200ms 块持续（PCM 不足时 5ms 轮询） |
| 任务线程（CAudioLinkMng） | asr / spk bmrt 上下文 | 段驱动，串行 ASR→SPK→Diarize |

双线程为音频工作流专属（`CAudioLinkMng::startWorkers/stopWorkers` 钩子起停，
线程对象留在基座 `CLinkMng`，先 join 再析构保证派生类析构路径安全）。

同一 bmrt 上下文内严格串行 launch；跨上下文天然并行（双线程已覆盖实时需求）。
模型组织前提：**所有模型均为独立 bmodel 文件、独立 bmrt 上下文（无融合模型）**，
显存占用 = Σ(常驻模型)，跨模型无共享（详见 docs/README.md 模型组织节）。

链路级自愈（P1②）：VAD launch 失败（`E_AL_LAUNCH_FAIL`）时复位设备 cache 与段状态
后重试一次；ASR 段处理失败时清 KV history（`clear_history`）后重试一次。任何一次
成功即恢复并计 `self_heal_count`（`get_stats` 可查）；连续失败逐次 WARN 上报，
连续 3 次升级 logErr（`E_AL_NET_NOT_MATCH` 等 shape 类错误不重试，直接计数）。

## 6. 说话人在线聚类（diarize_link）

- 每段 CAM++ embedding 与已有质心算余弦（`cos = dot/(|a||b|+1e-6)`）；
- `max_cos >= spk.threshold`（默认 0.6）→ 归入该说话人，
  质心滑动更新 `c = (c*utt_cnt + emb)/(utt_cnt+1)`（适应信道/音色漂移）；
- 否则新建说话人（编号 = 已有数+1），上限 `spk.max_speakers`（默认 8），
  超限时强制归入最相似者（不再分裂）；
- 无声纹段（spk 失败/未启用）`speaker_id=0`，行不带"讲话人N"前缀；
- **asr 卸载期间**：段仅做声纹聚类（归属继续学习），无文本不追加转写行；
- 转写缓冲/质心均在宿主内存，`unload_model` 不影响；`new_meeting` 清空。

## 7. 按模型卸载/重载（显存受限）

```c
AudioLinkCtrl(mgr, "asr", "unload_model", {0});  /* asr.onModelUnload → hub.unloadModel */
```
- `unload_model`：先置**卸载闸门** `m_task_pause`（任务线程不再取新段），再在
  `m_seg_mtx` 下轮询等待任务队列清空且任务线程空闲
  （最多 2s，超时返回 `E_AL_BUSY`；`m_task_busy` 由 `popSegment` 持锁置位，
  无 TOCTOU 窗口），再释放 Link 侧资源（CTextLLM dev_buffer、KV 引用失效），
  后 `bmrt_destroy` 释放全部子网设备内存；`bm_handle` 保留；完成/失败均清闸门；
- `load_model`：`bmrt_create + bmrt_load_bmodel` + 子网重枚举 + 角色映射 +
  **维度重推导**（deriveAsrDims）+ Link 侧重新绑定（CTextLLM::reload）；
- v1 仅支持卸载 asr（vad/spk 为小模型且 link 状态与其绑定）；卸载期间
  `processSegment` 直接返回 `E_AL_NO_FRAME`（任务线程跳过）；
- MeetingSummary 模块持有独立 CModelHub/bm_handle，与音频流模块互不干扰，
  由上层保证两者不同时驻留大模型（推荐时序见 docs/README.md）。

## 8. MeetingSummary 模块（text-only LLM）

- 模型要求：LLM-TPU chat 格式子网（embedding/block_i/block_cache_i/lm_head
  [/greedy_head]），**无 audio 子网**（deriveAsrDims 自动按 text-only 推导）；
- prompt：`config/meeting_summary.json` 加载顺序 `prompt_file`（外置模板文件，须含
  `{transcript}` 占位，默认指向 `config/summary_prompt.txt`）> `prompt_template`
  （JSON 内联模板）> 内置默认，再经 Qwen 文本对话模板渲染 + QwenTokenizer encode；
- **map-reduce 长转写**（参考 `application/meeting_summary` 例程）：输入预算 =
  min(MAX_INPUT_LENGTH, SEQLEN-128-max_new)；超预算时按讲话人行边界分块
  （单行超限按 token 比例硬切，**UTF-8 字符边界对齐**防切断汉字；模板开销
  吞掉预算时明确报错）→ 逐块 `prompt_template` 生成分段摘要 →
  `kMergeTemplate` 合并终总结，摘要仍超预算则迭代压缩（上限 4 轮）；
- 默认 prompt 为**字段化纪要模板**（基本信息/议程/摘要含发言人/行动项含
  责任人+完成时间/下次会议，缺省填"未提及"，对齐官方 prompt.txt）；
- 流式：Ctrl "set_partial_cb" 注册逐词回调，单块生成与 reduce 合并终总结
  阶段生效（map 分段阶段不推流）；
- 性能统计：generate 路径记录首 token 延迟（FTL）/decode token 数/TPS，
  经 Ctrl "get_model_info" 读取（asr 分步 prefill 路径仅 token 数有效）；
- greedy 解码至 `<|im_end|>` 或 `max_new_tokens`；每次 generate 前 clearHistory。

## 9. 扩展新模型/子网

1. JSON `models` 增加键 + `path` + `roles` 关键字；
2. 需要新链路能力时派生 `CBaseLink`，在 `CAudioLinkMng::createLinkImpl` 工厂注册新 type；
3. 未匹配子网自动暴露于 `list_nets`，可用 `Ctrl enable/disable` 控制。
