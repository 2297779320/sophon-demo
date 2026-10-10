# MixedAudioLink 五轴代码评审报告

评审框架：code-review-and-quality（correctness / readability / architecture /
security / performance），独立子 agent 全量扫描（22 源文件 + 配置 + CMake + docs）
+ 主导者定向复核。前序已有 3 轮 correctness 专项评审，本轮聚焦其余四轴与补漏。

## 评审发现与处置

### Security / 健壮性
| 级别 | 发现 | 处置 |
|---|---|---|
| Required | `pcm_link.h`：JSON `chunk_ms/ring_ms` 无下界校验，负值导致 erase/assign 负长度迭代（UB）或环形容器回绕无界增长 | ✅ 已修：`create()`/`chunkSamples()` 下界钳制 |
| Optional | `al_json.h` 递归下降无深度上限，畸形配置可栈溢出（本地文件风险低） | 遗留（建议加 64 层深度计数） |
| Nit | `readNpyF32` 对 hlen/rows*cols 无上限（恶意 npy 巨分配） | 遗留（本地文件风险低） |
| FYI | main.cpp wav 解析对畸形文件不死循环但无防御（demo-only） | 遗留 |
| 通过 | snprintf 出参全量核查（10 处）均按 size 截断，无溢出；外部输入（JSON/PCM/transcript/Ctrl param）无未校验直达 memcpy/下标的路径 | — |

### Performance
| 级别 | 发现 | 处置 |
|---|---|---|
| Required | `asr_mel.h`：`std::norm` 写在 128 bin 内层，每帧 128×201 次复数模方（~128 倍冗余，实时热路径） | ✅ 已修：功率谱先算 `spec[201]` 再点积 |
| Required | `audio_summary.cpp` splitByBudget 每行 `encode(cur+line)` 重编码全部累积文本，O(N²) | ✅ 已修：增量 token 计数（每行 encode 一次） |
| Required | `text_llm.h` forwardNext 每 token 重建 mask(SEQLEN×2B) + 每层 8 个 vector 堆分配 | ✅ 已修：`m_next_mask/m_dec_ins/m_dec_outs` 成员复用 |
| Nit | pcm_link vector 头部 erase O(n)（200ms 一次 ~150KB memmove） | 遗留（量级可接受，可换环形下标） |
| FYI | vad 滤波器构造期建表非每块重算 ✓；diarize ≤8 人线性扫描可接受 ✓；对象池锁无瓶颈 ✓ | — |

### Correctness 补漏
| 级别 | 发现 | 处置 |
|---|---|---|
| Nit | `asr_tokenizer.h` encode 特殊 token 走 `m_tok2id.at()`，vocab 缺该 token 时 out_of_range terminate | ✅ 已修：优先 specials 表 |
| Nit | `set_max_tokens` 未校验正值，负值静默空结果且反向放大 inputBudget | ✅ 已修：[1, SEQLEN-128] 校验 |
| Nit | `al_json.h` \u 序列不足 4 hex 时错位消费 | ✅ 已修：逐字符消费+原样保留 |
| 通过 | 浮点比较、std::function 空调用（P1-3 已判空）、负错误码链路复验无新问题 | — |

### Readability / 死代码（**已全部删除，经用户确认**）
```
DEAD CODE REMOVED:
- model_hub.h netEnabled()            — 删除（Ctrl enable/disable 降级为标记后无调用方）
- link_mng.h listLinkNames()          — 删除（list_nets 走 hub.dumpNets()）
- link_mng.h CBaseLink::releaseFrame() — 删除（删除器直调 frameRelease）
- link_mng.h m_vad_model/m_spk_model  — 删除（vad/spk link 硬编码模型键名 "vad"/"spk"）
- ALFrame::type                       — 删除（只写不读）
- CSinkLink::emit 的 is_final 参数    — 删除（从未消费；diarize.process 同步去参）
```
另：audio_summary "llm not ready" 返回 E_AL_NET_NOT_MATCH 与 asr 同场景
E_AL_NO_FRAME 不一致（建议统一，语义级 Nit）。

### Architecture
| 级别 | 发现 | 处置 |
|---|---|---|
| Required | `enable/disable` Ctrl 宣称"参与调度"实为 no-op（disabled 集合无消费者），API 文档失真 | ✅ 已修：audio_link.h/API.md 如实降级为"启停标记，能力级开关用各 Link Ctrl" |
| FYI | diarize→sink 用 findLink 反查、feat→asr 用回调注入，两种解耦并存 | 遗留（可统一为注入） |

## 验证边界
静态评审（无编译环境）：符号/类型/逻辑逐条人工核查 + 5 轮 agent 交叉评审；
编译期与运行时行为以上板实测为准。

## Verdict
- Critical：0　Required：5（**全部已修**）　Nit：6（3 已修，3 遗留）　Optional/FYI：3
- **结论：Approve** —— Required 全部闭环，遗留项均为 Nit/Optional（死代码删除
  待用户裁决、递归深度/数值对齐属低风险加固），不阻塞。
