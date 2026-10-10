//===----------------------------------------------------------------------===//
//
// meeting_summary.h - MeetingSummary 会议内容总结模块（独立于音频流 Link 链）
// 非流式的文字集合处理：取转写文本（音频流模块 Ctrl "get_transcript"），
// 用 text-only LLM（LLM-TPU chat 格式 bmodel，Qwen3 系列通用）生成会议总结。
// 显存受限：总结前由上层 Ctrl("asr","unload_model") 卸载音频流大模型，
// 总结后 MeetingSummaryDeinit + Ctrl("asr","load_model") 恢复。
//
//===----------------------------------------------------------------------===//
#ifndef MEETING_SUMMARY_H
#define MEETING_SUMMARY_H

#include "audio_link.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 创建总结模块并加载 text-only LLM。
 * json_cfg 指向 meeting_summary.json（模型路径/tokenizer/prompt 模板）。
 * 失败返回 NULL（错误经 MeetingSummaryCtrl(h,"get_last_error") 查询）。
 */
AL_HANDLE MeetingSummaryInit(const char *json_cfg, int dev_id);

/**
 * 生成会议总结（阻塞，greedy 解码）。
 * @param transcript 转写文本集合（"讲话人1：xxx\n…"，来自 Ctrl "get_transcript"）
 * @param out        出参 char 缓冲（out.data/out.size），UTF-8 总结文本
 */
int MeetingSummaryRun(AL_HANDLE h, const char *transcript, AL_PARAM out);

/**
 * 独立总结模式：从文本文件读取转写内容生成总结（对齐官方 meeting.txt 用法）。
 * 不依赖音频流模块（无需先 AudioLinkInit），UTF-8 文本（"讲话人N：xxx\n…" 或纯文本），
 * 内部转调 MeetingSummaryRun（map-reduce 长转写全部继承）。
 * @param text_path 转写文本文件路径
 * @param out       出参 char 缓冲（out.data/out.size），UTF-8 总结文本
 */
int MeetingSummaryRunFile(AL_HANDLE h, const char *text_path, AL_PARAM out);

/** 控制口："set_prompt"（替换 prompt 模板，param.data=模板字符串，须含 {transcript}）
 *          "set_max_tokens"（int）/"get_last_error"（char[]）/"get_model_info"（char[]） */
int MeetingSummaryCtrl(AL_HANDLE h, const char *cmd, AL_PARAM param);

/** 释放总结模型显存并销毁模块 */
int MeetingSummaryDeinit(AL_HANDLE h);

#ifdef __cplusplus
}
#endif
#endif /* MEETING_SUMMARY_H */
