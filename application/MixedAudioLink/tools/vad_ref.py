#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vad_ref.py - FSMN-VAD 前端数值参考实现（FunASR/kaldi 语义）

用途：为 cpp/al_feature.h 的 C++ 前端实现提供逐帧数值基准，用于交叉验证。
本脚本严格复刻 FunASR 官方前端的计算链：
  wav(int16) --kaldi fbank--> [T, 80] --apply_lfr--> [T, 400] --apply_cmvn--> [T, 400]

来源依据：
  - funasr/frontends/wav_frontend.py : WavFrontend.forward -> kaldi.fbank(...) -> apply_lfr -> apply_cmvn
      默认 snip_edges=True, window=hamming, n_mels=80, frame_length=25ms, frame_shift=10ms
      官方 vad_config.yaml 显式设 dither=0.0, lfr_m=5, lfr_n=1
  - kaldi.fbank 输入为 int16 量纲（FunASR 内部 `waveform * (1 << 15)`）
  - load_cmvn: AddShift 记作 means, Rescale 记作 vars
  - apply_cmvn: (x + means) * vars

用法:
  python3 vad_ref.py <wav> <am.mvn> [--out-prefix P] [--chunk-samples 3200]

输出（默认写到当前目录）:
  <P>_full_fbank.npy     整段 kaldi fbank         [T, 80]  float32
  <P>_full_lfr_cmvn.npy  整段 LFR+CMVN            [T, 400] float32
  <P>_chunk0_fbank.npy   第 0 块 kaldi fbank      [n, 80]
  <P>_chunk0_lfr_cmvn.npy 第 0 块 LFR+CMVN        [n, 400]
  <P>_cmvn.npy           [2, 400]  (means, vars) 供 C++ 侧一致性核对
"""
import sys
import wave
import argparse
import numpy as np
import kaldi_native_fbank as knf


# ----------------------------------------------------------------------------
# FunASR 官方前端复刻
# ----------------------------------------------------------------------------
def load_cmvn(cmvn_file):
    """复刻 funasr 的 load_cmvn：<AddShift> 后一行的 [ ] 内为 means，<Rescale> 后一行为 vars。"""
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.split()]
    means = vars_ = None
    for i, ln in enumerate(lines):
        tok = ln.split()
        if tok[0] == "<AddShift>" and i + 1 < len(lines):
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                means = np.array(nxt[3:len(nxt) - 1], dtype=np.float32)
        elif tok[0] == "<Rescale>" and i + 1 < len(lines):
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                vars_ = np.array(nxt[3:len(nxt) - 1], dtype=np.float32)
    if means is None or vars_ is None:
        raise ValueError("no <AddShift>/<Rescale> found in " + cmvn_file)
    return means, vars_


def apply_lfr(inputs, lfr_m, lfr_n):
    """复刻 funasr 的 apply_lfr（左补 (lfr_m-1)//2 帧，右补末帧，stride=lfr_n）。"""
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    pad_l = (lfr_m - 1) // 2
    out = np.zeros((T_lfr, lfr_m * inputs.shape[1]), dtype=np.float32)
    padded = np.vstack([inputs[0:1].repeat(pad_l, axis=0), inputs]) if pad_l else inputs
    Tp = padded.shape[0]
    for i in range(T_lfr):
        for m in range(lfr_m):
            t = i * lfr_n + m
            src = padded[t] if t < Tp else padded[Tp - 1]
            out[i, m * inputs.shape[1]:(m + 1) * inputs.shape[1]] = src
    return out


def apply_cmvn(inputs, cmvn):
    """复刻 funasr 的 apply_cmvn: (x + means) * vars """
    means, vars_ = cmvn
    dim = inputs.shape[1]
    return ((inputs + means[:dim]) * vars_[:dim]).astype(np.float32)


def kaldi_fbank(samples_int16_as_float, n_mels=80, snip_edges=True, dither=0.0):
    """kaldi 风格 fbank（kaldi_native_fbank，与 funasr 所用的 kaldi.fbank 同源）。"""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.dither = dither
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.snip_edges = snip_edges
    opts.mel_opts.num_bins = n_mels
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0
    opts.use_energy = False
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, samples_int16_as_float.tolist())
    fb.input_finished()
    n = fb.num_frames_ready
    mat = np.zeros((n, n_mels), dtype=np.float32)
    for i in range(n):
        mat[i] = fb.get_frame(i)
    return mat


def read_wav_int16(path):
    w = wave.open(path, "rb")
    assert w.getsampwidth() == 2, "only 16-bit PCM wav supported"
    nch, nfr = w.getnchannels(), w.getnframes()
    raw = w.readframes(nfr)
    w.close()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("cmvn")
    ap.add_argument("--out-prefix", default="vad_ref")
    ap.add_argument("--chunk-samples", type=int, default=3200)
    ap.add_argument("--lfr-m", type=int, default=5)
    ap.add_argument("--lfr-n", type=int, default=1)
    args = ap.parse_args()

    samples = read_wav_int16(args.wav)          # int16 量纲（kaldi 约定）
    print("[ref] wav samples = %d (%.2fs)" % (len(samples), len(samples) / 16000.0))

    means, vars_ = load_cmvn(args.cmvn)
    print("[ref] cmvn dim = %d ; means[0:3] = %s ; vars[0:3] = %s"
          % (len(means), means[:3], vars_[:3]))
    np.save(args.out_prefix + "_cmvn.npy",
            np.stack([means, vars_]).astype(np.float32))

    # ---- 整段 ----
    fb_full = kaldi_fbank(samples)
    print("[ref] full fbank shape =", fb_full.shape,
          "mean %.4f min %.4f max %.4f" % (fb_full.mean(), fb_full.min(), fb_full.max()))
    lfr_full = apply_lfr(fb_full, args.lfr_m, args.lfr_n)
    feat_full = apply_cmvn(lfr_full, (means, vars_))
    print("[ref] full lfr_cmvn shape =", feat_full.shape,
          "mean %.4f min %.4f max %.4f" % (feat_full.mean(), feat_full.min(), feat_full.max()))
    np.save(args.out_prefix + "_full_fbank.npy", fb_full)
    np.save(args.out_prefix + "_full_lfr_cmvn.npy", feat_full)

    # ---- 第 0 块（模拟框架逐块处理：仅用块内样本做 fbank/LFR/CMVN）----
    c0 = samples[:args.chunk_samples]
    fb0 = kaldi_fbank(c0)
    lfr0 = apply_lfr(fb0, args.lfr_m, args.lfr_n)
    feat0 = apply_cmvn(lfr0, (means, vars_))
    print("[ref] chunk0 fbank shape =", fb0.shape, " lfr_cmvn shape =", feat0.shape)
    np.save(args.out_prefix + "_chunk0_fbank.npy", fb0)
    np.save(args.out_prefix + "_chunk0_lfr_cmvn.npy", feat0)

    # ---- 模拟 funasr 流式（带 input_cache carry-over）每块帧数 ----
    cache = np.zeros(0, dtype=np.float32)
    counts = []
    for k in range(8):
        blk = samples[k * args.chunk_samples:(k + 1) * args.chunk_samples]
        if len(blk) < args.chunk_samples:
            break
        cur = np.concatenate([cache, blk])
        nfr = (1 + (len(cur) - 400) // 160) if len(cur) >= 400 else 0
        cache = cur[len(cur) - (len(cur) - nfr * 160):]
        counts.append(nfr)
    print("[ref] funasr streaming frame counts per 200ms chunk =", counts)


if __name__ == "__main__":
    main()
