#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vad_decode_check.py - 用官方 ONNX 模型验证「特征 + 248 类解码 + 状态机」链路

验证目标（无板环境下的最大可及验证）：
  1) 用 tools/vad_ref.py 的 golden 400 维特征喂官方 ONNX，确认模型能正常出 [1,T,248]
  2) 按 FunASR GetFrameState 公式判决每帧语音/静音，检查阈值是否落在合理区间
  3) 用框架的状态机参数（min_speech/min_silence/padding）切段，打印时间戳

注：bmodel 需在板端运行，此处以同源 ONNX 代替（前端已与官方逐帧对齐，见 compare_feat.py）。

用法: python vad_decode_check.py <speech_fsmn_vad_opt.onnx> <ref_full_lfr_cmvn.npy> [thres] [ratio]
"""
import sys
import numpy as np
import onnxruntime as ort


def main():
    onnx_path = sys.argv[1]
    feat_path = sys.argv[2]
    thres = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    feat = np.load(feat_path).astype(np.float32)      # [T, 400]
    T = feat.shape[0]
    print("[chk] feat %s  T=%d dim=%d" % (feat.shape, T, feat.shape[1]))

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("[chk] inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[chk] outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

    feed = {}
    for i in sess.get_inputs():
        n = i.name.lower()
        if "speech" in n or n in ("input", "feats"):
            tmax = i.shape[1] if (len(i.shape) == 3 and isinstance(i.shape[1], int)) else T
        elif "cache" in n:
            feed[i.name] = np.zeros((1, 128, 19, 1), dtype=np.float32)
        else:
            print("[chk] WARN unknown input", i.name)
    print("[chk] 模型内定 T =", tmax, "（按此分块流式喂入，cache 递推）" if tmax and tmax < T else "")

    # 分块流式：每块 tmax 帧，cache 在块间递推（模拟 FunASR streaming）
    step = tmax if (tmax and tmax < T) else T
    caches = {}
    for i in sess.get_inputs():
        if "cache" in i.name.lower():
            caches[i.name] = np.zeros((1, 128, 19, 1), dtype=np.float32)
    logits_all = []
    for b0 in range(0, T, step):
        blk = feat[b0:b0 + step]
        if blk.shape[0] < step:                       # 末块：复制末帧补齐（等价 LFR 右补末帧）
            pad = np.repeat(blk[-1:], step - blk.shape[0], axis=0)
            blk = np.concatenate([blk, pad], axis=0)
        feed = {}
        for i in sess.get_inputs():
            n = i.name.lower()
            if "speech" in n or n in ("input", "feats"):
                feed[i.name] = blk[None, :, :]
            elif "cache" in n:
                feed[i.name] = caches[i.name]
        out = sess.run(None, feed)
        names = [o.name for o in sess.get_outputs()]
        for n, o in zip(names, out):
            if "cache" in n.lower():
                caches[n] = np.asarray(o)
        lg = None
        for n, o in zip(names, out):
            if "logits" in n.lower():
                lg = np.asarray(o)
        if lg is None:
            lg = np.asarray(out[0])
        if lg.ndim == 3:
            lg = lg[0]
        logits_all.append(lg[:min(step, T - b0)])
    logits = np.concatenate(logits_all, axis=0)

    print("[chk] logits shape=%s  rowsum mean=%.4f  min=%.4f max=%.4f"
          % (logits.shape, logits.sum(axis=1).mean(), logits.min(), logits.max()))
    # 判别是否已 softmax：行和≈1 则为概率
    if abs(float(logits.sum(axis=1).mean()) - 1.0) > 0.05:
        print("[chk] 行和 != 1 -> 视为未 softmax，施加 softmax")
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        logits = e / e.sum(axis=1, keepdims=True)
        print("[chk] softmax 后 rowsum mean=%.4f" % logits.sum(axis=1).mean())

    scores = logits[:T]
    p_sil = scores[:, 0]                                # sil_pdf_ids = [0]
    p_speech = 1.0 - p_sil
    thresh = np.power(np.maximum(p_sil, 1e-12), ratio) + thres
    is_speech = p_speech >= thresh

    print("[chk] P_sil: mean %.4f min %.4f max %.4f" % (p_sil.mean(), p_sil.min(), p_sil.max()))
    print("[chk] 判决阈值 thres=%.2f ratio=%.2f -> 语音帧比例 %.1f%% (%d/%d)"
          % (thres, ratio, 100.0 * is_speech.mean(), int(is_speech.sum()), T))

    # 游程
    runs = []
    i = 0
    while i < T:
        j = i
        while j < T and is_speech[j] == is_speech[i]:
            j += 1
        runs.append(("SPEECH" if is_speech[i] else "sil", i * 10, j * 10))
        i = j
    print("[chk] 帧级游程(ms):", ", ".join("%s[%d,%d)" % r for r in runs))

    # 框架状态机（min_speech_ms=250 / min_silence_ms=500 / padding=300）
    min_sp, min_sil, pad = 250, 500, 300
    segs = []
    in_sp, beg, last_voice, sil = False, 0, -1, 0
    for f in range(T):
        t = f * 10
        if not in_sp:
            if is_speech[f]:
                if beg == 0 and not segs and f > 0:
                    pass
                if last_voice < 0:
                    beg = t
                last_voice = t
                if t - beg + 10 >= min_sp:
                    in_sp = True
                    sil = 0
            else:
                beg, last_voice = 0, -1
        else:
            if is_speech[f]:
                last_voice, sil = t, 0
            else:
                sil += 10
                if sil >= min_sil:
                    segs.append((max(0, beg - pad), last_voice + pad))
                    in_sp, beg, last_voice, sil = False, 0, -1, 0
    if in_sp:
        segs.append((max(0, beg - pad), last_voice + pad))
    print("[chk] 状态机切段(ms):", segs if segs else "无")
    print("[chk] 段数 =", len(segs), " 总语音时长 =", sum(b - a for a, b in segs), "ms")


if __name__ == "__main__":
    main()
