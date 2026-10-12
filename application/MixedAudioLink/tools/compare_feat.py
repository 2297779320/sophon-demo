#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_feat.py - 对比 C++ 前端(feat_dump) 与 FunASR/kaldi 参考(vad_ref) 的逐帧差异。"""
import sys
import numpy as np


def load_bin(p):
    with open(p + ".dim") as f:
        r, c = [int(x) for x in f.read().split()]
    return np.fromfile(p, dtype=np.float32).reshape(r, c)


def report(tag, cpp, ref):
    if cpp.shape != ref.shape:
        print("[%s] SHAPE MISMATCH cpp=%s ref=%s" % (tag, cpp.shape, ref.shape))
        return
    d = cpp - ref
    print("[%s] shape=%s" % (tag, cpp.shape))
    print("      cpp  mean %+.5f  min %+.5f  max %+.5f" % (cpp.mean(), cpp.min(), cpp.max()))
    print("      ref  mean %+.5f  min %+.5f  max %+.5f" % (ref.mean(), ref.min(), ref.max()))
    print("      diff mean %+.6f  std %.6f  min %+.6f  max %+.6f  |max| %.6f"
          % (d.mean(), d.std(), d.min(), d.max(), np.abs(d).max()))
    mx = np.abs(d).max()
    if mx < 1e-3:
        print("      => 一致（|max|<1e-3）")
    elif d.std() < 1e-4:
        print("      => 纯常量偏移 %.6f" % d.mean())
    else:
        print("      => 存在逐点差异，相对量级 %.3f%%（ref span %.4f）"
              % (100.0 * d.std() / (ref.max() - ref.min() + 1e-9), ref.max() - ref.min()))


def main():
    cpp_dir, ref_dir = sys.argv[1], sys.argv[2]
    print("=== full fbank (?,80) ===")
    report("full.fbank", load_bin(cpp_dir + "/cpp.fbank.bin"), np.load(ref_dir + "/ref_full_fbank.npy"))
    print("=== full lfr+cmvn (?,400) ===")
    report("full.lfr_cmvn", load_bin(cpp_dir + "/cpp.lfr_cmvn.bin"),
           np.load(ref_dir + "/ref_full_lfr_cmvn.npy"))
    print("=== chunk0 fbank (18,80) ===")
    report("chunk0.fbank", load_bin(cpp_dir + "/cpp.chunk0.fbank.bin"),
           np.load(ref_dir + "/ref_chunk0_fbank.npy"))
    print("=== chunk0 lfr+cmvn (18,400) ===")
    report("chunk0.lfr_cmvn", load_bin(cpp_dir + "/cpp.chunk0.lfr_cmvn.bin"),
           np.load(ref_dir + "/ref_chunk0_lfr_cmvn.npy"))
    print("=== cmvn (1,400) ===")
    report("cmvn", load_bin(cpp_dir + "/cpp.cmvn.bin"),
           np.load(ref_dir + "/ref_cmvn.npy")[0:1, :])


if __name__ == "__main__":
    main()
