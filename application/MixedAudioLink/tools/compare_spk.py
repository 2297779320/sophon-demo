#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_spk.py - CAM++ 声纹前端对拍：本仓 al_feature.h（povey/high_freq=0）
                      vs 官方 sample/CAM++/cpp/dependencies 前端。

用法:
    python3 tools/compare_spk.py <our_prefix> <ref_prefix>

对比:
    <p>.fbank(.bin/.dim)  vs <r>.fbank   -> 允许常量偏移（int16 vs [-1,1] 量纲差 log(32767)）
    <p>.mean (.bin/.dim)  vs <r>.mean    -> 去整句均值后应一致（|max| < 1e-3）

生成:
    our: g++ -O2 -std=c++17 -I cpp tools/spk_feat_dump.cpp -o /tmp/spk_feat_dump
         /tmp/spk_feat_dump test16k.wav /tmp/spk
    ref: 见 tools/spk_ref_dump.cpp 头部（链接 CAM++ dependencies 源码）
         /tmp/spk_ref_dump test16k.wav /tmp/spkref
"""
import sys

import numpy as np


def load_bin(p):
    with open(p + ".bin.dim") as f:
        r, c = [int(x) for x in f.read().split()]
    return np.fromfile(p + ".bin", dtype=np.float32).reshape(r, c)


def main():
    our_prefix, ref_prefix = sys.argv[1], sys.argv[2]
    ok = True

    a = load_bin(our_prefix + ".fbank")
    b = load_bin(ref_prefix + ".fbank")
    print("=== raw log-mel fbank ===")
    if a.shape != b.shape:
        print("  FAIL 形状 %s vs %s" % (a.shape, b.shape))
        ok = False
    else:
        d = a - b
        print("  our  mean %+.5f  min %+.5f  max %+.5f" % (a.mean(), a.min(), a.max()))
        print("  ref  mean %+.5f  min %+.5f  max %+.5f" % (b.mean(), b.min(), b.max()))
        print("  diff mean %+.6f  std %.6f  min %+.6f  max %+.6f  |max| %.6f"
              % (d.mean(), d.std(), d.min(), d.max(), np.abs(d).max()))
        if np.abs(d).max() < 1e-3:
            print("  => 逐点一致 OK（|max| < 1e-3）")
        elif d.std() < 1e-3:
            # 允许纯常量偏移（int16 vs [-1,1] 量纲差，期望 ≈ 2*log(32767) = 20.79）
            print("  => 纯常量偏移 %.4f（期望 2*log(32767)=%.4f）— OK"
                  % (d.mean(), 2.0 * np.log(32767.0)))
        else:
            print("  => FAIL 逐点差异过大")
            ok = False

    a = load_bin(our_prefix + ".mean")
    b = load_bin(ref_prefix + ".mean")
    print("=== mean-subtracted feature（spk_link 实际输入）===")
    if a.shape != b.shape:
        print("  FAIL 形状 %s vs %s" % (a.shape, b.shape))
        ok = False
    else:
        d = a - b
        mx = float(np.abs(d).max())
        print("  our  mean %+.5f  min %+.5f  max %+.5f" % (a.mean(), a.min(), a.max()))
        print("  ref  mean %+.5f  min %+.5f  max %+.5f" % (b.mean(), b.min(), b.max()))
        print("  diff mean %+.6f  std %.6f  |max| %.6f" % (d.mean(), d.std(), mx))
        print("  => %s（|max| < 1e-3）" % ("一致 OK" if mx < 1e-3 else "FAIL"))
        if mx >= 1e-3:
            ok = False

    print()
    print("结论：" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
