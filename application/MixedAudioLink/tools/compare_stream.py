#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_stream.py - 流式前端(StreamFrontend) 对拍：① 同二进制整段基准 ② kaldi/FunASR 基准。

用法:
    python3 tools/compare_stream.py <stream_prefix> <ref_dir> [lfr_n=1]

对比对象:
    ① <stream_prefix>.stream.bin   逐块 push + finish 拼接（流式）
       <stream_prefix>.batch.bin   同一二进制内整段一次性 compute+applyLfr+CMVN
       -> 期望**逐位相等**（|max| == 0）：证明流式与整段在数值上完全等价。
          （必须同二进制：跨二进制会引入 ~1e-6 的浮点代码生成差异）
    ② <ref_dir>/ref_full_lfr_cmvn.npy   kaldi/FunASR 官方前端基准（vad_ref.py 产出）
       -> 期望 |max| < 1e-3（与 compare_feat.py 同容差；残余为跨实现浮点/运算次序差）

用法示例:
    python3 tools/compare_stream.py tools/_out/stream/st_200 tools/_out/ref
"""
import os
import sys

import numpy as np


def load_bin(p):
    with open(p + ".dim") as f:
        r, c = [int(x) for x in f.read().split()]
    return np.fromfile(p, dtype=np.float32).reshape(r, c)


def diff_report(tag, a, b, tol):
    if a.shape != b.shape:
        print("      [%s] FAIL 形状 %s vs %s" % (tag, a.shape, b.shape))
        return False
    n = a.shape[0]
    if n == 0:
        print("      [%s] 空（跳过）" % tag)
        return True
    d = a - b
    amax = float(np.abs(d).max())
    ok = amax <= tol
    print("      [%s] shape=%s  diff mean %+.9f std %.9f |max| %.9g  (tol %.0e)  %s"
          % (tag, a.shape, d.mean(), d.std(), amax, tol, "OK" if ok else "FAIL"))
    if not ok:
        ri, ci = np.unravel_index(int(np.abs(d).argmax()), d.shape)
        print("             worst at row %d col %d : %.6f vs %.6f" % (ri, ci, a[ri, ci], b[ri, ci]))
    return ok


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    prefix, ref_dir = sys.argv[1], sys.argv[2]
    lfr_n = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    st = load_bin(prefix + ".stream.bin")
    print("=== 流式 vs 同二进制整段基准 / vs 官方基准 ===")
    print("      stream: %s" % (st.shape,))

    allok = True

    batch_path = prefix + ".batch.bin"
    if os.path.exists(batch_path):
        ba = load_bin(batch_path)
        m = min(st.shape[0], ba.shape[0])
        allok &= diff_report("stream-vs-batch(same-bin)", st[:m], ba[:m], 0.0)
    else:
        print("      [stream-vs-batch] 跳过（无 %s）" % batch_path)

    ref_path = os.path.join(ref_dir, "ref_full_lfr_cmvn.npy")
    if os.path.exists(ref_path):
        rf = np.load(ref_path)
        m = min(st.shape[0], rf.shape[0])
        allok &= diff_report("stream-vs-ref(kaldi)", st[:m], rf[:m], 1e-3)
        if st.shape[0] != rf.shape[0]:
            print("      [WARN] 行数 stream=%d ref=%d" % (st.shape[0], rf.shape[0]))
    else:
        print("      [stream-vs-ref] 跳过（无 %s）" % ref_path)

    print("      => %s" % ("PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
