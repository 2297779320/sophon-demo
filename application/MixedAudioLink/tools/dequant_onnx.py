#!/usr/bin/env python3
"""把 ONNX Runtime 动态量化(dynamic-quant)的 ONNX 还原成等价 float ONNX。

背景：fsmn_vad 官方 ONNX（ModelScope iic/speech_fsmn_vad_zh-cn-16k-common-onnx
的 `model_quant.onnx`）是 ORT 动态量化产物，含 MatMulInteger / DynamicQuantizeLinear，
tpu-mlir 不支持 → 必须先还原为纯 float 图再转 bmodel。

模式（每处 MatMul）:
    [q]  DynamicQuantizeLinear(x)          -> x_q, x_scale, x_zp
    [m1] Mul(x_scale, W_scale)             -> combined_scale
    [mm] MatMulInteger(x_q, W_q, x_zp, W_zp) -> acc_i32
    [c]  Cast(acc_i32 -> float)            -> acc_f32
    [m2] Mul(acc_f32, combined_scale)      -> y
    [a]  Add(bias, y)                      -> out

还原为:
    [mm'] MatMul(x, W_fp32)                -> y      (W_fp32 = (W_q - W_zp) * W_scale)
    [a']  Add(bias, y)                     -> out

用法:
    python3 tools/dequant_onnx.py <model_quant.onnx> <fsmn_vad_float.onnx>

依赖: numpy, onnx
"""
import sys
from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)
SRC, DST = sys.argv[1], sys.argv[2]

m = onnx.load(SRC)
g = m.graph

init = {i.name: i for i in g.initializer}
producer = {}                       # tensor -> node index
for idx, n in enumerate(g.node):
    for o in n.output:
        producer[o] = idx

nodes = list(g.node)


def np_of(name):
    return numpy_helper.to_array(init[name])


del_idx = set()
new_nodes = {}                      # idx -> replacement node
added_inits = []                    # 新增的 float 权重 initializer

mm_ids = [i for i, n in enumerate(nodes) if n.op_type == "MatMulInteger"]
print(f"MatMulInteger 数量: {len(mm_ids)}")

for i in mm_ids:
    n = nodes[i]
    x_q, w_q_name, x_zp, w_zp_name = n.input[0], n.input[1], n.input[2], n.input[3]
    acc_q = n.output[0]

    # q: x_q 的生产者 DynamicQuantizeLinear -> 其输入 x (float)
    q = nodes[producer[x_q]]
    assert q.op_type == "DynamicQuantizeLinear", q.op_type
    x_float = q.input[0]

    # c: Cast(acc_q -> float)
    cast_id = next(j for j, nn in enumerate(nodes)
                   if nn.op_type == "Cast" and acc_q in nn.input)
    acc_f = nodes[cast_id].output[0]

    # m2: Mul(acc_f, combined_scale) -> y
    m2_id = next(j for j, nn in enumerate(nodes)
                 if nn.op_type == "Mul" and acc_f in nn.input)
    y = nodes[m2_id].output[0]

    # m1: combined_scale 的生产者 Mul(x_scale, W_scale)
    combined = [t for t in nodes[m2_id].input if t != acc_f][0]
    m1_id = producer[combined]
    m1 = nodes[m1_id]
    # W_scale = m1 输入中"非 DynQuant 输出"的那个（即静态 initializer）
    w_scale_name = next(t for t in m1.input if t not in producer or
                        nodes[producer[t]].op_type != "DynamicQuantizeLinear")
    assert w_scale_name in init, w_scale_name

    Wq = np_of(w_q_name).astype(np.float32)
    Wz = np_of(w_zp_name).astype(np.float32) if w_zp_name in init else np.float32(0)
    Ws = np_of(w_scale_name).astype(np.float32)
    Wf = (Wq - Wz) * Ws

    new_name = w_q_name.replace("_quantized", "_float")
    init[new_name] = numpy_helper.from_array(Wf, new_name)
    added_inits.append(init[new_name])

    new_nodes[i] = helper.make_node("MatMul", [x_float, new_name], [y],
                                    name=n.name.replace("_quant", "_dequant"))
    del_idx.update({i, cast_id, m2_id, m1_id, producer[x_q]})

# 组装新图（保持原顺序：被替换的位置放入新节点，其余原样保留）
out_nodes = []
for i, n in enumerate(nodes):
    if i in del_idx:
        if i in new_nodes:
            out_nodes.append(new_nodes[i])
        continue
    out_nodes.append(n)

# 死代码消除（迭代）
live_check = True
while live_check:
    live_check = False
    used = set()
    for n in out_nodes:
        used.update(n.input)
    used.update(o.name for o in g.output)
    keep = []
    for n in out_nodes:
        if all(o not in used for o in n.output):
            live_check = True
            continue
        keep.append(n)
    out_nodes = keep

g.ClearField("node")
g.node.extend(out_nodes)

# 清理 initializer / value_info 中已不用的项
used_names = set()
for n in out_nodes:
    used_names.update(n.input)
    used_names.update(n.output)
used_names.update(o.name for o in g.output)
used_names.update(i.name for i in g.input)

keep_init = [i for i in list(g.initializer) + added_inits if i.name in used_names]
g.ClearField("initializer")
g.initializer.extend(keep_init)

keep_vi = [v for v in g.value_info if v.name in used_names]
g.ClearField("value_info")
g.value_info.extend(keep_vi)

onnx.checker.check_model(m)
onnx.save(m, DST)

print("新算子统计:", dict(Counter(n.op_type for n in g.node)))
print("新 initializer 数:", len(g.initializer))
print("已保存:", DST)
