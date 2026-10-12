#!/bin/bash
# ============================================================================
# convert_vad.sh —— 把 fsmn_vad 官方 ONNX 转成 **静态 T=20** 的 BM1684X F16 bmodel
#
# 为什么是静态 T=20：
#   BM1684X 上以 bmodel 部署时，VAD 子网用固定 chunk（默认 200ms）推理；
#   本仓 vad_link 内用**特征 FIFO** 攒满 T 行才 launch（T = vad.chunk_ms/10），
#   故 bmodel 直接冻结为静态 T=20，避免 dynamic 形状带来的运行时开销/不一致。
#
# 前置：
#   1) 用官方 dfss（或 tools/dfss_fetch.py）取 fsmn_vad 官方 ONNX（含 model_quant.onnx
#      与 am.mvn）；来源 ModelScope `iic/speech_fsmn_vad_zh-cn-16k-common-onnx`：
#         model_quant.onnx   （ORT 动态量化，含 MatMulInteger，tpu-mlir 不支持）
#         am.mvn              （Kaldi Nnet 文本 CMVN → 本仓 config/audio/am.mvn）
#   2) 反量化：python3 tools/dequant_onnx.py model_quant.onnx fsmn_vad_float.onnx
#
# 用法（在 tpu-mlir 镜像容器内，见 docs/model_conversion.md）：
#   T=20 ONNX=/path/fsmn_vad_float.onnx bash scripts/convert_vad.sh
#
# 环境变量（可由外部覆盖）：
#   T      静态帧数（默认 20 = 200ms / 10ms）
#   ONNX   反量化后的 float ONNX 路径（默认 ./fsmn_vad_float.onnx）
#   OUT    输出 bmodel 路径（默认 ./models/fsmn_vad.bmodel）
#   TPUC   tpu-mlir install 根（默认 /workspace/tpu-mlir/install）
# ============================================================================
set -e

T=${T:-20}
ONNX=${ONNX:-$(pwd)/fsmn_vad_float.onnx}
OUT=${OUT:-$(pwd)/models/fsmn_vad.bmodel}
TPUC=${TPUC:-/workspace/tpu-mlir/install}
WORK=${WORK:-$(pwd)/build/vad_t${T}}
mkdir -p "$WORK" "$(dirname "$OUT")"
cd "$WORK"

# tpu-mlir 环境（install 树；缺 libortools.so.9 时多为 LD_LIBRARY_PATH 未含 install/lib）
export TPUC_ROOT="$TPUC"
export PATH="$TPUC/bin:$TPUC/../python/tools:$TPUC/../python/utils:$PATH"
export PYTHONPATH="$TPUC/python:$TPUC/../python:$PYTHONPATH"
export LD_LIBRARY_PATH="$TPUC/lib:$TPUC/../capi/lib:$LD_LIBRARY_PATH"

echo "==== [1/3] ONNX -> MLIR (静态 T=$T，注意：不传 --dynamic) ===="
model_transform.py \
    --model_name speech_fsmn_vad_T${T} \
    --model_def "$ONNX" \
    --input_shapes [[1,${T},400],[1,128,19,1],[1,128,19,1],[1,128,19,1],[1,128,19,1]] \
    --input_types float32,float32,float32,float32,float32 \
    --output_names logits,out_cache0,out_cache1,out_cache2,out_cache3 \
    --mlir fsmn_vad_T${T}.mlir

echo "==== [2/3] MLIR -> bmodel (F16, BM1684X) ===="
model_deploy.py \
    --mlir fsmn_vad_T${T}.mlir \
    --quantize F16 \
    --processor BM1684X \
    --model fsmn_vad_T${T}_bm1684x_f16.bmodel

echo "==== [3/3] info ===="
model_tool --info fsmn_vad_T${T}_bm1684x_f16.bmodel
cp -f fsmn_vad_T${T}_bm1684x_f16.bmodel "$OUT"
ls -lh "$OUT"
echo "已输出: $OUT"
