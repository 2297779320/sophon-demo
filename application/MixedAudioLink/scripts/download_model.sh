#!/bin/bash
# ============================================================================
# download_model.sh —— 拉取 MixedAudioLink 运行所需的官方 bmodel 与配套资产
#
# 拉取清单（全部来自 Sophgo 官方源）：
#   models/qwen3_asr.bmodel                  Qwen3-ASR-1.7B bf16 seq512 BM1684X 1dev（3.81GB）
#   models/campplus.bmodel                   CAM++ FP32 dynamic 1b（32MB，官方导出产物）
#   models/qwen1.5-7b_int4_6k_1dev.bmodel    Qwen1.5-7B int4 6k BM1684X 1dev（4.94GB，会议总结）
#   models/vad_config.yaml                   fsmn_vad funasr 运行时配置（参考）
#   config/audio/{vocab.json,merges.txt}     Qwen 分词器（= token_config.zip 内容）
#   config/summary/{summary_vocab.json,summary_merges.txt}
#
# **vad bmodel 不在本脚本内**：它是本仓用官方 ONNX 自转的静态 T=20，
# 见 scripts/convert_vad.sh 与 docs/model_conversion.md §3。
#
# 用法：
#   bash scripts/download_model.sh            # 全部
#   bash scripts/download_model.sh models     # 只拉 bmodel
#   bash scripts/download_model.sh assets     # 只拉 tokenizer 资产
#
# 下载器：优先 `python3 -m dfss`（官方）；若 dfss 报 "No available server found"
# （其内置测速在部分网络下失效），本仓自带 tools/dfss_fetch.py 直连 SFTP 兜底。
# ============================================================================
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MIRROR=${MIRROR:-open@sophgo.com}
WHAT=${1:-all}

DL_DIR="$ROOT/build/_dl"
mkdir -p "$ROOT/models" "$ROOT/config/audio" "$ROOT/config/summary" "$DL_DIR"

python3 -c "import dfss" 2>/dev/null && HAS_DFSS=1 || HAS_DFSS=0

# fetch <远端路径> <本地文件>
fetch() {
    local remote="$1" local="$2"
    if [ -f "$local" ] && [ -s "$local" ]; then
        echo "[skip] $local 已存在"
        return 0
    fi
    echo "[get ] $MIRROR:$remote -> $local"
    if [ "$HAS_DFSS" = "1" ]; then
        if python3 -m dfss --url="$MIRROR:$remote" --dst_dir="$(dirname "$local")" ; then
            return 0
        fi
        echo "[warn] dfss 失败，改用 tools/dfss_fetch.py（直连 SFTP）"
    fi
    if [ -f "$ROOT/tools/dfss_fetch.py" ] && python3 -c "import paramiko" 2>/dev/null; then
        python3 "$ROOT/tools/dfss_fetch.py" get "/$remote" "$local"
    else
        echo "[ERR ] 需要 dfss 或 paramiko 之一：pip3 install dfss  (或)  pip3 install paramiko" >&2
        return 1
    fi
}

dl_models() {
    fetch "/ext_model_information/LLM/LLM-TPU/qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_20260408_094600.bmodel" \
          "$DL_DIR/qwen3_asr.bmodel"
    [ -s "$ROOT/models/qwen3_asr.bmodel" ] || cp -f "$DL_DIR/qwen3_asr.bmodel" "$ROOT/models/qwen3_asr.bmodel"

    fetch "/sophon-demo/Qwen1_5/qwen1.5-7b_int4_6k_1dev.bmodel" \
          "$ROOT/models/qwen1.5-7b_int4_6k_1dev.bmodel"

    # CAM++ 官方模型包（内部含 BM1684X/campplus_bm1684x_fp32_1b.bmodel）
    if [ ! -s "$ROOT/models/campplus.bmodel" ]; then
        fetch "/sophon-demo/campplus/models_1023/models.zip" "$DL_DIR/campplus_models.zip"
        rm -rf "$DL_DIR/campplus_x" && mkdir -p "$DL_DIR/campplus_x"
        unzip -q -o "$DL_DIR/campplus_models.zip" -d "$DL_DIR/campplus_x"
        src=$(find "$DL_DIR/campplus_x" -name "campplus_bm1684x_fp32_1b.bmodel" | head -1)
        [ -n "$src" ] || { echo "[ERR ] 未在 zip 内找到 campplus_bm1684x_fp32_1b.bmodel" >&2; exit 1; }
        cp -f "$src" "$ROOT/models/campplus.bmodel"
    else
        echo "[skip] models/campplus.bmodel 已存在"
    fi
}

dl_assets() {
    fetch "/sophon-demo/Qwen1_5/token_config.zip" "$DL_DIR/token_config.zip"
    rm -rf "$DL_DIR/token_x" && mkdir -p "$DL_DIR/token_x"
    unzip -q -o "$DL_DIR/token_config.zip" -d "$DL_DIR/token_x"
    local v m
    v=$(find "$DL_DIR/token_x" -name "vocab.json" | head -1)
    m=$(find "$DL_DIR/token_x" -name "merges.txt" | head -1)
    [ -n "$v" ] && [ -n "$m" ] || { echo "[ERR ] token_config.zip 内缺 vocab.json/merges.txt" >&2; exit 1; }
    cp -f "$v" "$ROOT/config/audio/vocab.json"
    cp -f "$m" "$ROOT/config/audio/merges.txt"
    cp -f "$v" "$ROOT/config/summary/summary_vocab.json"
    cp -f "$m" "$ROOT/config/summary/summary_merges.txt"
    echo "[ok  ] 分词器已就位（config/audio + config/summary）"
}

# 测试音频（16k/单声道/16bit）；仓库不收录音频，自测前用本步取一段
dl_datasets() {
    fetch "/sophon-demo/campplus/datasets_1023/datasets.zip" "$DL_DIR/datasets.zip"
    rm -rf "$DL_DIR/ds_x" && mkdir -p "$DL_DIR/ds_x"
    unzip -q -o "$DL_DIR/datasets.zip" -d "$DL_DIR/ds_x"
    local w
    w=$(find "$DL_DIR/ds_x" -name "*.wav" | sort | head -1)
    [ -n "$w" ] || { echo "[ERR ] datasets.zip 内无 wav" >&2; exit 1; }
    cp -f "$w" "$ROOT/test16k.wav"
    echo "[ok  ] 测试音频 -> test16k.wav (源: $(basename "$w"))"
}

case "$WHAT" in
    all)      dl_models; dl_assets; dl_datasets ;;
    models)   dl_models ;;
    assets)   dl_assets ;;
    datasets) dl_datasets ;;
    *) echo "用法: bash scripts/download_model.sh [all|models|assets|datasets]" >&2; exit 2 ;;
esac

echo
echo "==== 完成。当前 models/ ===="
ls -lh "$ROOT/models"
echo
echo "提示：vad bmodel 需另行转换 → bash scripts/convert_vad.sh（见 docs/model_conversion.md）"
