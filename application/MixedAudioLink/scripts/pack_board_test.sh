#!/bin/bash
# ============================================================================
# pack_board_test.sh —— 打包「板端实测」所需的全部内容成一个自包含目录
#
# ★ 必须在 Linux 环境执行（板端是 aarch64 SoC）：
#   - 在 Windows/MSYS 上直接跑本脚本时，会自动转交容器（默认 soc-cross）内执行，
#     避免 NTFS 丢执行位、tar 内 mode 不对。
#   - 容器内路径映射可用 HOST_MOUNT / GUEST_MOUNT 覆盖（默认 /d/sophgo ↔ /workspace）。
#
# 产出结构（dist/board_test_<时间戳>/）：
#   bin/            aarch64 可执行（audiolink_demo、sample_c）
#   config/         audio/ 与 summary/（vocab/merges/mel_filters.npy/am.mvn/prompt）
#   models/         bmodel（all / small / none 三档）
#   test/           测试音频 test16k.wav（+ --audio 追加）+ meeting_sample.txt
#   run_board_test.sh   板端驱动脚本（自检 + 4 场景 + 落日志 + 汇总）
#   README.md       这个包是什么 / 板端前置条件 / 怎么跑 / 怎么读日志
#   MANIFEST.md5    逐文件 md5（md5sum -c 可直接校验）
#   lib/            可选（--with-libs）：aarch64 运行库。**默认不打**——
#                   板端通常已自带 libsophon（libbmrt/libbmlib），无需重复携带。
#
# 用法（主机或容器内均可，主机上会自动进容器）：
#   bash scripts/pack_board_test.sh                  # 含全部 4 个 bmodel（≈8.8GB）
#   bash scripts/pack_board_test.sh --small-models   # 只含 vad+spk（≈37MB）
#   bash scripts/pack_board_test.sh --no-models      # 只出骨架（模型已在板端时）
#   bash scripts/pack_board_test.sh --link           # 直接跑在容器内时可用硬链接省磁盘
#   bash scripts/pack_board_test.sh --tar            # 额外打 .tar（不压缩，快）
#   bash scripts/pack_board_test.sh --tar-gz         # 额外打 .tar.gz（压缩，慢）
#   bash scripts/pack_board_test.sh --with-libs      # 额外携带 SDK 运行库（板端缺库时用）
#   bash scripts/pack_board_test.sh --sdk DIR        # 指定 SDK 根（默认 \$SDK 或 /workspace/soc-sdk）
#   bash scripts/pack_board_test.sh --bin DIR --models DIR --out DIR
# ============================================================================
set -e

# ---------------------------------------------------------------------------
# 平台自检：非 Linux → 转交容器内执行（板端目标为 aarch64）
# ---------------------------------------------------------------------------
if [ "$(uname -s)" != "Linux" ]; then
    HOST_MOUNT="${HOST_MOUNT:-/d/sophgo}"
    GUEST_MOUNT="${GUEST_MOUNT:-/workspace}"
    CONTAINER="${CONTAINER:-soc-cross}"
    ROOT_H=$(cd "$(dirname "$0")/.." && pwd)
    case "$ROOT_H" in
        "$HOST_MOUNT"/*) CROOT="$GUEST_MOUNT/${ROOT_H#"$HOST_MOUNT"/}" ;;
        *)
            echo "[ERR] 仓库不在容器挂载范围内：$ROOT_H" >&2
            echo "      请设置 HOST_MOUNT（主机挂载点）与 GUEST_MOUNT（容器内路径）后重试" >&2
            echo "      当前：HOST_MOUNT=$HOST_MOUNT  GUEST_MOUNT=$GUEST_MOUNT" >&2
            exit 2 ;;
    esac
    echo "[i] 当前为 $(uname -s)（非 Linux）→ 转交容器 $CONTAINER 内打包（aarch64 目标）"
    echo "    主机 $ROOT_H  →  容器 $CROOT"
    # 时间戳在主机侧生成（容器 TZ 常为 UTC，会让目录名对不上本地时间）
    MSYS_NO_PATHCONV=1 docker exec -e STAMP="$(date +%Y%m%d-%H%M%S)" "$CONTAINER" bash -lc \
        "test -d '$CROOT' || { echo '[ERR] 容器内不存在 $CROOT'; exit 2; }; cd '$CROOT' && bash scripts/pack_board_test.sh $(printf '%q ' "$@")"
    exit $?
fi

# ---------------------------------------------------------------------------
# 以下均在 Linux（容器/板端）内执行
# ---------------------------------------------------------------------------
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

BIN_DIR="$ROOT/build"
MODELS_DIR="$ROOT/models"
OUT_BASE="$ROOT/dist"
SDK="${SDK:-/workspace/soc-sdk}"
MODE="all"           # all | small | none
WITH_LIBS=0          # 默认不打包运行库（板端一般自带 libsophon）
DO_TAR=""
USE_LINK=0
NO_MD5=0
EXTRA_AUDIO=()

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bin)          BIN_DIR="$2"; shift 2 ;;
        --models)       MODELS_DIR="$2"; shift 2 ;;
        --out)          OUT_BASE="$2"; shift 2 ;;
        --sdk)          SDK="$2"; shift 2 ;;
        --no-models)    MODE="none"; shift ;;
        --small-models) MODE="small"; shift ;;
        --all-models)   MODE="all"; shift ;;
        --no-libs)      WITH_LIBS=0; shift ;;
        --with-libs)    WITH_LIBS=1; shift ;;
        --audio)        EXTRA_AUDIO+=("$2"); shift 2 ;;
        --link)         USE_LINK=1; shift ;;
        --tar)          DO_TAR="tar"; shift ;;
        --tar-gz)       DO_TAR="targz"; shift ;;
        --no-md5)       NO_MD5=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "[ERR] 未知参数: $1" >&2; usage; exit 2 ;;
    esac
done

if [ "$USE_LINK" = "1" ] && [ -n "$DO_TAR" ]; then
    echo "[ERR] --link 与 --tar 互斥（tar 会使硬链接实体化）" >&2; exit 2
fi

# ---- 前置校验：必须有已编译好的 aarch64 产物 ----
for b in audiolink_demo sample_c; do
    if [ ! -f "$BIN_DIR/$b" ]; then
        echo "[ERR] 缺少可执行文件: $BIN_DIR/$b" >&2
        echo "      请先在 soc-cross（或板端 SDK 容器）内构建：" >&2
        echo "      SDK=\$SDK ./scripts/build_soc.sh" >&2
        exit 1
    fi
done
for b in audiolink_demo sample_c; do
    M=$(readelf -h "$BIN_DIR/$b" 2>/dev/null | awk '/Machine:/{print $2}')
    [ "$M" = "AArch64" ] || echo "[warn] $b 架构为 '$M'，非 AArch64（板端可能跑不了）" >&2
done

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
NAME="board_test_$STAMP"
STAGE="$OUT_BASE/$NAME"
mkdir -p "$STAGE"/{bin,config,test,models}
[ "$WITH_LIBS" = "1" ] && mkdir -p "$STAGE/lib"
echo "==== 组装 $STAGE ===="
echo "  运行环境: $(uname -s) $(uname -m)"

# ---- 复制助手 ----
put() { # put <src> <dst>
    mkdir -p "$(dirname "$2")"
    if [ "$USE_LINK" = "1" ]; then
        if ! cp -l "$1" "$2" 2>/dev/null; then
            [ -z "${_LINK_WARNED:-}" ] && { echo "  [i] 硬链接不可用，改用复制" >&2; _LINK_WARNED=1; }
            cp -f "$1" "$2"
        fi
    else
        cp -f "$1" "$2"
    fi
}

# ---- 1. 可执行（Linux 下 chmod 生效，tar/拷贝都保留 755） ----
put "$BIN_DIR/audiolink_demo" "$STAGE/bin/audiolink_demo"
put "$BIN_DIR/sample_c"       "$STAGE/bin/sample_c"
chmod 755 "$STAGE/bin/audiolink_demo" "$STAGE/bin/sample_c"
echo "  + bin/audiolink_demo, bin/sample_c（$(du -sh "$STAGE/bin" | cut -f1)）"

# ---- 2. aarch64 运行库（二进制的 RUNPATH 指向构建机，板端必须靠 lib/ 兜底） ----
copy_libs() {
    [ "$WITH_LIBS" = "1" ] || { echo "  (运行库不打：板端自带 libsophon；如需请加 --with-libs)"; return 0; }
    local want=(libbmrt.so.1.0 libbmlib.so.0) l miss=()
    for l in "${want[@]}"; do
        if [ -f "$SDK/lib/$l" ]; then
            put "$SDK/lib/$l" "$STAGE/lib/$l"
            chmod 755 "$STAGE/lib/$l"
        else
            miss+=("$l")
        fi
    done
    # 便捷软链（ldd/手工链接时用得上；tar 会保留软链）
    ( cd "$STAGE/lib" && [ -f libbmrt.so.1.0 ] && ln -sf libbmrt.so.1.0 libbmrt.so
      [ -f libbmlib.so.0 ] && ln -sf libbmlib.so.0 libbmlib.so ) 2>/dev/null || true
    if [ ${#miss[@]} -gt 0 ]; then
        echo "  [warn] $SDK/lib 下缺：${miss[*]}（板端需自行保证系统内有对应运行库）" >&2
    fi
    [ -n "$(ls -A "$STAGE/lib" 2>/dev/null)" ] && echo "  + lib/ $(ls "$STAGE/lib" | tr '\n' ' ')"
}
copy_libs

# ---- 3. 配置与资产 ----
cp -rf "$ROOT/config/." "$STAGE/config/"
echo "  + config/ ($(find "$STAGE/config" -type f | wc -l | tr -d ' ') 个文件)"

# vad_config.yaml（参考件，若在 models/ 下则一并带上）
[ -f "$MODELS_DIR/vad_config.yaml" ] && put "$MODELS_DIR/vad_config.yaml" "$STAGE/models/vad_config.yaml"

# ---- 4. 模型 ----
copy_models() {
    local list=()
    case "$MODE" in
        all)   list=(fsmn_vad.bmodel campplus.bmodel qwen3_asr.bmodel qwen1.5-7b_int4_6k_1dev.bmodel) ;;
        small) list=(fsmn_vad.bmodel campplus.bmodel) ;;
        none)  return 0 ;;
    esac
    local m miss=()
    for m in "${list[@]}"; do
        if [ -f "$MODELS_DIR/$m" ]; then
            put "$MODELS_DIR/$m" "$STAGE/models/$m"
            printf "  + models/%s (%s)\n" "$m" "$(du -h "$MODELS_DIR/$m" | cut -f1)"
        else
            miss+=("$m")
        fi
    done
    if [ ${#miss[@]} -gt 0 ]; then
        echo "  [warn] 源目录缺以下模型，已跳过：${miss[*]}" >&2
        printf '%s\n' "${miss[@]}" > "$STAGE/models/MISSING.txt"
    fi
}
copy_models

# ---- 5. 测试素材 ----
if [ -f "$ROOT/test16k.wav" ]; then
    put "$ROOT/test16k.wav" "$STAGE/test/test16k.wav"
    echo "  + test/test16k.wav"
else
    echo "  [warn] 无 test16k.wav（可 bash scripts/download_model.sh datasets 取一段）" >&2
fi
for f in "${EXTRA_AUDIO[@]}"; do
    [ -f "$f" ] || { echo "  [warn] --audio 文件不存在: $f" >&2; continue; }
    put "$f" "$STAGE/test/$(basename "$f")"
    echo "  + test/$(basename "$f")"
done

cat > "$STAGE/test/meeting_sample.txt" <<'EOT'
讲话人1：大家好，今天这个会主要对一下这个季度的项目进展，还有下个阶段的分工。
讲话人2：好的，我先说一下算法这条线。上个月我们把语音切分的模型从动态改成静态了，板子上的延迟降了大概三分之一。
讲话人1：那挺好。识别这条线呢？
讲话人2：识别现在用的是 1.7B 的模型，精度够，但是显存占用比较大，跟后边的总结模型不能同时驻留。我们做了一个换入换出的流程，总结之前先把识别模型卸掉。
讲话人1：这个流程要写进文档，别人接手的时候容易踩坑。
讲话人3：接口这边我补充一下。对外的 C 接口已经稳定了，回调里面直接给带讲话人归属的整句文本。下周我把示例程序补一个纯 C 的版本。
讲话人1：行。那行动项就是：第一，换入换出流程补文档，负责人讲话人2，下周三之前；第二，纯 C 示例程序，讲话人3，下周五之前；第三，下个季度我们再看要不要上更小的模型。今天就先到这。
EOT
echo "  + test/meeting_sample.txt"

# ---- 6. 板端驱动脚本 ----
cat > "$STAGE/run_board_test.sh" <<'EOS'
#!/bin/bash
# ============================================================================
# 板端实测驱动脚本（在板子上、本目录内执行）
#
#   bash run_board_test.sh [devid]         # devid 默认 0
#
# 依次跑 4 个场景，逐场景落日志到 logs/<时间戳>/，并打印汇总。
# 场景：
#   01_demo_smoke      audiolink_demo：VAD切段→ASR+CAM++→转写→总结（完整时序）
#   02_sample_c        sample_c：同上的 C ABI 版本（音频流 + 总结）
#   03_sample_c_swap   sample_c --swap：显存受限换入换出（总结前卸 asr，总结后重载）
#   04_summary_only    sample_c -t：不跑音频流，只对文本文件做总结
# ============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
DEVID="${1:-0}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/$STAMP"
mkdir -p "$LOG"

# 可执行位兜底（经 Windows 中转时可能丢）
chmod +x ./bin/* ./run_board_test.sh 2>/dev/null || true

say() { echo; echo "================ $* ================"; echo; }
has_bmsmi() { command -v bm-smi >/dev/null 2>&1; }

say "环境自检"
echo "目录      : $HERE"
echo "内核      : $(uname -a)"
echo "设备号    : $DEVID"

# 运行库：二进制的 RUNPATH 是构建机路径（板端无效）。
# 默认用板端系统自带的 libsophon；若包内带了 lib/（--with-libs）则优先用包内的。
if [ -z "${NO_BUNDLED_LIBS:-}" ] && [ -e lib/libbmrt.so.1.0 ]; then
    export LD_LIBRARY_PATH="$HERE/lib:${LD_LIBRARY_PATH:-}"
    echo "运行库    : 包内 lib/（LD_LIBRARY_PATH 已前置）"
else
    echo "运行库    : 板端系统库（缺 libbmrt/libbmlib 时检查 libsophon，或用 --with-libs 重打包）"
fi

[ -f ./bin/audiolink_demo ] && echo "audiolink_demo : OK" || echo "audiolink_demo : MISSING(!)"
[ -f ./bin/sample_c ]       && echo "sample_c       : OK" || echo "sample_c       : MISSING(!)"
if command -v ldd >/dev/null 2>&1 && [ -f ./bin/audiolink_demo ]; then
    MISS=$(ldd ./bin/audiolink_demo 2>/dev/null | awk '/not found/{print $1}')
    [ -n "$MISS" ] && echo "缺库(!)   : $MISS" || echo "动态依赖  : 全部可解析"
fi
echo "模型:"
for m in models/*.bmodel; do [ -e "$m" ] && ls -lh "$m" | awk '{print "  "$5"\t"$9}'; done
[ -e models/MISSING.txt ] && { echo "缺失模型（板端需自行补齐）:"; sed 's/^/  - /' models/MISSING.txt; }
echo "音频:"
for w in test/*.wav; do [ -e "$w" ] && ls -lh "$w" | awk '{print "  "$5"\t"$9}'; done
if has_bmsmi; then bm-smi > "$LOG/bm-smi-before.txt" 2>&1; echo "显存基线  : $LOG/bm-smi-before.txt"; else echo "bm-smi 不存在（跳过显存记录）"; fi

run() { # run <场景名> <命令...>
    local name="$1"; shift
    say "$name"
    echo "\$ $*" > "$LOG/$name.log"
    "$@" 2>&1 | tee -a "$LOG/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "[exit=$rc]  <- 日志: $LOG/$name.log" | tee -a "$LOG/$name.log"
    return $rc
}

FAIL=0
[ -f ./bin/audiolink_demo ] && { run 01_demo_smoke    ./bin/audiolink_demo -c config/audio/audio_link.json -d "$DEVID" -i test/test16k.wav || FAIL=$((FAIL+1)); }
[ -f ./bin/sample_c ] && { run 02_sample_c      ./bin/sample_c -c config/audio/audio_link.json -s config/summary/meeting_summary.json -d "$DEVID" -i test/test16k.wav || FAIL=$((FAIL+1)); }
[ -f ./bin/sample_c ] && { run 03_sample_c_swap ./bin/sample_c --swap -c config/audio/audio_link.json -d "$DEVID" -i test/test16k.wav || FAIL=$((FAIL+1)); }
[ -f ./bin/sample_c ] && { run 04_summary_only  ./bin/sample_c -s config/summary/meeting_summary.json -d "$DEVID" -t test/meeting_sample.txt || FAIL=$((FAIL+1)); }

has_bmsmi && { bm-smi > "$LOG/bm-smi-after.txt" 2>&1; }

say "汇总"
echo "日志目录 : $LOG"
echo "失败场景 : $FAIL"
if grep -lEi "failed|error|segmentation|core dumped|E_AL_" "$LOG"/*.log >/dev/null 2>&1; then
    echo "疑似异常日志："
    grep -lEi "failed|error|segmentation|core dumped|E_AL_" "$LOG"/*.log | sed 's/^/  [!] /'
else
    echo "未在日志中发现异常关键字"
fi
echo
echo "人工核对要点："
echo "  1) 01/02 是否输出中文转写行，且每行形如 [start-end ms] 讲话人N：xxx"
echo "  2) 01/02 是否输出 '==== summary ====' 且有中文纪要"
echo "  3) 03 的 unload/reload asr 是否成功（显存受限设备关键路径）"
echo "  4) 对比 bm-smi-before / after，确认退出后显存是否释放"
exit $FAIL
EOS
chmod 755 "$STAGE/run_board_test.sh"
echo "  + run_board_test.sh"

# ---- 7. 包内说明 ----
MODE_DESC=$([ "$MODE" = all ] && echo "全部 4 个 bmodel" || { [ "$MODE" = small ] && echo "小档：仅 fsmn_vad + campplus" || echo "无模型（骨架）"; })
if [ "$WITH_LIBS" = "1" ]; then
    LIB_TREE='lib/                 aarch64 运行库：libbmrt.so.1.0、libbmlib.so.0（+ 软链）'
    LIB_PREREQ='- 板端 libsophon 需能与包内 `lib/libbmrt.so.1.0` 匹配；驱动脚本会把 `lib/` 前置到 `LD_LIBRARY_PATH`（不想要可 `NO_BUNDLED_LIBS=1` 走系统库）'
    LIB_RUN='export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"   # 包内运行库优先'
    LIB_NOTE='- 包内已带 aarch64 运行库，驱动脚本会把 `lib/` 前置到 `LD_LIBRARY_PATH`。'
else
    LIB_TREE=''
    LIB_PREREQ='- 板端需自带 libsophon（`libbmrt.so.1.0` / `libbmlib.so.0`）；本包**不带运行库**（板端缺库时可加 `--with-libs` 重打包）'
    LIB_RUN='# 板端已内置 libsophon，通常无需设置 LD_LIBRARY_PATH'
    LIB_NOTE='- 二进制的 `RUNPATH` 指向构建机路径（板端无效），运行时由**板端系统库**提供 `libbmrt.so.1.0` / `libbmlib.so.0`。'
fi
cat > "$STAGE/README.md" <<EOT
# MixedAudioLink 板端实测包

生成时间：$STAMP
生成环境：$(uname -s) $(uname -m)  （aarch64 交叉产物）
模型档位：$MODE（$MODE_DESC）

## 包内容

\`\`\`
bin/                 aarch64 可执行：audiolink_demo、sample_c
$LIB_TREE
config/audio/        audio_link.json + vocab/merges/mel_filters.npy/am.mvn
config/summary/      meeting_summary.json + vocab/merges/prompt
models/              bmodel（档位 = $MODE）
test/                test16k.wav（+ 追加音频）、meeting_sample.txt
run_board_test.sh    板端驱动脚本
MANIFEST.md5         逐文件 md5（md5sum -c 校验）
\`\`\`

## 板端前置条件

- 芯片：BM1684X（BM1688 需重转 bmodel）
$LIB_PREREQ
- 建议安装 \`bm-smi\` 以便记录显存
- 音频须 16k / 单声道 / PCM16

## 怎么跑

\`\`\`bash
# 若从 Windows 中转过来丢了执行位，先补一次：
chmod +x bin/* run_board_test.sh

# 在本包目录内执行（务必从本目录启动：models 路径以 CWD 为基准）
bash run_board_test.sh 0        # 0 = 设备号
\`\`\`

单场景手跑：

\`\`\`bash
$LIB_RUN
./bin/audiolink_demo -c config/audio/audio_link.json -d 0 -i test/test16k.wav
./bin/sample_c       -c config/audio/audio_link.json -s config/summary/meeting_summary.json -d 0 -i test/test16k.wav
./bin/sample_c --swap -c config/audio/audio_link.json -d 0 -i test/test16k.wav
./bin/sample_c -s config/summary/meeting_summary.json -d 0 -t test/meeting_sample.txt
\`\`\`

## 完整性校验

\`\`\`bash
md5sum -c MANIFEST.md5
\`\`\`

## 显存提示

- asr（Qwen3-ASR-1.7B，3.81GB）与 summary（Qwen1.5-7B int4，4.94GB）**不能共存**。
- 完整时序（01/02）会自动「总结前卸 asr、总结后重载」；\`--swap\`（03）是这条时序的显式版本。
- 若显存不足，改用 \`--small-models\` 包并跳过 01/02 的总结段，或先只验 VAD+声纹链路。

## 路径基准（易踩坑）

- \`models.*.path\` 相对 **应用根（启动 CWD）** 解析 → 所以要在本包目录内启动。
- \`vocab/merges/mel_filters/cmvn_file/prompt_file\` 相对 **启动 JSON 所在目录** 解析。
$LIB_NOTE
EOT
echo "  + README.md"

# ---- 8. MANIFEST（Linux md5sum，天然 LF） ----
if [ "$NO_MD5" = "1" ]; then
    echo "  (--no-md5：跳过校验清单)"
else
    echo "  生成 MANIFEST.md5 ..."
    ( cd "$STAGE" && find . -type f ! -name MANIFEST.md5 | sed 's|^\./||' | LC_ALL=C sort \
        | while IFS= read -r p; do md5sum "$p"; done > MANIFEST.md5 )
    echo "  + MANIFEST.md5 ($(wc -l < "$STAGE/MANIFEST.md5" | tr -d ' ') 条)"
fi

# ---- 9. 体积汇总与打包 ----
echo
echo "==== 完成：$STAGE（合计 $(du -sh "$STAGE" | cut -f1)） ===="
for d in bin lib config models test; do
    [ -d "$STAGE/$d" ] && printf "  %-8s %s\n" "$d/" "$(du -sh "$STAGE/$d" | cut -f1)"
done
if [ -n "$DO_TAR" ]; then
    cd "$OUT_BASE"
    if [ "$DO_TAR" = "tar" ]; then
        echo "打包 $NAME.tar ..."
        tar -cf "$NAME.tar" "$NAME"
        ls -lh "$NAME.tar" | awk '{print "  -> "$5"  "$9}'
    else
        echo "打包 $NAME.tar.gz（压缩，耗时取决于模型大小）..."
        tar -czf "$NAME.tar.gz" "$NAME"
        ls -lh "$NAME.tar.gz" | awk '{print "  -> "$5"  "$9}'
    fi
fi
echo
echo "下一步：整目录（或 .tar）拷到板子，在包内执行  bash run_board_test.sh 0"
