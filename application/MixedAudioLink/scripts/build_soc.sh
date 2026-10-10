#!/bin/bash
# 板端/容器内构建脚本（SoC 模式交叉编译示例）
# 用法：SDK=/path/to/sophon-sdk ./scripts/build_soc.sh
set -e
cd "$(dirname "$0")/.."
if [ -z "$SDK" ]; then
  echo "ERROR: SDK env not set (export SDK=/path/to/sophon-sdk, need >= 23.x)" >&2
  exit 1
fi
if [ ! -d "$SDK/include" ] || [ ! -d "$SDK/lib" ]; then
  echo "ERROR: $SDK missing include/ or lib/" >&2
  exit 1
fi
export CC="${CC:-aarch64-linux-gnu-gcc}"
export CXX="${CXX:-aarch64-linux-gnu-g++}"
cmake -B build -DSDK="$SDK"
cmake --build build -j
echo "done: build/libaudiolink.a build/audiolink_demo build/sample_c"
