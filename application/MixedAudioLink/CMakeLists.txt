cmake_minimum_required(VERSION 3.10)
project(MixedAudioLink CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 交叉编译示例（SoC 模式，在板端或 SDK 容器内）：
#   export SDK=/path/to/sophon-sdk
#   export CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
#   cmake -B build -DSDK="$SDK" && cmake --build build
set(SDK "" CACHE PATH "sophon-sdk 路径（含 bmlib/bmruntime 头与库）；也可经环境变量 SDK 传入")
if(NOT SDK AND DEFINED ENV{SDK})
  set(SDK "$ENV{SDK}" CACHE PATH "sophon-sdk 路径（来自环境变量）" FORCE)
endif()

find_library(BMLIB_LIB NAMES bmlib HINTS ${SDK}/lib)
# 库文件实际名为 libbmrt.so（头文件是 bmruntime.h），故同时探测 bmrt / bmruntime
find_library(BMRUNTIME_LIB NAMES bmrt bmruntime HINTS ${SDK}/lib)
if(NOT BMLIB_LIB OR NOT BMRUNTIME_LIB)
  message(FATAL_ERROR "bmlib/bmruntime not found: set -DSDK=/path/to/sophon-sdk "
                      "(need SDK >= 23.x for addr_mode / bm_mem_get_device_addr)")
endif()

# 版本注入（可选）：默认 "unknown" 占位，构建时可覆盖注入，如
#   cmake -B build -DAL_GIT_HASH=$(git rev-parse --short HEAD) -DAL_BUILD_DATE=$(date +%F)
# 为避免引入 execute_process 对 git 的构建期依赖，此处不做自动探测
set(AL_GIT_HASH "unknown" CACHE STRING "git hash（注入 AudioLinkGetVersion）")
set(AL_BUILD_DATE "unknown" CACHE STRING "构建日期（注入 AudioLinkGetVersion）")

add_library(audiolink STATIC
  cpp/audio_link.cpp
  cpp/meeting_summary.cpp
  cpp/link_mng.cpp
  cpp/audio_link_mng.cpp
  cpp/model_hub.cpp
  cpp/links/vad_link.cpp
  cpp/links/asr_link.cpp
)
target_compile_definitions(audiolink PRIVATE
  AL_GIT_HASH="${AL_GIT_HASH}"
  AL_BUILD_DATE="${AL_BUILD_DATE}"
)
target_include_directories(audiolink PUBLIC
  cpp
  ${SDK}/include
)
target_link_libraries(audiolink PUBLIC ${BMLIB_LIB} ${BMRUNTIME_LIB} pthread)

add_executable(audiolink_demo cpp/main.cpp)
target_link_libraries(audiolink_demo PRIVATE audiolink)

# 纯 C 调用示例（验证 C ABI 可被 C 程序直接链接）
add_executable(sample_c examples/sample_c.c)
target_link_libraries(sample_c PRIVATE audiolink)
