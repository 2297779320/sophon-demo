# tools —— 对拍与回归工具

本目录是开发/回归用工具集，**不参与主库构建**（`CMakeLists.txt` 不含它们）。
纯头文件依赖 `cpp/`，可在任意有 g++ 的环境直接编译；需要 numpy 的 Python 脚本
在含 numpy 的环境执行。

## 目录

| 文件 | 用途 |
|---|---|
| `vad_seg_test.cpp` + `vad_seg_ref.py` | VAD 端点状态机切段对拍（C++ vs 官方 Python 参考，bit-exact） |
| `lfr_stream_test.cpp` | LFR 流式 vs 批处理一致性 |
| `feat_dump.cpp` + `compare_feat.py` | VAD 前端 fbank/LFR/CMVN（C++ vs kaldi/FunASR 参考） |
| `stream_frontend_dump.cpp` + `compare_stream.py` | 流式前端 vs 批处理 bit-exact |
| `spk_feat_dump.cpp` / `spk_ref_dump.cpp` + `compare_spk.py` | CAM++ 声纹前端对拍（本仓 vs 官方前端源码） |
| `path_resolve_test.cpp` | 路径双基准解析自测 |
| `dequant_onnx.py` | ORT 量化 ONNX → float ONNX（图手术反量化） |
| `dfss_fetch.py` | 官方源 SFTP 直连下载器（`dfss` 测速失效时的兜底） |
| `vad_ref.py` / `vad_decode_check.py` | VAD 后处理参考实现 / 解码校验 |

## VAD 与前端回归

```bash
cd <应用根>
# 编译（x86 或 aarch64 均可，纯头文件）
g++ -O2 -std=c++17 -I cpp tools/vad_seg_test.cpp          -o /tmp/vad_seg_test
g++ -O2 -std=c++17 -I cpp tools/lfr_stream_test.cpp       -o /tmp/lfr_stream_test
g++ -O2 -std=c++17 -I cpp tools/feat_dump.cpp             -o /tmp/feat_dump
g++ -O2 -std=c++17 -I cpp tools/stream_frontend_dump.cpp  -o /tmp/stream_dump

/tmp/lfr_stream_test                                       # 期望 0 failed
VAD_SEG_BIN=/tmp/vad_seg_test python3 tools/vad_seg_ref.py # 期望 97/97 bit-exact
```

## spk 前端对拍（需官方 CAM++ 前端源码）

```bash
D=<sophon-demo>/sample/CAM++/cpp/dependencies/include
g++ -O2 -std=c++17 -I cpp tools/spk_feat_dump.cpp -o /tmp/spk_feat_dump
g++ -O2 -std=c++17 -I "$D" tools/spk_ref_dump.cpp \
    "$D"/feature/feature_basic.cpp "$D"/feature/feature_common.cpp \
    "$D"/feature/feature_fbank.cpp "$D"/feature/feature_functions.cpp \
    "$D"/utils/wav_reader.cpp -o /tmp/spk_ref_dump
/tmp/spk_feat_dump test16k.wav /tmp/spk_out/our
/tmp/spk_ref_dump  test16k.wav /tmp/spk_out/ref
python3 tools/compare_spk.py /tmp/spk_out/our /tmp/spk_out/ref   # 期望 PASS
```

## 路径解析自测

```bash
g++ -O2 -std=c++17 -I cpp tools/path_resolve_test.cpp -o /tmp/path_resolve_test
/tmp/path_resolve_test <应用根>
```
