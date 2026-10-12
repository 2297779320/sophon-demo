#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""funasr fsmn_vad_streaming 端点状态机参考实现 + 与 C++ FsmnVadSegmenter 对拍。

参考实现按 funasr/models/fsmn_vad_streaming/model.py 的字面风格书写
（output_data_buf + 逐帧 pop + PopDataBufTillFrame / DropCachedFrames），
C++ 侧用「出段时一次性截取」的等价简化 —— 两者结构不同，对拍才能验证简化正确。

用法：
  # 1) 编译 C++ 测试（宿主机即可，段状态机不依赖 bmrt）
  docker exec soc-cross bash -lc 'cd /workspace/MixedAudioLink && \
      g++ -O2 -std=c++17 -I cpp tools/vad_seg_test.cpp -o /tmp/vad_seg_test'
  # 2) 跑对拍
  docker exec soc-cross bash -lc 'cd /workspace/MixedAudioLink && \
      python3 tools/vad_seg_ref.py'

环境变量：VAD_SEG_BIN 指定 C++ 测试二进制（默认 /tmp/vad_seg_test）、VAD_SEG_VERBOSE=1 打印每例段表。
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

FS = 16000


# ============================ 参考实现（funasr 字面移植） ============================

class WindowDetector:
    def __init__(self, window_size_ms, sil_to_speech_time, speech_to_sil_time, frame_size_ms):
        self.win_size_frame = int(window_size_ms / frame_size_ms)
        self.sil_to_speech_frmcnt_thres = int(sil_to_speech_time / frame_size_ms)
        self.speech_to_sil_frmcnt_thres = int(speech_to_sil_time / frame_size_ms)
        self.reset()

    def reset(self):
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.cur_win_pos = 0
        self.pre_frame_state = 0
        self.cur_frame_state = 0

    def detect_one_frame(self, frame_state):
        cur = 1 if frame_state == 1 else 0
        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += cur
        self.win_state[self.cur_win_pos] = cur
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame
        if self.pre_frame_state == 0 and self.win_sum >= self.sil_to_speech_frmcnt_thres:
            self.pre_frame_state = 1
            return "Sil2Speech"
        if self.pre_frame_state == 1 and self.win_sum <= self.speech_to_sil_frmcnt_thres:
            self.pre_frame_state = 0
            return "Speech2Sil"
        if self.pre_frame_state == 0:
            return "Sil2Sil"
        return "Speech2Speech"


class RefVad:
    """funasr FsmnVADStreaming 等价体（单文件字面移植）"""

    def __init__(self, cfg):
        self.c = cfg
        self.frame_ms = cfg["frame_ms"]
        self.shift = cfg["frame_samples"]
        self.extra = cfg["extra_samples"]
        self.wd = WindowDetector(cfg["window_ms"], cfg["sil2sp_ms"], cfg["sp2sil_ms"], cfg["frame_ms"])
        self.max_end_sil_frame_cnt_thresh = cfg["max_end_sil_ms"] - cfg["sp2sil_ms"]
        self.st = {
            "latest_confirmed_speech_frame": 0,
            "lastest_confirmed_silence_frame": -1,
            "confirmed_start_frame": -1,
            "confirmed_end_frame": -1,
            "continous_silence_frame_count": 0,
            "vad_state_machine": 0,            # 0 StartNotDetected / 1 InSpeech / 2 EndDetected
            "number_end_time_detected": 0,
            "data_buf": [],
            "data_buf_start_frame": 0,
            "output_data_buf": [],
            "sil_frame": 0,
        }
        self.completed = []
        self._last_end_ms = None

    # ---- 缓冲原语 ----
    def pop_data_buf_till_frame(self, frame_idx):
        st = self.st
        if st["data_buf_start_frame"] < frame_idx:
            drop = (frame_idx - st["data_buf_start_frame"]) * self.shift
            drop = min(drop, len(st["data_buf"]))
            st["data_buf"] = st["data_buf"][drop:]
            st["data_buf_start_frame"] = frame_idx

    def pop_data_to_output_buf(self, start_frm, frm_cnt, first_is_start, last_is_end, sent_end):
        st = self.st
        self.pop_data_buf_till_frame(start_frm)
        expected = int(frm_cnt * FS * self.frame_ms / 1000)
        if last_is_end:
            expected += self.extra
        if sent_end:
            expected = max(expected, len(st["data_buf"]))
        if len(st["output_data_buf"]) == 0 or first_is_start:
            st["output_data_buf"].append({
                "start_ms": start_frm * self.frame_ms,
                "end_ms": start_frm * self.frame_ms,
                "pcm": [],
                "final": 0,
                "contain_end": False,
            })
        cur = st["output_data_buf"][-1]
        data_to_pop = expected if sent_end else int(frm_cnt * self.frame_ms * FS / 1000)
        data_to_pop = min(data_to_pop, len(st["data_buf"]))
        cur["pcm"].extend(st["data_buf"][:data_to_pop])
        st["data_buf"] = st["data_buf"][data_to_pop:]
        st["data_buf_start_frame"] += frm_cnt
        cur["end_ms"] = (start_frm + frm_cnt) * self.frame_ms
        if last_is_end:
            cur["contain_end"] = True
            if sent_end:
                cur["final"] = 1

    def drop_cached_frames(self, drop_frames):
        st = self.st
        real = drop_frames - st["sil_frame"]
        if real > st["data_buf_start_frame"]:
            drop = (real - st["data_buf_start_frame"]) * self.shift
            st["data_buf"] = st["data_buf"][drop:]
            st["data_buf_start_frame"] = real

    # ---- 状态机回调 ----
    def latency_frm_num_at_start_point(self):
        return self.wd.win_size_frame + (
            int(self.c["lookback_ms"] / self.frame_ms) if self.c["do_extend"] else 0)

    def on_voice_start(self, start_frame, fake=False):
        self.st["confirmed_start_frame"] = start_frame
        if not fake and self.st["vad_state_machine"] == 0:
            self.pop_data_to_output_buf(start_frame, 1, True, False, False)

    def on_voice_detected(self, valid_frame):
        self.st["latest_confirmed_speech_frame"] = valid_frame
        self.pop_data_to_output_buf(valid_frame, 1, False, False, False)

    def on_silence_detected(self, valid_frame):
        self.st["lastest_confirmed_silence_frame"] = valid_frame
        if self.st["vad_state_machine"] == 0:
            self.pop_data_buf_till_frame(valid_frame)

    def on_voice_end(self, end_frame, fake, is_last_frame):
        st = self.st
        for t in range(st["latest_confirmed_speech_frame"] + 1, end_frame):
            self.on_voice_detected(t)
        st["confirmed_end_frame"] = end_frame
        if not fake:
            st["sil_frame"] = 0
            self.pop_data_to_output_buf(end_frame, 1, False, True, is_last_frame)
            # 段交付（C++ 侧在 onVoiceEnd 即产出，这里对齐同一时点）
            self._last_end_ms = st["output_data_buf"][-1]["end_ms"]
            self.completed.extend(st["output_data_buf"])
            st["output_data_buf"] = []
        st["number_end_time_detected"] += 1

    def reset_detection(self):
        st = self.st
        st["continous_silence_frame_count"] = 0
        st["latest_confirmed_speech_frame"] = 0
        st["lastest_confirmed_silence_frame"] = -1
        st["confirmed_start_frame"] = -1
        st["confirmed_end_frame"] = -1
        st["vad_state_machine"] = 0
        self.wd.reset()
        st["sil_frame"] = 0
        # funasr ResetDetection → DropCachedFrames(last_seg.end_ms / frame_in_ms)
        if self._last_end_ms is not None:
            self.drop_cached_frames(int(self._last_end_ms / self.frame_ms))
            self._last_end_ms = None

    # ---- 逐帧 ----
    def detect_one_frame(self, cur_frm_state, cur_frm_idx, is_final_frame):
        st = self.st
        tmp = 1 if cur_frm_state == 1 else 0     # fe_prior_thres 恒真 → 透传
        change = self.wd.detect_one_frame(tmp)
        frm_shift = self.frame_ms
        max_frames = self.c["max_single_seg_ms"] / frm_shift

        if change == "Sil2Speech":
            st["continous_silence_frame_count"] = 0
            if st["vad_state_machine"] == 0:
                start_frame = max(st["data_buf_start_frame"],
                                  cur_frm_idx - self.latency_frm_num_at_start_point())
                self.on_voice_start(start_frame)
                st["vad_state_machine"] = 1
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.on_voice_detected(t)
            elif st["vad_state_machine"] == 1:
                for t in range(st["latest_confirmed_speech_frame"] + 1, cur_frm_idx):
                    self.on_voice_detected(t)
                if cur_frm_idx - st["confirmed_start_frame"] + 1 > max_frames:
                    self.on_voice_end(cur_frm_idx, False, False)
                    st["vad_state_machine"] = 2
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx)
                else:
                    self.on_voice_end(cur_frm_idx, False, True)
                    st["vad_state_machine"] = 2
        elif change == "Speech2Sil":
            st["continous_silence_frame_count"] = 0
            if st["vad_state_machine"] == 1:
                if cur_frm_idx - st["confirmed_start_frame"] + 1 > max_frames:
                    self.on_voice_end(cur_frm_idx, False, False)
                    st["vad_state_machine"] = 2
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx)
                else:
                    self.on_voice_end(cur_frm_idx, False, True)
                    st["vad_state_machine"] = 2
        elif change == "Speech2Speech":
            st["continous_silence_frame_count"] = 0
            if st["vad_state_machine"] == 1:
                if cur_frm_idx - st["confirmed_start_frame"] + 1 > max_frames:
                    self.on_voice_end(cur_frm_idx, False, False)
                    st["vad_state_machine"] = 2
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx)
                else:
                    self.on_voice_end(cur_frm_idx, False, True)
                    st["vad_state_machine"] = 2
        else:  # Sil2Sil
            st["continous_silence_frame_count"] += 1
            if st["vad_state_machine"] == 0:
                if ((self.c["detect_mode"] == 0 and
                     st["continous_silence_frame_count"] * frm_shift > self.c["max_start_sil_ms"]) or
                        (is_final_frame and st["number_end_time_detected"] == 0)):
                    for t in range(st["lastest_confirmed_silence_frame"] + 1, cur_frm_idx):
                        self.on_silence_detected(t)
                    self.on_voice_start(0, True)          # fake start（不出段）
                    self.on_voice_end(0, True, False)     # fake end（不出段）
                    st["vad_state_machine"] = 2
                elif cur_frm_idx >= self.latency_frm_num_at_start_point():
                    self.on_silence_detected(cur_frm_idx - self.latency_frm_num_at_start_point())
            elif st["vad_state_machine"] == 1:
                if st["continous_silence_frame_count"] * frm_shift >= self.max_end_sil_frame_cnt_thresh:
                    lookback_frame = int(self.max_end_sil_frame_cnt_thresh / frm_shift)
                    if self.c["do_extend"]:
                        lookback_frame -= int(self.c["lookahead_ms"] / frm_shift)
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.on_voice_end(cur_frm_idx - lookback_frame, False, False)
                    st["vad_state_machine"] = 2
                elif cur_frm_idx - st["confirmed_start_frame"] + 1 > max_frames:
                    self.on_voice_end(cur_frm_idx, False, False)
                    st["vad_state_machine"] = 2
                elif self.c["do_extend"] and not is_final_frame:
                    if st["continous_silence_frame_count"] <= int(self.c["lookahead_ms"] / frm_shift):
                        self.on_voice_detected(cur_frm_idx)
                elif is_final_frame:
                    self.on_voice_end(cur_frm_idx, False, True)
                    st["vad_state_machine"] = 2

        if st["vad_state_machine"] == 2 and self.c["detect_mode"] == 1:
            self.reset_detection()

    # ---- 驱动 ----
    def push(self, chunk_pcm, abs_beg, decisions, frame0, is_final_batch):
        self.st["data_buf"].extend(chunk_pcm)
        n = len(decisions)
        for f in range(n):
            self.detect_one_frame(decisions[f], frame0 + f, is_final_batch and f == n - 1)

    def drain(self):
        out = self.completed
        self.completed = []
        return out


# ============================ fixture / 对拍 ============================

def fnv1a(samples):
    h = 1469598103934665603
    for v in samples:
        for x in struct.pack("<h", v):
            h ^= x
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def make_pcm(nframes, shift):
    n = nframes * shift
    return [(i % 2000) - 1000 for i in range(n)]


def run_ref(cfg, nframes, dec, chunks):
    ref = RefVad(cfg)
    pcm = make_pcm(nframes, cfg["frame_samples"])
    shift = cfg["frame_samples"]
    f0 = 0
    out = []
    for ci, n in enumerate(chunks):
        if f0 >= nframes:
            break
        n = min(n, nframes - f0)
        if n <= 0:
            break
        cp = pcm[f0 * shift:(f0 + n) * shift]
        d = [1 if (f0 + i < len(dec) and dec[f0 + i] == "1") else 0 for i in range(n)]
        ref.push(cp, f0 * shift, d, f0, ci + 1 == len(chunks))
        for s in ref.drain():
            out.append({"beg_ms": s["start_ms"], "end_ms": s["end_ms"],
                        "final": s["final"], "n": len(s["pcm"]),
                        "hash": fnv1a(s["pcm"])})
        f0 += n
    return out


def gen_fixture(path, cfg, nframes, dec, chunks):
    with open(path, "w") as f:
        for k, v in cfg.items():
            f.write("%s %d\n" % (k, (1 if v else 0) if isinstance(v, bool) else v))
        f.write("nframes %d\n" % nframes)
        f.write("decision %s\n" % dec)
        f.write("chunks %s\n" % " ".join(str(c) for c in chunks))


def default_cfg(**kw):
    cfg = dict(frame_ms=10, frame_samples=160, extra_samples=240, window_ms=200,
               sil2sp_ms=150, sp2sil_ms=150, max_end_sil_ms=800, max_start_sil_ms=3000,
               max_single_seg_ms=20000, do_extend=1, lookback_ms=200, lookahead_ms=100,
               detect_mode=1)
    cfg.update(kw)
    return cfg


def split_chunks(nframes, size, rng=None, jitter=False):
    chunks, left = [], nframes
    while left > 0:
        n = size
        if jitter:
            n = max(1, size + rng.randint(-size // 2, size // 2))
        n = min(n, left)
        chunks.append(n)
        left -= n
    return chunks


def build_cases():
    rng = random.Random(20261010)
    cases = []

    def add(name, cfg, dec, chunks):
        cases.append((name, cfg, len(dec), dec, chunks))

    # 1) 随机判决 × 多种稀疏度 × 多种切块
    for p in (0.02, 0.08, 0.2, 0.5):
        for nf in (400, 1200):
            dec = "".join("1" if rng.random() < p else "0" for _ in range(nf))
            for cs in (1, 3, 7, 18, 20, 64, 200):
                add("rand_p%d_n%d_c%d" % (int(p * 100), nf, cs), default_cfg(), dec,
                    split_chunks(nf, cs))
            add("rand_p%d_n%d_jit" % (int(p * 100), nf), default_cfg(), dec,
                split_chunks(nf, 20, rng, jitter=True))

    # 2) 纯静音 / 纯语音
    for nf in (100, 900, 3000):
        add("silence_n%d" % nf, default_cfg(), "0" * nf, split_chunks(nf, 20))
        add("speech_n%d" % nf, default_cfg(), "1" * nf, split_chunks(nf, 20))

    # 3) 短促突发（考验 20 帧窗 ≥15 帧阈值）
    for burst in (5, 10, 14, 15, 16, 25, 40):
        dec = "0" * 60 + "1" * burst + "0" * 200
        add("burst_%d" % burst, default_cfg(), dec, split_chunks(len(dec), 20))
        add("burst_%d_c1" % burst, default_cfg(), dec, split_chunks(len(dec), 1))

    # 4) 超长语音（触发 max_single_segment 截断）
    for nf in (2500, 4200):
        add("long_speech_n%d" % nf, default_cfg(), "1" * nf, split_chunks(nf, 20))
    add("long_speech_16s", default_cfg(max_single_seg_ms=16000), "1" * 3000,
        split_chunks(3000, 20))

    # 5) 多句（模拟连续对话）
    seg = "0" * 40 + "1" * 300 + "0" * 200 + "1" * 150 + "0" * 120 + "1" * 500 + "0" * 300
    for cs in (1, 20, 47, 200):
        add("multi_sent_c%d" % cs, default_cfg(), seg, split_chunks(len(seg), cs))

    # 6) 参数变体
    add("no_extend", default_cfg(do_extend=0), seg, split_chunks(len(seg), 20))
    add("end_sil_400", default_cfg(max_end_sil_ms=400), seg, split_chunks(len(seg), 20))
    add("end_sil_1500", default_cfg(max_end_sil_ms=1500), seg, split_chunks(len(seg), 20))
    add("lookahead_200", default_cfg(lookahead_ms=200), seg, split_chunks(len(seg), 20))
    add("mode0_silence", default_cfg(detect_mode=0), "0" * 800, split_chunks(800, 20))
    add("mode0_speech", default_cfg(detect_mode=0), "1" * 600, split_chunks(600, 20))

    return cases


def main():
    binary = os.environ.get("VAD_SEG_BIN", "/tmp/vad_seg_test")
    verbose = os.environ.get("VAD_SEG_VERBOSE", "") not in ("", "0")
    if not os.path.exists(binary):
        print("ERROR: C++ 测试二进制不存在: %s（先按文件头注释编译）" % binary)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="vadseg_")
    cases = build_cases()
    nfail = 0
    nseg = 0
    for name, cfg, nframes, dec, chunks in cases:
        fx = os.path.join(tmpdir, name + ".fx")
        gen_fixture(fx, cfg, nframes, dec, chunks)
        ref = run_ref(cfg, nframes, dec, chunks)
        r = subprocess.run([binary, fx], capture_output=True, text=True)
        if r.returncode != 0:
            print("[FAIL] %-26s C++ 退出码 %d  %s" % (name, r.returncode, r.stderr.strip()))
            nfail += 1
            continue
        cpp = json.loads(r.stdout)["segments"]
        nseg += len(ref)
        if len(ref) != len(cpp):
            print("[FAIL] %-26s 段数不一致 ref=%d cpp=%d" % (name, len(ref), len(cpp)))
            print("        ref=%s" % json.dumps(ref))
            print("        cpp=%s" % json.dumps(cpp))
            nfail += 1
            continue
        bad = []
        for i, (a, b) in enumerate(zip(ref, cpp)):
            if (a["beg_ms"] != b["beg_ms"] or a["end_ms"] != b["end_ms"] or
                    a["final"] != b["final"] or a["n"] != b["n"] or a["hash"] != b["hash"]):
                bad.append((i, a, b))
        if bad:
            print("[FAIL] %-26s %d/%d 段不一致" % (name, len(bad), len(ref)))
            for i, a, b in bad[:4]:
                print("        #%d ref=%s" % (i, json.dumps(a)))
                print("           cpp=%s" % json.dumps(b))
            nfail += 1
            continue
        if verbose:
            print("[ ok ] %-26s segments=%d %s" % (name, len(ref), json.dumps(ref)))

    print("=" * 72)
    print("用例 %d 个，段合计 %d，失败 %d" % (len(cases), nseg, nfail))
    print("结论：%s" % ("全部一致（bit-exact：段边界 + PCM 长度 + PCM 内容哈希）"
                       if nfail == 0 else "存在不一致，见上"))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
