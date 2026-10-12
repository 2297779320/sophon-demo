//===----------------------------------------------------------------------===//
//
// path_resolve_test.cpp - 路径解析单测 + 出厂配置资产可寻址自测
//
// 目的：验证 cpp/al_path.h 的**双基准**约定，并在一份真实配置上端到端检查
//       "从应用根启动时所有资产/模型均可寻址"。
//
// 构建/运行（纯头文件，宿主 x86 或容器内均可）：
//   g++ -O2 -std=c++17 -I cpp tools/path_resolve_test.cpp -o /tmp/path_resolve_test
//   /tmp/path_resolve_test                 # 仅单测
//   /tmp/path_resolve_test <app_root>      # 单测 + 出厂配置可寻址检查（在应用根下运行）
//
//===----------------------------------------------------------------------===//
#include "al_json.h"
#include "al_path.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;

static void check(const std::string &tag, const std::string &got, const std::string &want) {
    bool ok = (got == want);
    if (!ok) g_fail++;
    printf("[%s] %s  got=%s want=%s\n", ok ? "OK  " : "FAIL", tag.c_str(),
           got.c_str(), want.c_str());
}

/* ---------- 1) 单测：al::resolveAsset / al::resolveModelPath ---------- */
static void unit_tests() {
    const char *cfg = "config/audio/audio_link.json";
    check("asset.rel",        al::resolveAsset(cfg, "vocab.json"),        "config/audio/vocab.json");
    check("asset.mvn",        al::resolveAsset(cfg, "am.mvn"),            "config/audio/am.mvn");
    check("asset.nested",     al::resolveAsset(cfg, "sub/mel_filters.npy"), "config/audio/sub/mel_filters.npy");
    check("asset.no_dir",     al::resolveAsset("audio_link.json", "vocab.json"), "./vocab.json");
    check("asset.win_sep",    al::resolveAsset("config\\audio\\audio_link.json", "vocab.json"),
                                                                          "config/audio/vocab.json");
    check("asset.empty",      al::resolveAsset(cfg, ""),                  "");
    check("asset.abs_unix",   al::resolveAsset(cfg, "/abs/vocab.json"),   "/abs/vocab.json");
    check("asset.abs_win",    al::resolveAsset(cfg, "C:/abs/vocab.json"), "C:/abs/vocab.json");

    check("model.rel",        al::resolveModelPath("models/fsmn_vad.bmodel"), "models/fsmn_vad.bmodel");
    check("model.win_sep",    al::resolveModelPath("models\\fsmn_vad.bmodel"),
                                                                            "models/fsmn_vad.bmodel");
    /* 关键区别：模型相对路径**不**拼接 JSON 目录 */
    check("model.no_cfg_dir", al::resolveModelPath("models/x.bmodel"),    "models/x.bmodel");
    check("model.empty",      al::resolveModelPath(""),                   "");
    check("model.abs",        al::resolveModelPath("/abs/x.bmodel"),      "/abs/x.bmodel");
}

/* ---------- 2) 出厂配置资产可寻址（应用根基准 + JSON 目录基准） ---------- */
static std::string readAll(const std::string &p) {
    std::ifstream ifs(p);
    if (!ifs) return "";
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static void checkExists(const std::string &tag, const std::string &p) {
    bool ok = fs::exists(p);
    if (!ok) g_fail++;
    printf("[%s] %s -> %s\n", ok ? "OK  " : "FAIL", tag.c_str(), p.c_str());
}

/* 检查一份音频配置：models 走应用根；配置资产走 JSON 目录 */
static void checkAudioCfg(const std::string &json_path) {
    std::string text = readAll(json_path);
    if (text.empty()) { g_fail++; printf("[FAIL] 无法读取 %s\n", json_path.c_str()); return; }
    std::string err;
    al::JsonPtr root = al::JsonParser::parse(text, err);
    if (!root) { g_fail++; printf("[FAIL] 解析失败 %s: %s\n", json_path.c_str(), err.c_str()); return; }

    if (auto models = root->get("models")) {
        for (auto &kv : models->obj) {
            std::string p = kv.second ? kv.second->getStr("path") : "";
            if (p.empty()) continue;
            checkExists("models." + kv.first, al::resolveModelPath(p));
        }
    }
    if (auto asr = root->get("asr")) {
        for (const char *k : {"vocab", "merges", "mel_filters"}) {
            std::string p = asr->getStr(k);
            if (!p.empty()) checkExists(std::string("asr.") + k, al::resolveAsset(json_path, p));
        }
    }
    if (auto vad = root->get("vad")) {
        for (const char *k : {"cmvn_file", "cmvn_mean", "cmvn_std"}) {
            std::string p = vad->getStr(k);
            if (!p.empty()) checkExists(std::string("vad.") + k, al::resolveAsset(json_path, p));
        }
    }
}

static void checkSummaryCfg(const std::string &json_path) {
    std::string text = readAll(json_path);
    if (text.empty()) { g_fail++; printf("[FAIL] 无法读取 %s\n", json_path.c_str()); return; }
    std::string err;
    al::JsonPtr root = al::JsonParser::parse(text, err);
    if (!root) { g_fail++; printf("[FAIL] 解析失败 %s: %s\n", json_path.c_str(), err.c_str()); return; }

    if (auto models = root->get("models")) {
        for (auto &kv : models->obj) {
            std::string p = kv.second ? kv.second->getStr("path") : "";
            if (!p.empty()) checkExists("models." + kv.first, al::resolveModelPath(p));
        }
    }
    if (auto sc = root->get("summary")) {
        for (const char *k : {"vocab", "merges", "prompt_file"}) {
            std::string p = sc->getStr(k);
            if (!p.empty()) checkExists(std::string("summary.") + k, al::resolveAsset(json_path, p));
        }
    }
}

int main(int argc, char **argv) {
    printf("===== [1] 路径解析单测 =====\n");
    unit_tests();

    if (argc > 1) {
        std::string root = argv[1];
        fs::current_path(root);   /* 模拟"从应用根启动" */
        printf("\n===== [2] 出厂配置资产可寻址（app_root=%s）=====\n", root.c_str());
        checkAudioCfg("config/audio/audio_link.json");
        checkSummaryCfg("config/summary/meeting_summary.json");
    }

    printf("\n结论：%s（失败 %d 项）\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
