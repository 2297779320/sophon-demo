//===----------------------------------------------------------------------===//
//
// al_path.h - 配置/模型路径解析（双基准约定，纯字符串、无外部依赖）
//
// 双基准规则（MixedAudioLink 全局约定）：
//   · models.*.path            → **应用根（启动 CWD）** 基准：resolveModelPath
//   · 配置资产（vocab/merges/mel_filters/cmvn_file/prompt_file）
//                              → **启动 JSON 所在目录** 基准：resolveAsset
// 二者均：空串/绝对路径原样返回；返回统一 POSIX '/' 分隔符。
//
// 抽成本头文件的目的是可离线单测（见 tools/path_resolve_test.cpp），
// CLinkMng::resolveAsset / CMeetingSummary::resolveAsset 均委托此处实现。
//
//===----------------------------------------------------------------------===//
#ifndef AL_PATH_H
#define AL_PATH_H

#include <string>

namespace al {

/** 绝对路径判定：'/' 或 '\' 开头，或 Windows 盘符 "X:" 起始 */
inline bool pathIsAbsolute(const std::string &p) {
    return !p.empty() && ('/' == p[0] || '\\' == p[0] ||
                          (p.size() >= 2 && ':' == p[1]));
}

/** 反斜杠 → 正斜杠（就地归一，返回同一字符串） */
inline std::string pathToPosix(std::string p) {
    for (size_t i = 0; i < p.size(); i++)
        if ('\\' == p[i]) p[i] = '/';
    return p;
}

/** dirname（无分隔符时基准为 "."） */
inline std::string pathDirname(const std::string &p) {
    size_t pos = p.find_last_of("/\\");
    return (std::string::npos == pos) ? std::string(".") : p.substr(0, pos);
}

/** 配置资产路径解析：相对 cfg_json_path 所在目录 */
inline std::string resolveAsset(const std::string &cfg_json_path, const std::string &p) {
    if (p.empty() || pathIsAbsolute(p)) return p;
    return pathToPosix(pathDirname(cfg_json_path) + "/" + p);
}

/** 模型路径解析：相对应用根（CWD），仅做分隔符归一（交由运行期按 CWD 解析） */
inline std::string resolveModelPath(const std::string &p) {
    if (p.empty() || pathIsAbsolute(p)) return p;
    return pathToPosix(p);
}

} /* namespace al */

#endif /* AL_PATH_H */
