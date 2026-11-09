# MiniMax-H3 gfx936 源码交付

分支：`minimax-h3`；配套 InfiniLM：
https://github.com/aoshuo-tian/InfiniLM/tree/minimax_h3

本版本从实际上线的 `tp-qualified-runtime.tar.gz` 提取源码并与上游指定分支逐项比较。运行包 SHA256：`3e79d685a81ff92db2ef24282b8252e9839807f8706250713d9a0cf98db81450`。

包括已有海光九齿 Attention/MHA、cast、Embedding/SwiGLU 适配、RMSNorm 入口修复、AdaLN、投影/SwiGLU、有门控残差和 Dist 等受保护实现，以及对应生成源码。存在于源码中不表示每一种形状或每一个运行分支都启用。QK 归一化的 InfiniLM 适配与支持代码在配套仓库中，进一步 QK+RoPE 候选保持关闭。

`build/ninetoothed*` 是本提交有意保留的生成 C/C++/头文件，内含 gfx936 嵌入设备代码；不是误上传的主机 `.o/.so`。不要清除后再声称无需重新生成即可构建。历史设备代码生成器也保留在对应算子源码目录。

完整运行配置、构建步骤、原始平台入口、依赖、评分结果和验证边界见配套 InfiniLM 的 `H3_SUBMISSION.md`。特别是实际八卡运行叠加过 CCL-on 库，旧 `--ccl=n` 基础框架构建不等于通信就绪。

本次交付用于 2026-11-09 的 shared TP8 cache 基线：两个原生引擎分别使用同一组 0–7 卡，768×928、50 步、124 帧。没有合并后续视频 VAE Attention/SwiGLU/cast 替换；这不意味着 InfiniCore 的通用 cast 从未适配过。

源码来源清单：`h3_submission/runtime-source-manifest.json`。原上游许可证与子模块引用保留。新整理的公开构建流程尚未在目标 DCU 重建，本轮没有新 GPU 验证或性能结论。
