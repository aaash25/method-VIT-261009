# Eager 迁移代码终稿 · 2026-10-13

四模型 × 四方法 × FP16/FP32 共 **32/32 项代码验收关闭**。本目录沿用 r6 的公开重建入口，当前组合身份为 `native_20261013_final`，标签为 `eager-final-20261013`。

当前源码由 r6 加已验收的 LLaVA DToMe clone 修复组成。**继承各组件已通过验收**；本次没有新跑组合包 GPU，不将组件回归描述为一次新整包运行。全量质量与正式 280 结果按另行计划执行。FA2 继续独立预审，非本次终稿。

## 代码与验收

- [32 项终稿索引](FINAL_METHOD_INDEX_20261013.json)
- [组合源码封印](RELEASE_SHA256.json) · [公开模板/私有依赖清单](FILE_MAP.json)
- [脱敏组件验收摘要](FINAL_ACCEPTANCE_20261013.json) · [组合来源](SOURCE_RECEIPT.json)
- [Smol r6 原验收](VALIDATION.json) · [Qwen r5 原验收](../native_20261012_r5/VALIDATION.json)
- [r4 有界回归](../native_20261012_r4/VALIDATION_20261012_R4.json) · [LLaVA DToMe 原验收](../../models/llava15-7b/dymu/VALIDATION_20261013.json)

Smol 实际 `illava_step` 先 `.half()` 再 `.to(x.dtype)`，保留作者 FP16 舍入并满足 scatter 同 dtype；选择、排序、权重与恢复不变。38 项 CPU、CPU/CUDA 舍入各 4 单元通过，无 skip；V100 双精度各 14 模式，共 28 个自然 EOS，激活、Dense 恢复、journal/导出通过。其完整混合作者模块仍是哈希固定的本地依赖。

LLaVA DToMe 的批内目的视图使用 `.clone()`，避免原地合并改写已经保存的位置追踪。仅采用已验收核心、测试与 bundle；其他模型的同名核心保持原字节。FP16/V100、FP32/A800 各 9 次自然 EOS，独立审计通过；B2/T6 CPU 反例验证输入 membership/size 不变、列和为 1、保存槽位恢复为 30，并核验返回结果。clone 成本包含在 ViT 计时中。

原 r4 方法说明中 Qwen/Smol 使用输入 dtype 的历史表述，由 r5/r6 舍入恢复取代；冻结源码内的旧 docstring 仍保留，实际改动以本页和固定哈希为准。

## 精确重建

`materialize.py` 从 **96 个公开模板与 6 个哈希固定本地作者依赖**重建 102 个文件。使用已有 r6 私有依赖，`--project-root` 必须是原验证部署根；未测试的路径替换不能冒充该封印。`FILE_MAP.json` 记录逐文件字节身份，`RELEASE_SHA256.json` 记录当前组合封印。

公开包仍需要既有离线权重、数据和冻结运行环境。统一 `run.py` 是有界回归入口；代码终稿不会自动启动正式实验。每题保留全部原图，顺序 batch size 1，不跨题复用 ViT；纯 ViT/E2E 来自同一次完整生成。附加图片验证按用户要求跳过。

成功退出后的临时文件清理可能被 no-delete guard 拦截；既有推理退出码和独立审计均通过。服务器保护、历史失败和旧提交保留。

[组件许可与版权](../../third_party/COMPONENT_NOTICES.md) · [FA2 独立路线](../../backends/qwen_fa2_fp16/README.md) · [返回仓库](../../README.md)
