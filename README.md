# VLM 视觉编码器方法迁移

当前版本为 **2026-10-13 eager 迁移代码终稿，32/32 项验收关闭**：Qwen2-VL-2B、SmolVLM2-2.2B、LLaVA-1.5-7B、LLaVA-OneVision-7B，各包含 PiToMe、DToMe、iLLaVA、IPCV 的 FP16/FP32 路径。

## 当前入口

- [终稿代码、重建与验收说明](revisions/native_20261012_r6/README.md)
- [32 项模型/方法/精度索引](revisions/native_20261012_r6/FINAL_METHOD_INDEX_20261013.json)
- [102 文件源码封印](revisions/native_20261012_r6/RELEASE_SHA256.json)
- [96 模板与 6 个私有作者依赖](revisions/native_20261012_r6/FILE_MAP.json)
- [脱敏组件验收摘要](revisions/native_20261012_r6/FINAL_ACCEPTANCE_20261013.json)
- [当前公开 Git 文件清单](sources/PUBLIC_REPOSITORY_MANIFEST_20261013_final.json)

标签 `eager-final-20261013` 固定终稿。公开重建继续使用现有 r6 目录，组合身份为 `native_20261013_final`。普通后继提交保留所有历史；作者私有依赖只固定哈希，不公开完整源。

## 验收依据

终稿继承各组件已通过验收：r4 的 26 个任务、316 次自然 EOS；Qwen r5 的 24 次自然 EOS；Smol r6 双精度的 28 次自然 EOS；LLaVA DToMe clone 修复双精度的 18 次自然 EOS及 CPU membership 反例。**本次组合整理没有新跑整包 GPU**，这些是各版本组件的有界回归记录。

Smol 合并结果先 `.half()` 再 `.to(x.dtype)`，保留作者舍入并满足 scatter 写回类型；LLaVA clone 防止批内合并改写已保存的位置追踪。其余已验收算法保持原字节，Qwen 文本 QK32 与必要显存、宿主接口适配保留。选择、排序、权重和重建按作者方法迁移；候选输出不以复刻旧迁移输出为目标。

**全量质量和正式 280 结果仍按另行计划执行。** 代码终稿不表示正式实验已经完成；代表题回归不作为全数据集准确率或速度结论。[FA2 FP16 路线](backends/qwen_fa2_fp16/README.md) 继续独立预审，非本次终稿。

统一有界回归入口为 `run.py`，仍需要已有离线权重、数据、私有作者依赖及冻结环境。每题使用全部原图，顺序 batch size 1，不跨题复用 ViT。纯 ViT 与端到端计时来自同一次完整生成，包含观察器同步开销；附加图片验证按用户要求跳过。

已记录的 no-delete guard 提示发生在成功后的临时文件清理；既有退出码和结果审计通过，服务器保护保持。应用侧写入/恢复验证不宣称解决了共享存储历史 ENOSPC 的基础设施根因。

## 历史与许可

- [r6 Smol 原验收](revisions/native_20261012_r6/VALIDATION.json) · [r5 Qwen 修复](revisions/native_20261012_r5/README.md)
- [r4 全方法回归](revisions/native_20261012_r4/README.md) · [原方法来源与适配说明](revisions/native_20261012_r4/METHOD_FIDELITY.md)
- [LLaVA DToMe 组件记录](models/llava15-7b/dymu/README.md)
- [r3 修订](revisions/native_20261012/README.md) · [2026-10-09 固定版本](https://github.com/aaash25/method-VIT-261009/tree/2284df03820d4b88914d446b3e31955641de929c)
- [旧源码快照说明](SOURCE_CANDIDATE_README.md) · [旧模型方法索引](sources/MODEL_METHOD_INDEX.json)
- [写入与恢复历史修复](revisions/storage_repairs_20261011/README.md)

旧材料保留各自日期和状态，当前代码入口以上述终稿索引为准。许可按组件继承：[组件许可与版权](third_party/COMPONENT_NOTICES.md)；不为全仓统一重新授权。模型权重、数据实体、真实题干/图片、私人路径、凭据和未筛选原始日志不进入公开仓库。
