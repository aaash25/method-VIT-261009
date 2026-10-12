# 2026-10-12 r4 原生方法修订

本目录对应服务器两卡实测的 `native_release_20261012_r4`。完整私有包 SHA-256：`f6e267147ebb9a542c55e876fac662d910f7c0d97fde88868f9f0ead951b1824`。本次只调整两张卡的任务分配，未修改实测源码。r3 目录和历史提交保留。

公开增量包包含 94 个源码模板；5 个含 iLLaVA 作者算法或混合实现的文件继续作为本地固定哈希依赖，不改变既有公开边界。LLaVA 的 clip_peer_ports.py 现在从既有冻结基包读取，其身份由 BASE_SOURCES_SHA256.json 检查。模型、数据集、冻结环境和本地依赖仍需服务器已有资源。

## 验收结果

[独立验收记录](VALIDATION_20261012_R4.json)：26 个任务、316 次完整自然结束生成、26 次独立审计通过。Qwen GPU 组件 96 passed / 0 skipped；OneVision FP16 GPU 组件 28 passed / 0 skipped，包含实际 CUDA 的 FP32 作者 .half() 测试。LLaVA 与 OneVision 的两项 FP32 iLLaVA 完整生成均通过。

200 份可比较的历史 Dense/禁用/恢复输出未变；5 份压缩候选输出发生变化，记录为观察，不按逐元素或逐 token 差异判失败。FP32 iLLaVA 本次未观察到候选 token 改变。源码冻结前后哈希、任务归档 CRC/SHA、完整 token、同步双计时事件顺序和 journal/export 一致性通过。图片附加验证按用户要求跳过。

仍有明确未解决的清理异常：26 个任务日志在退出时出现服务器禁止删除保护拦截临时文件清理的 PermissionError。任务均退出 0，结果保存及独立审计通过；按照用户限制未修改。这不是日志完全无异常的声明。

这是有界回归入口，不是全量数据集实验或性能结论。[方法与必要适配](METHOD_FIDELITY.md) 和 [r4 变更说明](templates/ADAPTATIONS_20261012.md) 保留原生方法、Qwen QK32 和 FP16 中间累加保护。源码中的 GPU pending 与 SOURCE_FREEZE_STATUS.json 是冻结时历史状态；后续独立验收记录覆盖当前实测状态，不改写原源码身份。

## 恢复与运行

恢复模板时逐文件验证与服务器实测源码相同，任一文件缺失或哈希不同立即失败：

```bash
python materialize.py --project-root "$VLM_PROJECT" --local-author-overlay "$R4_PRIVATE_OVERLAY" --output "$NEW_R4_RELEASE"
```

项目根路径必须与实测部署一致，否则拒绝生成新的未经验证身份。公开包不包含 WORK_STATUS.json；其冻结记录单独保存在 SOURCE_FREEZE_STATUS.json，不参与运行。

唯一回归入口仍为恢复后的 run.py。两卡各顺序执行、并发各 1；V100 先 Qwen/Smol FP16+FP32，再 LLaVA/OneVision FP16，A800 执行 LLaVA/OneVision FP32：

```bash
"$VLM_PYTHON" "$NEW_R4_RELEASE/run.py" --models qwen smol --precisions float16 float32 --dry-run
"$VLM_PYTHON" "$NEW_R4_RELEASE/run.py" --models llava onevision --precisions float16 --dry-run
"$VLM_PYTHON" "$NEW_R4_RELEASE/run.py" --models llava onevision --precisions float32 --dry-run
```

实跑移除 --dry-run 并提供平台核实的 --lease-remaining-seconds；不得在一张不匹配硬件的卡上运行全部组合。每题保留全部原图，batch size 1，不跨题共享 ViT；同一次完整生成同时记录纯 ViT 与 E2E。

[audit_evidence.py](audit_evidence.py) 用 --evidence 指向三份私有分组证据包、--source-root 指向恢复后的 r4、--output 指向新回执，可独立重复检查。公开验收包含双计时、完整输出哈希和硬件来源；不发布真实题干、回答或未筛选原始日志。

许可与作者归属沿用仓库的 [组件版权说明](../../third_party/COMPONENT_NOTICES.md)，不统一改授许可证。
