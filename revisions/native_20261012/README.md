# 2026-10-12 原生方法修订

本目录保存移除额外确定性处理后的迁移代码、统一回归入口和必要适配说明。
作者已提供目标模型实现的路径优先调用固定版本的作者代码；其余路径维持方法逻辑，并明确记录宿主模型所需的接口适配。

这是依赖既有离线模型与冻结运行环境的增量包，包含 95 个公开模板文件；另有 6 个含 iLLaVA 作者算法的文件沿用仓库既有公开边界，以固定哈希的本地依赖提供。完整私有服务器包包含这些文件。公开模板可以通过 `materialize.py` 恢复为已测试的服务器源码，恢复过程中逐文件校验，遇到缺失或改动立即失败。

## 本次修改

- Qwen PiToMe / DToMe：使用作者的匹配、排序和 scatter 规则，移除额外稳定排序与强制确定性分组求和。FP16 合并的中间累加保留 FP32，避免有限输入在求和阶段溢出；输出仍回到模型 dtype。
- Smol DToMe：使用作者的单实例 scatter-reduce 和批内 scatter-add 语义，移除迁移中额外的排序式 FP32 求和。诊断检查与逐层报告退出默认计时路径。
- LLaVA DToMe：保留原生作者视觉核心，逐层计数推迟到完整请求计时结束后读取。
- OneVision DToMe / PiToMe：取消额外的强制确定性调度；保留作者的批内排序、epsilon、padding、重建规则，报告在计时后汇总。
- Qwen IPCV：诊断 hook 放在不计时的预热检查中，完整请求计时不再携带该诊断。

Qwen FP16 文本 QK32 从未在此修订中删除。最初候选错误地连合并累加保护也一并移除，真实 POPE 请求在第 32 层出现有限输入、无穷输出。当前修订恢复该必要保护，保留失败证据；这与 QK 得分的 FP32 计算是两处独立适配。详见 [适配理由](templates/ADAPTATIONS_20261012.md) 和 [溢出回归测试](templates/qwen_ports/test_native_accumulation.py)。

## 运行入口

`templates/` 是发布模板，不能直接运行。先在已有冻结环境的服务器上恢复到新目录：

```bash
python materialize.py --project-root "$VLM_PROJECT" --local-author-overlay "$EXISTING_PRIVATE_OVERLAY" --output "$NEW_RELEASE"
```

`--local-author-overlay` 指向已交付的完整私有修复包。脚本只读取其中清单指定的 6 个文件，不下载作者代码，不修改旧包，不重新生成虚假的验收身份。项目根路径必须与已测试部署一致，否则哈希校验拒绝恢复；换路径需要独立验证的新发行身份。

恢复后使用项目已验证的 Python 解释器，唯一入口为 `run.py`：

```bash
"$VLM_PYTHON" "$NEW_RELEASE/run.py" --models qwen smol llava onevision --precisions float16 float32 --dry-run
```

实际运行移除 `--dry-run`，并提供经平台核实的 `--lease-remaining-seconds`。入口逐实例顺序运行，检查冻结队列、源码、CPU/GPU 组件、完整生成、方法激活、计时边界、journal 与导出，并保留失败记录。不要在不适合模型精度的显卡上盲目选择全部组合。历史子目录的 `launch.py` 不是本次统一入口。

依赖仍包括已命名的 Qwen r6、Smol fresh r2、LLaVA r7、OneVision 原生 FP16 与 FP32 冻结运行环境，以及对应离线权重、处理器和数据集。这些资源不在公开仓库中；模型简称不替代 checkpoint/revision 身份。

## 验收范围

此入口运行有界回归，不会启动全量数据集、阈值搜索或暂停中的正式实验。一题保留自己的全部原图，以 batch size 1 顺序执行，不跨题复用 ViT。每次完整自然结束的生成同时记录纯 ViT 与端到端时间。图片附加验证按用户要求跳过；不把跳过写成通过。

压缩后结果允许遵循作者方法发生变化。Dense、禁用模式和状态恢复仍用于检查宿主兼容性；它们不要求压缩输出复刻旧迁移结果。这些小规模回归不构成全量准确率或加速结论。

最终服务器实测与下载后的独立复核均通过：[验收记录](VALIDATION_20261012.json)。4 个模型 × 4 种方法 × FP16/FP32，由 26 个工作任务覆盖，共 316 次完整自然结束生成；20 组跨方法 Dense 完整输出一致。LLaVA/OneVision FP32 使用 A800，其余使用 V100。FP32 的 r2 证据仅复用于与 r3 源码完全相同的 LLaVA/OneVision 子包。Smol 每个精度的单个任务内执行四种方法，因此任务数不是 32。

[audit_evidence.py](audit_evidence.py) 可对私有证据包重复独立核查。公开验收文件含逐请求双计时、完整输出哈希和硬件来源，不包含真实题干、图片或原始回答。旧回执的历史元数据标签保持原样，其与实际调用路径的差别已在 [METHOD_FIDELITY.md](METHOD_FIDELITY.md) 披露。

## 历史与许可

旧源码、失败和既有验收身份保留。本次以新增目录和后继 Git 提交更新，不改写旧提交。`FILE_MAP.json` 记录公开模板与已测试源码之间的逐文件关系。

许可按组件继承仓库的 [版权与许可说明](../../third_party/COMPONENT_NOTICES.md)：PiToMe CC BY-NC 4.0，DyMU MIT，IPCV/HF/LLaVA 对应部分 Apache-2.0，InternVL 底层 MIT。新增作者核心的出处仍在源码和固定来源记录中。iLLaVA 作者实现继续作为本地依赖，不因此发布统一仓库许可证。
