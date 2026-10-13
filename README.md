# VLM 视觉编码器方法迁移

**2026-10-13 更新：LLaVA-1.5-7B × DToMe（FP16/FP32）当前对应版本备注为终稿。** [组合说明与源码](models/llava15-7b/dymu/README.md) · [18 次 GPU 完整生成及审计](models/llava15-7b/dymu/VALIDATION_20261013.json)。直接更新现有模板，其他模型、方法和 FA2 状态不变。

当前修订为 **2026-10-12 r6 Smol 舍入兼容版**。优先使用作者对应模型实现，去除额外稳定排序、强制确定性归约及计时路径中的诊断开销；保留并说明必要的接口、FP16/FP32 和显存适配。

r4 已完成四个模型、四种方法、两种精度的有界回归：**26 个任务、316 次完整自然结束生成通过**。r5 恢复 Qwen iLLaVA 作者 `.half()` 后，另完成 **24 次完整自然结束生成**；其余路径保持 r4 字节。这支持当前迁移和已知问题的回归验收，不代表全量数据集准确率或任何输入都不会报错。

2026-10-12 的 r6 为 Smol iLLaVA 保留作者 FP16 舍入并适配 scatter 写回类型，V100 双精度另完成 **28 次完整自然结束生成**；当时其他模型和方法保持 r5 字节；2026-10-13 的 LLaVA DToMe 更新以上述组合说明为准。

## 当前入口

- [r6 Smol 舍入兼容及验收](revisions/native_20261012_r6/README.md)
- [r5 Qwen 舍入恢复及验收](revisions/native_20261012_r5/README.md)
- [FA2 FP16 独立路线：标准预审与待交叉审核项](backends/qwen_fa2_fp16/README.md)

下列 r4 来源和全方法回归记录保留：

- [修订说明与统一运行入口](revisions/native_20261012_r4/README.md)
- [16 个模型×方法的来源、适配和边界](revisions/native_20261012_r4/METHOD_FIDELITY.md)
- [完整验收记录与逐请求双计时](revisions/native_20261012_r4/VALIDATION_20261012_R4.json)
- [公开模板与已测试代码的对应清单](revisions/native_20261012_r4/FILE_MAP.json)
- [写入、导出与两小时恢复验证的历史修复](revisions/storage_repairs_20261011/README.md)

统一有界回归入口为 `run.py`。服务器完整私有 r6 包已部署；r6 公共重建包包含 96 个模板文件，6 个作者源文件使用固定哈希的本地依赖。FA2 栏目当前是预审状态，未启动全量，不与 eager 结果混列。仍需既有离线模型、数据集和冻结环境，不能直接把公开模板作为独立模型发行包运行。

Qwen FP16 文本 QK32 一直保留。此次误删的是另一处合并中间累加保护，现已恢复为 FP32 累加，并通过真实失败题回归。压缩输出遵循作者方法，不以复刻旧迁移的逐元素结果为目标。

每题使用自己的全部原图，顺序 batch size 1，不跨题复用 ViT。纯 ViT 与端到端计时来自同一次完整生成；图片附加验证按用户要求跳过。显存适配与写入修复均保留证据，但应用侧验证不等于确认共享存储历史 ENOSPC 的基础设施根因。

仍有退出时临时文件清理被服务器禁止删除保护拦截的 PermissionError；26 个任务退出码和结果审计均通过。本轮未修复该提示。

## 历史版本

本次新增修订目录并创建后继提交，未改写旧提交。下列目录和报告保留 2026-10-09 当时的状态，其未通过标记不能当作当前修订结论，当前通过记录也不追溯覆盖旧失败。

- [r3 修订与验收](revisions/native_20261012/README.md) · [r3 固定提交](https://github.com/aaash25/method-VIT-261009/tree/4d3e8e27f128c43ac0fab313e12627d10cde1619)
- [2026-10-09 固定提交](https://github.com/aaash25/method-VIT-261009/tree/2284df03820d4b88914d446b3e31955641de929c)
- [旧源码快照说明](SOURCE_CANDIDATE_README.md) · [旧模型方法索引](sources/MODEL_METHOD_INDEX.json)
- [旧完整项目快照](migration_sources/current_archive/) · [旧节点选集](migration_sources/extracted_components/)
- [旧运行限制](sources/MIGRATION_RUNTIME_LIMITS.md) · [旧公开清单](sources/PUBLIC_REPOSITORY_MANIFEST_20261009_r3.json)

## 许可与文件身份

许可按组件继承：[组件许可与版权](third_party/COMPONENT_NOTICES.md)。PiToMe CC BY-NC 4.0、DyMU MIT、IPCV/HF/LLaVA Apache-2.0、InternVL 底层 MIT；不为全仓统一重新授权。iLLaVA 作者完整算法沿用已有本地依赖边界。

模型权重、数据集实体、真实题干、私人路径、凭据和未筛选原始日志不进入公开仓库。
[当前公开文件清单](sources/PUBLIC_REPOSITORY_MANIFEST_20261012_r6.json) 对应本次 Git 文件内容，排除清单自身；旧清单保持其历史身份。
