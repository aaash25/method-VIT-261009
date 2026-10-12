# VLM 视觉编码器方法迁移

当前修订为 **2026-10-12 原生方法修复版**。优先使用作者对应模型实现，去除额外稳定排序、强制确定性归约及计时路径中的诊断开销；保留并说明必要的接口、FP16/FP32 和显存适配。

服务器已完成四个模型、四种方法、两种精度的有界回归：**26 个任务、316 次完整自然结束生成通过**。这支持当前迁移和已知问题的回归验收，不代表全量数据集准确率或任何输入都不会报错。

## 当前入口

- [修订说明与统一运行入口](revisions/native_20261012/README.md)
- [16 个模型×方法的来源、适配和边界](revisions/native_20261012/METHOD_FIDELITY.md)
- [完整验收记录与逐请求双计时](revisions/native_20261012/VALIDATION_20261012.json)
- [公开模板与已测试代码的对应清单](revisions/native_20261012/FILE_MAP.json)
- [写入、导出与两小时恢复验证的历史修复](revisions/storage_repairs_20261011/README.md)

统一入口为恢复后的 `run.py`。服务器完整私有包已部署；公开仓库是包含 95 个模板文件的增量包，6 个含 iLLaVA 算法的文件继续使用固定哈希的本地依赖。仍需既有离线模型、数据集和冻结环境，不能直接把公开模板作为独立模型发行包运行。

Qwen FP16 文本 QK32 一直保留。此次误删的是另一处合并中间累加保护，现已恢复为 FP32 累加，并通过真实失败题回归。压缩输出遵循作者方法，不以复刻旧迁移的逐元素结果为目标。

每题使用自己的全部原图，顺序 batch size 1，不跨题复用 ViT。纯 ViT 与端到端计时来自同一次完整生成；图片附加验证按用户要求跳过。显存适配与写入修复均保留证据，但应用侧验证不等于确认共享存储历史 ENOSPC 的基础设施根因。

## 历史版本

本次新增修订目录并创建后继提交，未改写旧提交。下列目录和报告保留 2026-10-09 当时的状态，其未通过标记不能当作当前修订结论，当前通过记录也不追溯覆盖旧失败。

- [2026-10-09 固定提交](https://github.com/aaash25/method-VIT-261009/tree/2284df03820d4b88914d446b3e31955641de929c)
- [旧源码快照说明](SOURCE_CANDIDATE_README.md) · [旧模型方法索引](sources/MODEL_METHOD_INDEX.json)
- [旧完整项目快照](migration_sources/current_archive/) · [旧节点选集](migration_sources/extracted_components/)
- [旧运行限制](sources/MIGRATION_RUNTIME_LIMITS.md) · [旧公开清单](sources/PUBLIC_REPOSITORY_MANIFEST_20261009_r3.json)

## 许可与文件身份

许可按组件继承：[组件许可与版权](third_party/COMPONENT_NOTICES.md)。PiToMe CC BY-NC 4.0、DyMU MIT、IPCV/HF/LLaVA Apache-2.0、InternVL 底层 MIT；不为全仓统一重新授权。iLLaVA 作者完整算法沿用已有本地依赖边界。

模型权重、数据集实体、真实题干、私人路径、凭据和未筛选原始日志不进入公开仓库。
[当前公开文件清单](sources/PUBLIC_REPOSITORY_MANIFEST_20261012.json) 对应本次 Git 文件内容，排除清单自身；旧清单保持其历史身份。
