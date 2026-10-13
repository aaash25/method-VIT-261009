# LLaVA1.5-7B × DToMe/DyMU

当前版本：**2026-10-13，终稿（FINAL）**。仅此组合，FP16/V100 与 FP32/A800。用户确认；其他模型、方法和 FA2 路线未因此改为终稿。

## 对应实现

直接更新现有 [LLaVA DToMe 核心](../../../revisions/native_20261012_r6/templates/llava/author_dymu_core.py)、[HF CLIP 适配器](../../../revisions/native_20261012_r6/templates/llava/author_dymu_bridge.py) 和 [worker](../../../revisions/native_20261012_r6/templates/llava/worker.py)，不新增专属终稿目录。公开模板的私人部署根路径使用占位符；[现有重建入口和源清单](../../../revisions/native_20261012_r6/README.md) 逐文件核验实测字节。仍依赖既有离线权重、数据集、冻结基包及环境。

源包为新 r4，SHA-256 `5eeea6e7810b55c376905c13b20d65d2b444a11ead4a118a29a0f2adfa9e5028`，不是旧公开 r4 的 `f6e26714…1824`。本次发布未修改已测方法；相对先前公开版本只更新本组合的核心、回归测试和 bundle 哈希。旧版本由普通后继提交保留。

## 方法与必要适配

作者来源：[DyMU 固定实现](https://github.com/MikeWangWZHL/dymu/tree/3770d6add8fd9b972ae611f3f3b16a552dea13cc)。优先使用作者 OpenAI CLIP 核心，保持阈值、批内配对、合并权重、CLS 保护及位置追踪。HF Q/K/V 按原顺序转换为原生 MHA，沿用宿主 norm、MLP、激活及倒数第二层特征；恢复原视觉槽位以兼容 LLaVA 的占位符和连接器。这里是视觉方法迁移，不声明作者语言侧 Virtual Token Unmerging。

新包的 `dst.reshape(...).clone()` 防止批内 `scatter_add_` 原地修改共享视图，进而改写已保存层的位置追踪；选择、排序、权重和阈值不变。冻结核心文件头仍称 unmodified/no algorithm patches，该历史描述不能覆盖此修复披露。逐层统计取自作者 `output_stats`，计时结束后物化。原生 MHA 与 HF eager 的后端差异保留，不能把其全部归为合并收益。候选输出不以复刻旧迁移的逐元素或逐 token 结果为目标。

## 已验证

[验收记录](VALIDATION_20261013.json)：FP16、FP32 各 9 次完整自然 EOS 生成，共 **18 次**，服务器和下载后独立审计通过。覆盖 POPE/0、HRBench4K/0、MMMU_val/3414 原始五图题的 Dense、DToMe、恢复 Dense；每张卡各 8 项 CPU 组件检查通过。方法激活、Dense 恢复、journal CRC/导出、同请求 ViT/E2E 事件顺序通过。五图题每张图的 token 数在 24 层均不增加。

这是代表题回归，不是全数据集准确率或速度结论。无 OOM/dtype 推理错误，退出码均为 0；退出时禁止删除保护拦截临时文件清理的 PermissionError 保留，未声称日志完全无异常。附加图片验证按用户要求跳过。

重建现有公开包后，单独运行本组合使用 `llava/worker.py --dtype float16 --method dtome --output <新目录>`，FP32 对应 A800 并改为 float32；随后用 `audit_outputs.py --bundle llava --output <该目录>` 审计。使用已验证项目解释器与离线资源，先核实卡空闲和剩余时长。保持一题全部原图、请求 batch size 1、不跨题复用 ViT。不调用会运行四方法的 `llava/launch.py`，不自动启动正式全量实验。

许可 MIT，沿用 [组件版权说明](../../../third_party/COMPONENT_NOTICES.md)。[更新前版本](https://github.com/aaash25/method-VIT-261009/blob/467ac04d200c08579f4cc954b60878d1648d522e/models/llava15-7b/dymu/README.md) · [返回仓库导航](../../../README.md)。
