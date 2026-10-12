# Qwen FA2 FP16：全方法路线预审

本轮按用户最新指令只预审，不启动全量。目标保留四方法、原 15 个冻结数据集及既有策略，随后换模型交叉审核，再决定正式运行。FA2 是单列实验路线；不能称为备用版，也不能拿 eager 的验收覆盖 FA2。

## 判断标准

目标模型有作者原生实现时优先执行它；否则迁移需保持逻辑等价。允许设备、生成接口、模型边界和有证据的精度适配，逐项披露。压缩输出允许随作者方法和后端改变，不要求复刻历史迁移的逐元素结果。保留每题全部原图，问题 batch size 1，不跨题复用 ViT。额外图片校验仍按用户要求跳过。

## 当前逐方法结论

| 方法 | 已有内容 | 本次审核结论 | 交叉审核后仍需完成 |
|---|---|---|---|
| IPCV | 作者原生 Qwen vision/text FA2；既有 A800 whole-runner smoke 6 条完整输出通过；15 数据集清单及队列已准备 | 原生路线有来源和历史实测支持。旧计时协议没有当前同一完整请求的双计时，不能直接作为新正式入口 | 使用当前双计时边界的精确发布 smoke；多图压缩区间、RoPE、恢复及 Dense 同后端检查；新队列预检 |
| iLLaVA | 固定作者 Qwen `VisionFlashAttention2.forward` 和 `Qwen2VLVisionBlock.forward`；本次 A800 单／五图分段组件执行通过 | 可直接保留作者 FA2 分支和 `.half()`。它用 FA2 tuple[1] 的 `softmax_lse` 排序，而 eager 用输入注意力概率统计；这是作者原生代码差异，不能自行“修正”为 eager 排序 | 全模型 FA2 loader 与原生未压缩层的绑定；单／多图完整生成、禁用／Dense 恢复及双计时审核；确认记录的 FA2 版本和作者依赖 |
| PiToMe | r4 的 matching/merge/restoration 已审核；Qwen port 通过 `_native_block` 调用宿主 block | 算法不依赖显式注意力概率，有迁移到宿主 FA2 的基础。当前共享 loader 固定 eager，没有可宣称通过的完整 FA2 入口 | 新 FA2 loader、实际 kernel dispatch、同后端 Dense 和完整请求验证 |
| DToMe | r4 `_fused_attention` 显式计算 `QK/sqrt(d)+log(size)` 后 softmax；保留作者 proportional attention | 当前候选仍为 eager 视觉注意力。把模型配置设成 FA2 不会改变该路径；不能去掉 `log(size)`、改普通 FA2 后仍称忠实 | 保留 proportional attention 的逻辑等价实现或明确混合后端范围；交叉审核前不选未经验证的改写方案 |

## 实际准备范围

服务器已有 `qwen_ipcv_fa2_smoke_20261006_r1`、`qwen_ipcv_fa2_formal_prepare_20261006_r1`、`qwen_ipcv_fa2_queue_r2_20261007`，以及 FA2 2.7.3 独立离线 overlay。基本环境仍为 Torch 2.5.1+cu121／Transformers 4.57.6，没有向冻结主环境安装新包。旧 IPCV 队列及失败 eager 记录保留，未启动或续跑它们。

原 15 数据集为 HRBench4K、POPE、MMVP、RealWorldQA、MMStar、MMMU_val、ScienceQA_image_test、MMBench_EN_dev_v1、TextVQA_val、OCRBench、MME、ChartQA、CVBench2D、CVBench3D、AI2D。IPCV 既有清单为 32,338 原始问题、Dense/keep85/keep70；这只说明准备的覆盖范围，不是全量结果。

FA2 标准 CUDA 路径支持 Ampere/Ada/Hopper 和 FP16/BF16；本路线仅 FP16，使用 A800。V100 保留 eager 测试用途。来源：[FA2 官方说明](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)。FA2 2.7.3 的返回顺序见[固定版本接口](https://github.com/Dao-AILab/flash-attention/blob/v2.7.3/flash_attn/flash_attn_interface.py)。iLLaVA 原生分支见[固定作者源码](https://github.com/hulianyuyy/iLLaVA/blob/d2d291877ba9c7d4781cdc9ed1820661c8e47291/src/transformers-4.46.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)。

## 交叉审核应明确的结论

1. iLLaVA 保留作者 FA2/LSE 分支，而非改成 eager 统计，是否符合本项目作者原生优先标准；不能因输出不同就拒绝。
2. DToMe 采用什么逻辑等价的 proportional attention 路径，以及如何标记实际视觉／语言后端；不能把配置名当 kernel 证据。
3. 精确发布的当前同请求双计时、独立多图边界和全生成验收是否齐备。
4. GitHub 的 FA2 栏目是否将“已准备／组件通过／whole-runner 通过／全量结果”分开，是否误用历史 smoke 支持新发布。

当前结论：部分原生准备符合标准，但**四方法完整 FA2 发布尚未完成验收**。不批准也不启动全量。这里列出的缺口来自实际代码和协议，不追加阈值搜索、额外图片检查或逐元素性能约束。

## Smol 轻量复核

FP32 不兼容位于 `pooled.half()` → `output.scatter_`：目标是 FP32，来源是 FP16。Qwen 和其他作者路径的索引赋值会转换为目标 dtype，而 `scatter_` 要求来源与目标同 dtype。现有 GPU 的真实 MMMU_val 3414 五图请求复现该错误；`.half().to(x.dtype)` 在两精度完整请求均通过，并保留 FP16 舍入。本次 20 个轻量 CPU 组件再次通过，包含两个预期失败，0 跳过；Smol 生产源码未改。此前通过的五图完整请求峰值显存约 FP16 18.22 GiB、FP32 24.02 GiB，错误本身不是 OOM，不能由模型权重大小推断整请求占用低于 10 GB。
