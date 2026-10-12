# 方法来源与保留适配

判定范围为视觉分支的方法迁移：保持作者的打分目标、排序方向、选择范围、合并权重、发生位置和重建规则。允许有明确理由的宿主接口和精度适配；不要求压缩结果与旧迁移版本逐元素或逐 token 相同。这里不声明复现论文中所有语言侧压缩，也不把有限回归解释为所有输入等价。

固定来源：PiToMe `550b5deed94aadfeac28bfbe381d87e672044a40`；DyMU `3770d6add8fd9b972ae611f3f3b16a552dea13cc`；iLLaVA `d2d291877ba9c7d4781cdc9ed1820661c8e47291`；IPCV `29726f044f0f9c0c2be4eb0005888ddeb5fa188e`。来源链接和组件许可沿用仓库既有记录。

| 模型 | 方法 | 当前实现与必要适配 |
|---|---|---|
| Qwen2-VL | iLLaVA | 固定作者 Qwen 块；FP32 合并结果使用输入 dtype，替代作者硬编码 half。 |
| Qwen2-VL | PiToMe | 迁移作者选择、配对、排序、合并规则；保留原图分段、代表位置 RoPE 和 PatchMerger 槽位恢复。原生匹配 dtype；FP16 合并中间累加用 FP32 防止已实测溢出。 |
| Qwen2-VL | DToMe | 保留 DyMU 单实例/批内分支与 epsilon；一题的多个 ragged 图像段按作者批内语义处理。固定阈值时的全局排序计数与逐段保留所有过阈值边逻辑对应；不声称推广到另有全局预算上限的变体。FP16 加权累加用 FP32 防溢出。 |
| Qwen2-VL | IPCV | 作者原生 Qwen 全局选择与近邻重建；修正半开图像区间的端点归属，并适配现有 Transformers 的分段 attention、缓存和生成接口。 |
| SmolVLM2 | iLLaVA | SigLIP 规则迁移：保持 attention 两次均值次序、低重要性选择与递减权重；合并结果使用输入 dtype。 |
| SmolVLM2 | PiToMe | 作者 no-CLS HF 分支规则；BSM 零删除仍按作者顺序重排；energy 零删除的未定义空集合操作保留显式输入域保护。 |
| SmolVLM2 | DToMe | 作者 B=1 scatter-reduce、B>1 scatter-add 规则；保留原生 crop 批、阈值、padding、epsilon 和重建。已消费 key 及时释放，诊断默认不进入计时路径。 |
| SmolVLM2 | IPCV | InternVL 风格逐 crop 选择、跨 crop 全局近邻及 attention stabilization；保留作者近邻范围。 |
| LLaVA-1.5 | iLLaVA | 固定作者来源中未找到对应 CLIP 压缩实现，采用 SigLIP 规则迁移；保护 CLS、使用宿主倒数第二层、恢复原槽位。FP32 合并行保留作者 .half() 的一次舍入；CUDA 组件与完整生成通过。 |
| LLaVA-1.5 | PiToMe | 固定作者 HF-CLIP 原生实现；保留紧凑视觉输出和原生占位符适配。 |
| LLaVA-1.5 | DToMe | 固定作者 OpenAI CLIP 核心；转换 HF QKV 权重，沿用宿主 norm/MLP，并保留 CLS、特征层和恢复接口。原生 MHA 与 HF eager 的后端差别必须披露，不将其全部归为合并收益。 |
| LLaVA-1.5 | IPCV | InternVL 风格 CLIP 迁移；保护 CLS、逐图选择、全局保留池近邻，保持搜索范围。 |
| OneVision | iLLaVA | 固定作者 SigLIP attention/layer；FP32 保留作者合并行 .half() 舍入；槽位恢复及宿主原生 packing 保留。 |
| OneVision | PiToMe | no-CLS PiToMe 规则迁移；保留 BSM 重排与 energy 输入域保护，移除额外强制确定性调度。 |
| OneVision | DToMe | 固定作者 OneVision `tome_encoder.py` 与 DyMU 合并函数；保留 crop 批分配、单实例/批内分支和零边重排，恢复原槽位后使用宿主 packing。 |
| OneVision | IPCV | 作者 Qwen 全局 top-k / NGR 规则迁移；attention 按 crop 分段，选择与近邻重建的全局范围保持不变。 |

Qwen 文本 QK32 是独立的 FP16 防溢出适配，用户已明确接受，当前继续保留。它不等同于视觉路由升精度，也不等同于合并中间累加升精度。

Smol 的 key 清理和文本 attention 临时矩阵处理、OneVision V100 的逐 KV-head 组原生数学 SDPA，用于处理已复现的内存问题。后者保持数学操作但会改变浮点调度和端到端成本，Dense 与候选使用同一适配，不算作 ViT 方法的加速收益。

当前源码中的部分历史文件名、docstring 和 `qwen_native` 回执的旧归约描述沿自冻结依赖；它们不代表额外启用了迁移 core 的 FP32 确定性归约。实际 native 执行路径及本表是本次方法说明，原有记录不追溯重写。

最终运行覆盖以独立验收文件为准。原始同步双计时用于回归诊断；不据此给出全量准确率、性能置信区间或所有请求不报错的保证。应用侧写入与恢复验证也不能证明共享存储服务历史 ENOSPC 的后端根因已解决。


r4：Qwen IPCV eager attention 按原图分段，保留作者全局选择与近邻范围；未使用的 attn_scores 返回 None。Smol/OneVision 移除计时路径的重复 host-mask 扫描；LLaVA DToMe 从原生 output_stats 记录逐层计数。详细理由见 templates/ADAPTATIONS_20261012.md 的 r4 表。该文件中的 GPU pending 是源码冻结时状态，当前状态以本目录独立验收记录为准。
