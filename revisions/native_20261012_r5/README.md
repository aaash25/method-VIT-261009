# r5：恢复 Qwen iLLaVA 作者 FP16 舍入

Qwen 的合并行先执行作者 `.half()`，再由原来的索引赋值转换成宿主存储类型。FP32 下不再跳过这次 FP16 舍入。设备拼写和 eager 概率返回接口适配保留，选择、排序、权重、RoPE 和恢复规则不改；FP16 文本 QK32 保护完整保留。

服务器新 r5 发布的 FP16/FP32 均完成 3 原题 × 4 模式，共 **24 次完整自然 EOS 生成**。本地独立核验 journal、导出、激活、同请求双计时事件、Dense 恢复和源文件身份通过；所有输出与此前 GPU 验证的作者舍入绑定一致。这里是有界回归，不是全数据集性能结论。

仅 qwen_native 的 loader 绑定和源清单改变，新增作者 wrapper 与固定源依赖；其他模型和方法的文件保持 r4 字节。Smol 本轮仅轻量复核，生产代码未改。它的 FP32 直接 `.half()` 在 scatter 写回类型不兼容；`.half().to(x.dtype)` 已有真实 GPU 全请求通过证据，保留舍入而满足写回类型。

## 重建与运行

`materialize.py` 使用 FILE_MAP 将公共模板与既有本地私有作者文件重建为测试过的源文件。需以本次私有 r5 包作为 local-author-overlay，包含 6 个固定源文件。公共仓库不含模型、数据或作者完整算法源，模板本身不是独立可运行模型。

重建后的统一入口仍为 `run.py`，命令行参数保持 r4；它是有界回归入口，不自动启动正式全量。服务器已测试 r5 位于本次 QWEN_RESTORE_FA2_AUDIT_20261012 交付目录中的 native_release_20261012_r5。旧 r4 目录和 Git 历史完整保留。

原 ADAPTATIONS_20261012.md 的 r3/r4 文字作为历史来源保留；其中 Qwen iLLaVA 使用输入 dtype 的旧说明由本页恢复结论替代。其他模型的既有适配不变。

FA2 独立路线见 [Qwen FA2 FP16 预审](../../backends/qwen_fa2_fp16/README.md)，它没有使用本页 eager 回归冒充 FA2 验收。
