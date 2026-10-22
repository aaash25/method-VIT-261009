# 环境版本与适用范围

本说明记录环境来源，不改变运行中的环境、冻结算法、旧结果或验收标签。

## 本仓库32项同行迁移的核心约束

终稿为 `eager-final-20261013`，提交 `73b8bbd9d7baf6c880d153fcbd5698dedc2c5309`。应从主README的当前终稿入口判断适用版本，而不是把其他仓库的 requirements 或历史目录直接作为当前实验规范。

已公开的 LLaVA/Qwen 原生运行代码要求 PyTorch `2.5.1+cu121`、Transformers `4.57.6`；LLaVA 还明确要求 Python `3.11`。原验收使用 eager 路线、关闭 TF32，并保留各原生方法的数值设置。代码终稿验收不表示全量正式实验已经完成。

- [LLaVA核心环境门控](migration_sources/current_archive/releases/llava_r7_fidelity_r5_20261001_r1/route/hypotheses/v611_h303_llava15_five_method_fp32_a800/llava_runtime.py)
- [Qwen核心环境门控](migration_sources/current_archive/releases/qwen_minimal_peer_reuse_r6_20261001/source/method_runtime.py)
- [当前终稿说明](revisions/native_20261012_r6/README.md)

## 两套已记录环境

| 项目 | 原H302/H303 LLaVA验收 | 新平台准确率任务参考环境 |
|---|---|---|
| PyTorch | 2.5.1+cu121 | 2.5.1+cu121 |
| Transformers | 4.57.6 | 4.57.6 |
| torchvision | 0.20.1+cu121 | 0.20.1+cu121 |
| NumPy | 2.4.6 | 1.26.4 |
| Pillow | 12.3.0 | 10.2.0 |
| Accelerate | 1.14.0 | 1.10.1 |
| Python | 3.11.16 | 参考3.11.8；后续实例3.11.10/3.11.11 |

完整已记录版本分别见 [原LLaVA环境](environments/llava_h302_h303_legacy.json) 与 [新平台22项参考版本](environments/accuracy_new_platform_20261019.json)。原LLaVA锁只适用于该原验收环境，不替其他模型虚构统一完整依赖锁。新平台参考描述该次用户授权的环境对齐，也不反向改写原LLaVA冻结环境。

两者符合上述核心版本约束，但完整环境并不相同。辅助依赖不同本身不能证明旧结果错误、新结果发生偏移，或输出一定相同。原H302四种方法的Dense完整tokens互相一致，只能支持该批原结果内部的一致性，不能单独证明跨环境等价。

## 结果复用与变更

共享Dense时保留原模型/修订、原始输入、精度、数值栈及输出来源；不能把历史环境重标为新平台环境。复用资格仍按具体原生准入和证据核对，不能仅凭相同Torch/Transformers就忽略其他已知差异。

发现来源差异，先核对适用环境规范和实际记录；不能直接推导“旧环境有问题”，也不能因此自动重算整批已完成记录。正在运行的任务保持不动；必要的环境适配、分卡与后续验证按该次授权另行留痕。

本仓库的实验约束不由其他部署仓库的依赖清单替代。这里没有升级到另一发布中的Torch版本，也没有修改任何运行任务。
