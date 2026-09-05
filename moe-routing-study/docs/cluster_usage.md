# 集群运行说明

Mac 交付的代码与 CPU 测试不代表 H20 / CUDA / TE / NCCL 实测通过。公开部署模板中的项目目录为 `/path/to/moe-routing-switch`，代码目录为克隆仓库中的 `moe-routing-study/`；公共模型目录只读绑定，训练模型的全部参数正常参与 AdamW。

公开文件中的路径与 `YOUR_PARTITION` 均需按自己的环境填写；它们不代表某个实际集群。修改实验 YAML 中的模型/数据路径、机器 YAML 中的项目/镜像路径和分区，以及 `launch/slurm.sh` 中对应字段后使用。

## 一次性环境与数据准备

在集群 Linux 上构建一个实际使用的运行产物：

```bash
mkdir -p /path/to/moe-routing-switch/environment
apptainer build --fakeroot /path/to/moe-routing-switch/environment/nemo-25.11.01.sif launch/nemo.def
```

配套目标是 NeMo 25.11.01、Bridge 0.2.0、Core 0.15、TE 2.9、PyTorch 2.9、CUDA 13.0.1。Mac 安装的 PyTorch 是普通 2.9.0 CPU/Mac 包。`nemo.def` 是 Linux 构建输入，不是本机已生成的 SIF；完整镜像大小需在实际构建中记录。方案为新增运行环境留 32 GB；若完整工具集超过这个数，在构建环境阶段移除无关组件与缓存后保留一个产物，不改变训练精度或优化器状态以腾空间。

使用该环境或登录节点的数据准备 Python 环境，执行：

```bash
PYTHONPATH=src python -m moe_study.data experiments/h20_qwen_real_update.yaml
```

需要 `datasets`、`transformers`、`huggingface-hub`、NumPy、PyTorch 和 PyYAML。数据准备命令解析 FineWeb-Edu revision 为真实 commit 后开始有限流式读取，输出 manifest、uint32 token 文件和文档索引；它在 GPU 作业之外运行。tokenizer 只读取本地 Qwen 源目录，不下载替代模型。

## 三段同轨迹运行

从代码目录按段提交，前一段完成后再提交下一段：

```bash
sbatch launch/slurm.sh experiments/h20_qwen_real_update.yaml machines/h20_64.yaml schedules/h20_three_segments.yaml A
sbatch launch/slurm.sh experiments/h20_qwen_real_update.yaml machines/h20_64.yaml schedules/h20_three_segments.yaml B
sbatch launch/slurm.sh experiments/h20_qwen_real_update.yaml machines/h20_64.yaml schedules/h20_three_segments.yaml C
```

三条命令不是要求同时执行。A 默认至 X255，B 默认至 X511，C 更新至 X512 并执行后续分析。恢复读取 native checkpoint 中的真实 step、AdamW、学习率进度、消耗序列数和 RNG。提前退出时下一段仍从实际 checkpoint 接续，不把名义起点当成已完成工作。最多三份默认分配，关闭自动 requeue。

H20 启动脚本的 `#SBATCH` 与 `h20_64.yaml` 对应同一固定布局：8 节点，每节点一个 Slurm task，task 内 8 个 torchrun rank，DP64 / EP8 / expert-DP8。启动脚本仅启动进程，不运行安装、Git 拉取、测试或主机探针。迁移机器时另写机器配置及对应资源申请脚本。

## 代码适配点

`adapters/megatron_qwen.py` 从 Bridge Qwen3 recipe 建配置，在创建分布式模型、初始化优化器之前用 `AutoBridge.load_hf_weights()` 导入源权重。恢复作业直接读取 Bridge checkpoint。显式保留 head_dim=128，因此 Q 投影宽 4096，KV 宽 512。

Core 的 `seq_aux_loss` 对每层添加一次辅助项；配置传入 `0.001 / 48`，再由正常微批次/DP 平均完成方案式 (5)。selected-softmax、无 capacity 丢弃、FP32 Router、全层重计算以及 BF16 主计算分别有直接配置映射。

生产反事实使用 Core `dispatch_preprocess → dispatch → routed_experts_compute → combine`，与正常层复用 TE grouped GEMM。局部参考在节点内 EP 汇集序列块，每个 rank 只计算自己 16 个专家，并还原 token 顺序。FP32 全网参考有独立参数内存，QKV 解包依照 Megatron 每个 KV 组对应 Q/K/V 的布局。

`MasterUpdate` 使用 Core 0.15 的 `model_param_gbuf_map`、`_get_model_param_range_map`、`_get_main_param_and_optimizer_states` 保存当前 rank 的 FP32 主参数片段。重构时普通参数在 DP64 组求和，专家参数在 expert-DP8 组求和。仅参考模型接收插值参数，训练参数与 Adam 状态不被改写。

这些 API 的版本与接口已按官方源码对应编写，跨卡 token 布局、TE 权重布局、模型导入、真实恢复及吞吐仍属于第一次集群业务执行中的适配工作。发生实际接口错误直接保留 traceback 并修复对应函数，没有替代精度或替代模型路径。

## 实際记录与后续整理

运行输出在 `outputs/qwen3_real_optimizer_update/`。运行配置、版本和未提交 patch、逐步梯度裁剪信息、每专家负载、主存与 GPU 峰值、训练与测量耗时都随结果保存。目录没有 Git 时仍正常运行并记录 unversioned；开发者可按普通 Git 流程提交修改。

时间预算由正常更新/完整测量块之间的判断与 Bridge 定时退出负责，预留最后 30 分钟保存。扫描按完整 α 块结束；完成点和未完成点均由进度文件与已有结果说明。资源文件中的磁盘退出占用是实测值，磁盘峰值未采样时明确写未采样，不把方案 983.9 GB 当成峰值实测。

```bash
PYTHONPATH=src python -m moe_study.report outputs/qwen3_real_optimizer_update
```

报告读取已存在的结果，缺失训练步或 α 点保持缺测。需要更多分配时另列新的实验安排。

## 对应官方源码

- [Bridge 0.2.0 Qwen3 配方](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.2.0/src/megatron/bridge/recipes/qwen/qwen3_moe.py)
- [Bridge Qwen3 MoE 转换](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.2.0/src/megatron/bridge/models/qwen/qwen3_moe_bridge.py)
- [Bridge 原生 setup](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.2.0/src/megatron/bridge/training/setup.py)
- [Core 0.15 Router](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.0/megatron/core/transformer/moe/router.py)
- [Core 0.15 分布式优化器](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.0/megatron/core/optimizer/distrib_optimizer.py)
