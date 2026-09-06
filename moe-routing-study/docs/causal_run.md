# 从现有检查点运行一次路由因果补测

本文记录上一版 12 次前向诊断。当前主入口为 `moe_study.engineering`，请使用[三步工程实验链及运行说明](engineering_chain.md)。

新增入口为 `python -m moe_study.causal`。它恢复已有优化器、主参数、数据游标与 RNG，正常执行一次更新，然后比较两种实现中各六次真实前向。对于当前 X512，目标是 X513。配置、模型、数据和运行镜像继续使用集群上已经跑通的版本。

## 集群拉取与提交

先在仓库根目录更新代码，再进入代码目录：

```bash
git pull --rebase --autostash
cd moe-routing-study
```

集群的实际路径配置留在本地。`launch/causal_slurm.sh` 使用两个环境变量：运行镜像 `MOE_RUNTIME_IMAGE`，以及需要映射到容器中的共享存储路径 `MOE_BIND_PATHS`。后者采用 Apptainer 的逗号分隔 bind 格式；包含模型、数据、源检查点和新输出所在的共享目录。

下面的路径换成该集群已有的值。实验 YAML、机器 YAML 可以继续使用已填写好路径的现有文件。

```bash
export MOE_RUNTIME_IMAGE=/path/to/the-existing-working-runtime.sif
export MOE_BIND_PATHS=/path/to/shared-storage:/path/to/shared-storage

sbatch --partition=YOUR_PARTITION launch/causal_slurm.sh \
  experiments/h20_qwen_real_update.yaml machines/h20_64.yaml \
  --checkpoint /path/to/previous-run/checkpoints \
  --output /path/to/new-run/qwen3_causal_512_513
```

`--checkpoint` 指向含 `latest_checkpointed_iteration.txt` 的 **checkpoints 根目录**，不是 `iter_0000512` 子目录。`--output` 使用新的结果目录。该入口读取源检查点，不在源目录保存、轮换或删除文件，也不创建第二套 Adam 检查点。

启动脚本申请 8 节点、每节点 8 卡，默认时间 2 小时 30 分。可通过 `sbatch --time=...` 指定本次分配时间；程序按固定工作量顺序执行，不做指标放行、自动缩样本或自动重排队。它不会重复原来的 512 步训练，也不会运行 α 扫描或七组随机/单层干预。

默认 1,024 条主序列，详细诊断使用其中前 64 条完整长度序列。为让每个 EP collective 都有对应调用，这 64 条序列排在测量顺序最前面，再分给各 rank；原始 sequence ID、文档 ID、valid mask 均保留。当前 64 卡配置的主序列数和诊断数使用 64 的倍数，且诊断数不超过选入主样本的完整长度序列数。

不需要重新安装 Python 包；启动器设置 `PYTHONPATH` 后，直接从拉取的 `src/` 运行。代码已经纳入上一轮在集群实际使用的三处兼容性修改：`dataloader_type=single`、关闭 permute fusion、关闭周期检查点保存。机器路径和镜像信息没有写入公共代码。

## 这次实际执行什么

| 阶段 | 内容 |
|---|---|
| 恢复 X512 | 完整原生训练状态；学习率规则沿用原配置 |
| 旧状态 | BF16 的 N0 两次；FP32 主参数全网的 N0 两次 |
| 真实更新 | 原训练 batch、全参数 AdamW，一次 optimizer step |
| 保存端点 | 各 rank 保存自己持有的 X513 FP32 主参数分片 |
| 新状态 | 每种实现、每条序列按 N1₁→F1₁→F1₂→N1₂ 执行 |
| 汇总 | 文档组配对区间、真实重复差、logits 差、诊断标量和 Markdown 报告 |

N1 是自然新路由，F1 是相同新参数在旧支持集上的完整前向。F1 使用该实现首次 N0 的支持集，两次重复不替换它。每次前向前恢复同一 RNG 状态，前向后恢复外部训练 RNG；旧支持集中的门权重用当前分数重新计算。

主前向期间只捕获实际 gating 的分数和必要张量，不调用额外的专家重放。FP32 本层重放在主前向之后执行。BF16 反事实沿用 Core 的降序分数 softmax 顺序，避免把按专家 ID 排序造成的额外舍入混入门权重比较。

FP32 精度本身不保证重复一致；原生 BF16 的归约仍可能带来执行差。程序保存这些读数，不以阈值中止测量，也不从效应中扣除噪声。两次重复是本次运行的直接观测，不能当作完整的执行方差估计。

## 结果文件

新目录中的 `report.md` 和 `effects.json` 是阅读入口。每种实现的 `network/summary.json`、`groups.jsonl`、`tokens.npz` 保存统计与逐位置数据。

| 文件或目录 | 内容 |
|---|---|
| `run.json` | 来源状态、样本顺序、实现定义、实际配置与源码记录 |
| `training.json` | 一步训练的损失、梯度、学习率、专家负载和原生 skipped 标记 |
| `bf16_execution/network/` | BF16 两次 V、U、总差、三种条件重复差、KL、中心化 logits RMS |
| `fp32_reference/network/` | 独立 FP32 路径的同一组指标 |
| `*/diagnostics/sequence_*.npz` | 诊断子集的全层支持集、margin、输入/输出重复差、本层 J 和原始残差流能量 |
| `*/forward_timing.json` | 各 rank、序列、条件、重复的实际前向时间 |
| `new_master_shards/rank_*.pt` | X513 的 FP32 主参数分片、分片范围与原执行 dtype；约 122 GB 十进制量级 |
| `completion.json` | 最终步、skipped 标记、完成的两种实现、耗时和显存记录 |

FP32 新端点按分片保存，不包含新的 Adam 一、二阶状态，因此不是可直接继续训练的完整检查点。源 X512 仍保留完整训练状态。重放 X513 权重时，根据 `run.json` 中的 EP/DP 组与各分片范围还原；BF16 执行权重由主参数转换到记录的执行 dtype。

完整 logits 仅在内存中保留当前序列的比较条件，按 token 块计算双精度 KL 和去词表均值后的 RMS，计算完释放，不写入结果文件。诊断数组保留原序列位置；分析必须应用 `valid`。

下载分析结果时，排除新的主参数分片即可：

```bash
tar --exclude='./new_master_shards' \
  -C /path/to/new-run/qwen3_causal_512_513 \
  -czf /path/to/causal_analysis.tar.gz .
```

该分析包包含本地运行配置和源码快照，供私下分析；公开研究摘要另放在 `reports/`。

## Mac 开发验证

可以使用已有 CPU 示例生成开发检查点，再运行同一个补测控制器。CPU 输出明确标为软件验证数据。

```bash
.venv/bin/python -m moe_study.train \
  examples/cpu_experiment.yaml examples/cpu_machine.yaml examples/cpu_schedule.yaml ALL \
  --output outputs/cpu-causal-source

.venv/bin/python -m moe_study.causal \
  examples/cpu_experiment.yaml examples/cpu_machine.yaml \
  --checkpoint outputs/cpu-causal-source/checkpoints \
  --output outputs/cpu-causal-next \
  --sequences 8 --diagnostic-sequences 2
```

针对性测试覆盖：真正执行六次前向、相同旧支持集复用、RNG/tracker 恢复、无更新时零效应、修改参数后的非零事件、精确主参数端点、原生 gating hook、两进程专家并行与完整模型的一致性，以及恢复后的一步 AdamW 更新。Mac 上不具备 H20/Transformer Engine/Bridge 运行环境，因此 GPU 内核行为和实际耗时由这次集群执行给出。
