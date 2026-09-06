# MoE Routing Study

按上级目录 `MoE_H20_真实更新测量实验方案_2026-09-05.md` 实现的 Mac 开发工程。正式配置是 Qwen3-30B-A3B、512 次全参数 AdamW 更新、全部 48 层的配对测量；Mac 示例使用小尺寸网络执行真实更新以验证软件。

512 步结果见[公开分析摘要](reports/h20_20260906/README.md)：存在性与首层尺度得到支持，核心工程损害假设尚未得到支持。新增 `python -m moe_study.causal` 从现有检查点接着执行一次更新，完成 12 次测量集前向。集群拉取后的具体命令见[一次更新补测运行说明](docs/causal_run.md)，比较定义见[工程假设与下一步因果比较](docs/工程假设与下一步因果比较_2026-09-06.md)。

## 在当前 Mac 上使用

项目虚拟环境 `.venv/` 已安装 PyTorch 2.9.0、NumPy、PyYAML、pytest 和 Matplotlib，已可运行：

```bash
source .venv/bin/activate
python -m pytest tests
moe-train examples/cpu_experiment.yaml examples/cpu_machine.yaml examples/cpu_schedule.yaml ALL --output outputs/mac-development
moe-report outputs/mac-development
```

CPU 示例包含四次 AdamW 更新、两次大测量与连续窗口、完整网络旧支持集比较、三个单层干预、四个随机方向和八个 FP32 α 点。其结果明确标为软件开发数据，不作为 H20 科学实验结果。

测试分段恢复可用同一输出目录依次运行 CPU schedule 的 A、B。B 从 A 的完整优化器和数据游标恢复：

```bash
moe-train examples/cpu_experiment.yaml examples/cpu_machine.yaml examples/cpu_schedule.yaml A --output outputs/mac-segmented
moe-train examples/cpu_experiment.yaml examples/cpu_machine.yaml examples/cpu_schedule.yaml B --output outputs/mac-segmented
```

新 Mac 的普通安装方式：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

正式配置可以在 Mac 展开查看，不加载 Megatron/CUDA：

公开配置中的模型/数据/项目/镜像路径为 `/path/to/...`，分区为 `YOUR_PARTITION`；实际部署时填写自己的环境值。原始集群资料不随仓库发布。

```bash
moe-train experiments/h20_qwen_real_update.yaml machines/h20_64.yaml schedules/h20_three_segments.yaml A --print-config
moe-measure experiments/h20_qwen_real_update.yaml
```

## 代码布局

| 文件 | 职责 |
|---|---|
| `routing.py`、`metrics.py` | 完整 Top-K 并集重放、selected normalization、S/J/C/T、边界三项、U/V |
| `statistics.py` | 文档组配对 bootstrap、条件幅度、尾部与可分辨斜率 |
| `data.py` | 有限流式分词、文档划分、uint32 打包、mask 与游标索引 |
| `measure.py`、`scan.py` | 更新两侧捕获、三条网络路径、局部数值参考、α 与干预 |
| `causal.py`、`causal_measure.py`、`adapters/causal_train.py` | 从检查点接着走一步，独立重复、捕获后重放与任务效应报告 |
| `reference/` | SwiGLU 及保留 QK Norm、独立 head_dim 的完整 FP32 Qwen 参考 |
| `train.py`、`state.py` | CPU 开发训练、完整状态恢复、命令入口 |
| `adapters/` | Bridge/Core 训练接口、生产 dispatch、EP 参考与主参数分片 |
| `report.py` | Markdown 报告、CSV 和静态图表 |
| `experiments/`、`machines/`、`schedules/` | 三份显式配置，实验定义、机器条件、作业拆分分别列出 |
| `launch/` | H20 Slurm/torchrun 启动与 Linux 镜像构建说明 |

[测量定义](docs/measurement_definition.md) 说明公式、数值路径和文件格式；[集群使用说明](docs/cluster_usage.md) 说明版本映射、数据准备、三段运行及尚需集群实际执行的部分；[Mac 开发记录](docs/mac_development.md) 保存本次实际验证情况。

测试用于相关代码开发，不嵌入提交或训练启动链。工程没有模型冻结、指标阈值放行、自动降精度、自动跳过 batch、自动重排队或反复主机探测逻辑。
