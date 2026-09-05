# MoE Routing Switch

研究硬 Top-K MoE 中，真实优化器更新引起的路由集合变化、层输出变化和语言建模损失变化。

本轮实验采用现有 Qwen3-30B-A3B 权重进行全参数继续训练，计划在 64 张 H20 上执行 512 次 AdamW 更新，并测量全部 48 个 MoE 层。当前已完成 Mac 端开发和 CPU 软件验证；H20 实际训练与跨卡验证尚未执行。

## 仓库内容

- [实验方案](MoE_H20_真实更新测量实验方案_2026-09-05.md)：研究定义、配置、测量安排及资源预算。
- [工程与使用说明](moe-routing-study/README.md)：训练、配对测量、统计、FP32 参考、状态恢复与报告代码。
- [Mac 开发记录](moe-routing-study/docs/mac_development.md)：28 项通过的测试、完整 CPU 示例和分段恢复结果。
- [集群使用说明](moe-routing-study/docs/cluster_usage.md)：数据准备、Bridge/Core 适配与 Slurm 三段运行。
- [研究推导与原有数值验证](docs-2026-09-05/00_总览与阅读顺序.md)。

部署配置使用 `/path/to/...` 和 `YOUR_PARTITION` 占位值。原始集群资料与本地部署副本不纳入公开仓库。

## Mac 开发

```bash
git clone https://github.com/2005WUYU/MoE-Routing-Switch.git
cd MoE-Routing-Switch/moe-routing-study
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests
moe-train examples/cpu_experiment.yaml examples/cpu_machine.yaml examples/cpu_schedule.yaml ALL --output outputs/mac-development
moe-report outputs/mac-development
```

CPU 示例仅用于软件开发，不作为 Qwen3/H20 科学实验结果。生成结果、虚拟环境、检查点和模型权重不纳入 Git；配套文档中已有的小规模解析验证结果保留在仓库中。

开发遵循普通 Git 流程，不添加防御性编程、门禁、反复审计或冻结流程。
