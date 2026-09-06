# Mac 开发交付记录

## 2026-09-06 工程问题比较

新增 `moe_study.engineering` 与 `moe_study.engineering_report`，运行定义见 [engineering_chain.md](engineering_chain.md)。完整测试为 45 项，覆盖精确续训、收益抵消符号、不同长度文档加权、随机方向抽样不确定性、实际范数、原生接口后缀接续与双进程 EP 调用。第一步的主数据先保存，再执行方向计算。正式配置展开及两个新启动脚本的 shell 语法也已检查。

这些是 Mac 软件验证；本次未运行 H20 作业，也未验证新接口在实际 TE/Core GPU 内核下的数值分辨率和耗时。

## 原始开发记录

日期：2026-09-05。依据：上级目录《MoE_H20_真实更新测量实验方案_2026-09-05.md》及随附推导与数值验证。原始部署资料保留在本地。

## 本地环境与完成内容

本工程在独立 `.venv/` 中使用 Python 3.12.14、PyTorch 2.9.0、NumPy 2.5.2、pytest 9.1.1、PyYAML 6.0.3、Matplotlib 3.11.1。没有下载 Qwen 模型权重、FineWeb 正式训练数据或 GPU 容器。

已完成测量、统计、数据、状态与报告代码；可运行小尺寸完整 Qwen 参考网络，包含独立 head_dim、QK RMSNorm、RoPE、SwiGLU 和完整 selected normalization。CPU 入口执行真正的 AdamW 参数更新，实验规模与 H20 正式配置分别写在文件中。

已编写 Bridge 0.2.0 / Core 0.15 的模型导入、训练循环、逐序列辅助项映射、生产 dispatch 重放、EP 专家参考、主参数分片捕获和独立 FP32 网络参数转换；CUDA/TE/Bridge 只在集群入口导入。正式实验、H20 机器、三段日程和 Slurm 启动文件均已建立。

## 已执行的验证

`python -m pytest -q`：**28 passed**。测试围绕实际数学与状态语义，包括：

- 常数专家解析值、独立全专家求和、完整多专家交换、同支持集不同排列、当前分数重算旧集合门值。
- BF16 执行值先转 FP32 再做专家 GEMM，输出先转 FP64 再做差；S/J/C/T 与 U/V 恒等式和负交叉项。
- 固定旧边界专家对的有符号 margin、Router/上游/交叉三项，以及换入专家的真实旧排名。
- mask、uint32 移位标签、文档池分离、文档组加权、配对 bootstrap、零事件和未定义比值。
- 完整网络反事实重算下游输入；测量与随机方向不改变训练 RNG；完整 AdamW 恢复。
- QKV 分组解包保留所有 Q/K/V 头的顺序，Q 投影宽度可独立于 hidden size。
- 从大测量标量中选择预定窗口序列；α 扫描主体恰好 15 次完整网络前向；复用基准损失后的干预恰好 7 次完整前向。

`python -m compileall -q src` 和 `bash -n launch/slurm.sh launch/node.sh` 已执行。Mac 上的正式配置展开与测量清单入口也已执行，得到 4 次 1024 序列测量、62 次 64 序列测量，共 16,515,072 个序列位置观测；DP=64、expert-DP=8、每步 128 条序列。

## 开发示例结果

`outputs/mac-development/`：四次真实 CPU AdamW 更新，包含两次大测量、连续窗口、三种网络损失、全部八个 α 点、三个单层替换和四个同范数随机方向。示例报告为 `outputs/mac-development/report/report.md`，附 CSV、层图、窗口图和 α 图。已查看生成的层图，文字、曲线和布局正常。

`outputs/mac-segmented/`：A 执行前三次更新并保存，B 恢复完整状态完成第四次更新和测量。与连续运行相比，66 个参数 tensor 逐 bit 一致，保存的数据进度与 RNG 状态一致。单元测试另外验证了加载 AdamW 状态后继续更新的结果。

这些文件全部标注为 CPU 软件开发数据，不构成本轮 Qwen3/H20 的科学实验结果。图表没有将其冒充真实 48 层模型的训练观测。

## 集群执行边界

没有在 Mac 验证 TE/CUDA ABI、真实 Qwen 权重导入、64 rank dispatch/通信、分布式 checkpoint 恢复、H20 吞吐及显存峰值。这些工作在方案安排的集群实际业务执行中进行，入口和适配点见 `cluster_usage.md`。

运行代码不附加冻结、审批链、指标阈值、反复审计或自动降级。普通开发修改后按影响范围运行相关测试；训练启动脚本不自动运行测试。上述开发验证时目录尚未建立 Git 仓库，运行记录写为 unversioned，并保留小型源码快照；建立 Git 仓库后的运行同时记录 commit 和未提交 patch。
