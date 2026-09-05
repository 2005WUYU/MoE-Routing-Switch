"""Generate CSV, PNG and Markdown from completed measurements, preserving missing points."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from moe_study.metrics import empirical_energy_envelope
from moe_study.statistics import adjacent_slopes


def summaries(root: Path):
    for path in sorted((root / "measurements").rglob("summary.json")):
        yield path.relative_to(root / "measurements").parts[:-1], json.loads(path.read_text())


def scalar(summary: dict, key: str):
    return summary["statistics"][key]["estimate"]


def number(value):
    return "未定义" if value is None else f"{value:.6g}"


def interval(summary: dict, key: str):
    result = summary["statistics"][key]
    bounds = result["interval"]
    return number(result["estimate"]) + (f" [{number(bounds[0])}, {number(bounds[1])}]" if bounds else "")


def build_report(root: Path, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    records = list(summaries(root))
    rows = []
    for parts, summary in records:
        for name, result in summary["statistics"].items():
            bounds = result["interval"] or [None, None]
            rows.append({"measurement": "/".join(parts), "metric": name, "estimate": result["estimate"],
                         "low": bounds[0], "high": bounds[1], "valid_positions": summary["valid_positions"],
                         "document_groups": summary["document_groups"]})
    with (destination / "measurements.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=["measurement", "metric", "estimate", "low", "high", "valid_positions", "document_groups"])
        writer.writeheader()
        writer.writerows(rows)
    run_files = sorted(root.glob("run_*.json"))
    run = json.loads(run_files[-1].read_text())
    experiment = run["config"]["experiment_file"]
    measurement = experiment["measurement"]
    text = ["# MoE 真实更新测量报告", "", run["purpose"], "",
            "区间条件于本条训练轨迹及本轮文本分布，采用文档组配对 bootstrap。数值误差单独报告；缺测点不补零。", "",
            "## 全层一步输出变化", "",
            "S 是其余更新项；J 是本层支持集变化能量；T=S+J+2C。J/T 不作为相加到 100% 的贡献比例。", ""]
    layer_records = [(parts, summary) for parts, summary in records if parts[0].startswith("step_") and len(parts) == 3]
    for step in measurement["large_steps"]:
        selected = [(parts, summary) for parts, summary in layer_records if parts[0] == f"step_{step:06d}"]
        if not selected:
            text.append(f"第 {step} 步：尚无测量记录。\n")
            continue
        figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for precision, style in (("execution", "--"), ("fp32_reference", "-")):
            values = sorted([(int(parts[-1].split("_")[-1]), summary) for parts, summary in selected if parts[1] == precision])
            for axis, metric in zip(axes.flat, ("S", "J", "C", "T", "switched", "r_h")):
                axis.plot([v[0] for v in values], [scalar(v[1], metric) for v in values], style, label=precision)
                axis.set(xlabel="MoE layer (1-based)", ylabel=metric)
                axis.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(f"Real optimizer update {step-1} -> {step}")
        filename = f"layers_step_{step}.png"
        figure.savefig(destination / filename, dpi=160)
        plt.close(figure)
        text.append(f"![第 {step} 步全部层]({filename})\n")
    text += ["## 全网任务损失", "", "单位 nats/token。V=L1−L10；PPL 比为 exp(V)。", "",
             "| 更新 | U（区间） | V（区间） | 总损失变化 | PPL 比 | 同状态重复损失差 |", "|---|---:|---:|---:|---:|---:|"]
    networks = {int(parts[0][5:]): summary for parts, summary in records if parts[0].startswith("step_") and parts[1:] == ("network",)}
    for step in measurement["large_steps"]:
        if step in networks:
            value = networks[step]
            text.append(f"| {step-1}→{step} | {interval(value, 'U')} | {interval(value, 'V')} | {interval(value, 'loss_change')} | {interval(value, 'ppl_ratio')} | {number(scalar(value, 'repeat_loss_difference'))} |")
        else:
            text.append(f"| {step-1}→{step} | 缺测 | 缺测 | 缺测 | 缺测 | 缺测 |")
    text += ["", "## 连续窗口", ""]
    window_networks = {int(parts[0][12:]): summary for parts, summary in records
                       if parts[0].startswith("window_step_") and parts[1:] == ("network",)}
    figure, axes = plt.subplots(len(measurement["consecutive_windows"]), 1, figsize=(10, 3 * len(measurement["consecutive_windows"])), squeeze=False, constrained_layout=True)
    for axis, (start, end) in zip(axes.flat, measurement["consecutive_windows"]):
        steps = list(range(start, end + 1))
        for metric in ("U", "V", "loss_change"):
            axis.plot(steps, [scalar(window_networks[step], metric) if step in window_networks else np.nan for step in steps], ".-", label=metric)
        axis.set(xlabel="Optimizer update", ylabel="nats/token", title=f"Window {start}–{end}")
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(destination / "windows.png", dpi=160)
    plt.close(figure)
    text += ["![窗口](windows.png)", "", "## FP32 全网 α 扫描", "",
             "α=0 的旧集合来自本次 FP32 全网本身。横轴保留零事件点；相邻斜率只对正能量且经验数值区间可区分的点计算。", ""]
    alpha_records = [(float(parts[0][6:]), int(parts[-1].split("_")[-1]), summary) for parts, summary in records
                     if parts[0].startswith("alpha_") and len(parts) == 3 and parts[1] == "fp32_reference"]
    layers = sorted(set(row[1] for row in alpha_records))
    slopes = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for layer in layers:
        values = sorted([(alpha, summary) for alpha, number_, summary in alpha_records if number_ == layer])
        alphas = [a for a, _ in values]
        energies = [scalar(s, "J") for _, s in values]
        envelopes = [empirical_energy_envelope(scalar(s, "J"), 0.0 if alpha == 0 else scalar(s, "fp32_error_rms")) for alpha, s in values]
        axes[0].plot(alphas, energies, ".-", label=f"Layer {layer}")
        axes[1].plot(alphas, [s["tail"]["events"] for _, s in values], ".-")
        slopes.extend({"layer": layer, **row} for row in adjacent_slopes(alphas, energies, envelopes))
    axes[0].set(xlabel="alpha", ylabel="J (FP32 network)")
    axes[1].set(xlabel="alpha", ylabel="Observed switch events")
    if layers:
        axes[0].legend(ncol=4, fontsize=6)
    figure.savefig(destination / "alphas.png", dpi=160)
    plt.close(figure)
    (destination / "alpha_slopes.json").write_text(json.dumps(slopes, indent=2))
    observed = sorted(set(row[0] for row in alpha_records))
    missing = [alpha for alpha in measurement["alphas"] if alpha not in observed]
    text += ["![α 扫描](alphas.png)", "", f"尚无结果的 α 点：{missing}。", "",
             "## 单层替换与同范数方向", "", "差值为替换后损失减实际新路径损失；随机向量与实际替换向量同位置、同符号约定。", "",
             "| 干预 | 损失差（区间） |", "|---|---:|"]
    random_means = []
    for parts, summary in records:
        if parts[0] == "interventions":
            text.append(f"| {' / '.join(parts[1:])} | {interval(summary, 'replacement_loss_difference')} |")
            if parts[-1].startswith("random_"):
                random_means.append(scalar(summary, "replacement_loss_difference"))
    if random_means:
        text += ["", f"随机方向均值 {number(float(np.mean(random_means)))}，范围 [{number(min(random_means))}, {number(max(random_means))}] nats/token。"]
    text += ["", "## 数值路径与资源记录", "",
             "本层 FP32 误差 RMS 由少量 FP64 输入估计，只表示经验数值不确定性。执行重复误差、FP32 误差样本数、边界三项、换入旧排名、尾部和事件文档组数均保存在原始 summary.json / groups.jsonl；完整列可见 measurements.csv。", ""]
    for parts, summary in records:
        if parts[0].startswith("endpoint_"):
            text.append(f"FP32 {parts[0]} 对生产端点的损失差：{interval(summary, 'fp32_minus_production_loss')}；支持集差异层均值：{interval(summary, 'route_disagreement_layer_mean')}。\n")
    resource_files = sorted(root.glob("resources_*.json"))
    for path in resource_files:
        text += [f"资源记录 `{path.name}`：", "", "```json", path.read_text(), "```", ""]
    if not resource_files:
        text.append("本目录没有 H20 显存、主存、磁盘峰值及作业耗时实测记录。\n")
    text += ["## 结论范围", "",
             "1. 真实函数变化：按各层 J、切换条件幅度、文档组区间与数值路径差异读取，不能仅凭编号变化下结论。",
             "2. 实际任务影响：按 α=1 的 U/V、绝对 nats/token、PPL 比及同状态数值差异读取；单层损失差不相加。",
             "3. 普遍失稳、独立训练重复及 μP 宽度迁移仍需后续实验。CPU 开发数据不支持本轮 H20 科学结论。", ""]
    (destination / "report.md").write_text("\n".join(text))
    return destination / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_report(args.run, args.output or args.run / "report"))


if __name__ == "__main__":
    main()
