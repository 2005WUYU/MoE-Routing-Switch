"""Task harm and direction specificity at one actual optimizer update.

Full-network counterfactuals define U and V. Separate suffix interventions start
from a common old-route representation and compare +D with antithetic Gaussian
directions. These are different estimands and are never added across layers.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from moe_study.causal_measure import ForwardCapture, LayerCapture
from moe_study.engineering_statistics import cancellation_description, crossed_summary, resolution_projection
from moe_study.measure import batch


@dataclass
class OldTask:
    losses: np.ndarray
    sham: np.ndarray
    supports: dict


def gaussian_direction(delta, seed):
    """Match each token's norm in FP64; zero route changes give zero injections."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(delta.shape, generator=generator, dtype=torch.float64)
    return direction / direction.norm(dim=-1, keepdim=True) * delta.double().norm(dim=-1, keepdim=True)


class DocumentSums:
    """Keep [execution, direction] sums per document, merging its sequences."""
    def __init__(self):
        self.rows = {}

    def append(self, sample, columns):
        valid = sample["valid"].numpy().astype(bool)
        group = sample["group"]
        values = {name: value[..., valid].sum(-1) for name, value in columns.items()}
        if group in self.rows:
            row = self.rows[group]
            row["N"] += int(valid.sum())
            for name, value in values.items():
                row["columns"][name] += value
        else:
            self.rows[group] = {"N": int(valid.sum()), "columns": values}

    @staticmethod
    def merge(parts):
        pooled = {}
        for part in parts:
            for group, incoming in part.items():
                if group in pooled:
                    row = pooled[group]
                    row["N"] += incoming["N"]
                    for name, value in incoming["columns"].items():
                        row["columns"][name] += value
                else:
                    pooled[group] = incoming
        groups = sorted(pooled)
        counts = np.array([pooled[group]["N"] for group in groups])
        columns = {name: np.stack([pooled[group]["columns"][name] for group in groups], axis=-1)
                   for name in pooled[groups[0]]["columns"]}
        return groups, counts, columns


class EngineeringMeasurement:
    def __init__(self, samples, diagnostic_ids, output, engine, rng, device="cpu", logit_block_size=32,
                 announce=True, *, protocol):
        self.samples, self.output, self.engine = samples, Path(output), engine
        self.rng, self.device, self.announce = rng, device, announce
        self.protocol = protocol
        self.old, self.timings = {}, []
        self.task = DocumentSums()
        self.directions = {number: DocumentSums() for number in protocol["direction"]["layers"]}

    @torch.no_grad()
    def forward(self, model, sample, condition, repeat, supports=None):
        layers, modules = {}, {}

        def observe(number, layer, residual, hidden, scores, chosen, output):
            if number in self.directions:
                layers[number] = LayerCapture(*[value.detach().cpu().clone() for value in
                    (residual, hidden, scores, chosen.to(torch.int32), output)])
                modules[number] = layer

        previous_mode = model.training
        started = time.perf_counter()
        with self.rng.replay():
            model.eval()
            try:
                result = model.causal_forward(**batch(sample, self.device), supports=supports,
                                             diagnostic=observe, diagnostic_layers=list(self.directions), return_logits=False)
            finally:
                model.train(previous_mode)
        result.losses = result.losses.detach().cpu().reshape(-1).double()
        result.logits = None
        self.timings.append({"sequence": sample["sequence"], "condition": condition, "repeat": repeat,
                             "seconds": time.perf_counter() - started})
        return ForwardCapture(result, layers, modules)

    def capture_old(self, model):
        for index, sample in enumerate(self.samples):
            losses, sham = [], []
            for repeat in range(self.protocol["task_execution_repeats"]):
                natural = self.forward(model, sample, "N0", repeat)
                fixed = self.forward(model, sample, "sham_old", repeat, natural.network.supports)
                if repeat == 0:
                    supports = natural.network.supports
                losses.append(natural.network.losses.numpy())
                sham.append((fixed.network.losses - natural.network.losses).numpy())
            self.old[sample["sequence"]] = OldTask(np.stack(losses), np.stack(sham), supports)
            self.progress("old", index + 1)

    def measure_new(self, model, gather=lambda value: [value]):
        directions = []
        for index, sample in enumerate(self.samples):
            old = self.old.pop(sample["sequence"])
            natural_losses, fixed_losses, sham = [], [], []
            for repeat in range(self.protocol["task_execution_repeats"]):
                # Alternate F1 around N1; own-support sham always follows its N1.
                if repeat % 2:
                    fixed = self.forward(model, sample, "F1", repeat, old.supports)
                natural = self.forward(model, sample, "N1", repeat)
                own = self.forward(model, sample, "sham_new", repeat, natural.network.supports)
                if repeat % 2 == 0:
                    fixed = self.forward(model, sample, "F1", repeat, old.supports)
                if repeat == 0:
                    captured = natural
                natural_losses.append(natural.network.losses.numpy())
                fixed_losses.append(fixed.network.losses.numpy())
                sham.append((own.network.losses - natural.network.losses).numpy())
            n1, f1 = np.stack(natural_losses), np.stack(fixed_losses)
            columns = {"L0": old.losses, "L1": n1, "L10": f1, "U": f1 - old.losses,
                       "V": n1 - f1, "loss_change": n1 - old.losses,
                       "sham_old": old.sham, "sham_new": np.stack(sham)}
            for name in ("U", "V", "loss_change"):
                columns[f"{name}_positive"] = np.maximum(columns[name], 0)
                columns[f"{name}_negative"] = np.maximum(-columns[name], 0)
            self.task.append(sample, {name: value[:, None, :] for name, value in columns.items()})
            self.save_tokens("task", sample, columns)
            if sample["sequence"] < self.protocol["direction"]["sequences"]:
                directions.append((sample, captured, old.supports))
            self.progress("new", index + 1)
        # Persist the entire task comparison before spending work on directions.
        # Captured vectors for the smaller direction subset remain on CPU.
        self.write_groups("task", self.task, gather)
        for index, (sample, captured, old_supports) in enumerate(directions):
            for number in self.directions:
                self.measure_direction(model, sample, number, captured, old_supports)
            self.progress("directions", index + 1, len(directions))

    @torch.no_grad()
    def suffix(self, model, sample, number, hidden, condition, repeat):
        started = time.perf_counter()
        previous_mode = model.training
        with self.rng.replay():
            model.eval()
            try:
                result = model.causal_suffix(number, hidden.to(self.device), **batch(sample, self.device))
            finally:
                model.train(previous_mode)
        self.timings.append({"sequence": sample["sequence"], "layer": number, "condition": condition,
                             "repeat": repeat, "seconds": time.perf_counter() - started})
        return result.losses.detach().cpu().double().reshape(-1).numpy()

    @torch.no_grad()
    def measure_direction(self, model, sample, number, captured, old_supports):
        current, layer = captured.layers[number], captured.modules[number]
        fixed, natural = layer.reference_routes(current.hidden.to(self.device),
            old_supports[number].to(self.device), current.support.to(self.device))
        dtype = current.residual.dtype
        base = (current.residual + fixed.detach().cpu().to(current.output.dtype)).to(dtype)
        route = (current.residual + natural.detach().cpu().to(current.output.dtype)).to(dtype)
        observed = (current.residual + current.output).to(dtype)
        valid = sample["valid"].bool()
        route[~valid] = base[~valid]
        delta = route.double() - base.double()
        settings = self.protocol["direction"]
        seeds = [settings["seed"] + sample["sequence"] * 1000003 + number * 10007 + draw
                 for draw in range(settings["gaussian_pairs"])]
        random_vectors = [gaussian_direction(delta, seed) for seed in seeds]
        actual_norms = np.stack([np.stack([
            ((base.double() + sign * vector).to(dtype).double() - base.double()).square().sum(-1).numpy()
            for sign in (1, -1)]) for vector in random_vectors])
        base_losses, route_losses, observed_losses, random_losses = [], [], [], []
        for repeat in range(settings["execution_repeats"]):
            conditions = [("base", None, None), ("route", None, None), ("observed", None, None)]
            conditions += [(f"gaussian_{draw}_{sign:+d}", draw, sign)
                           for draw in range(len(random_vectors)) for sign in (1, -1)]
            # Same permutation on every EP rank; vary order between executions.
            order = np.random.default_rng(settings["seed"] + number * 10007 + repeat).permutation(len(conditions))
            losses = {}
            for index in order:
                name, draw, sign = conditions[index]
                hidden = ({"base": base, "route": route, "observed": observed}[name] if draw is None
                          else (base.double() + sign * random_vectors[draw]).to(dtype))
                losses[name] = self.suffix(model, sample, number, hidden, name, repeat)
            base_losses.append(losses["base"])
            route_losses.append(losses["route"])
            observed_losses.append(losses["observed"])
            random_losses.append(np.stack([np.stack([losses[f"gaussian_{draw}_{sign:+d}"] for sign in (1, -1)])
                                           for draw in range(len(random_vectors))]))
        anchor, routed = np.stack(base_losses), np.stack(route_losses)
        random_loss, observed_loss = np.stack(random_losses), np.stack(observed_losses)
        random_harm = random_loss.mean(2) - anchor[:, None, :]
        route_harm = np.broadcast_to((routed - anchor)[:, None, :], random_harm.shape)
        control = np.broadcast_to((observed_loss - routed)[:, None, :], random_harm.shape)
        energy = delta.square().sum(-1).numpy()
        energy_columns = {"route_energy": np.broadcast_to(energy, random_harm.shape),
            "random_realized_energy": np.broadcast_to(actual_norms.mean(1), random_harm.shape),
            "random_absolute_energy_mismatch": np.broadcast_to(np.abs(actual_norms - energy).mean(1), random_harm.shape)}
        self.directions[number].append(sample, {**energy_columns, "route_harm": route_harm, "random_harm": random_harm,
            "specificity": route_harm - random_harm, "observed_vs_route": control,
            "suffix_vs_full": np.broadcast_to((observed_loss - captured.network.losses.numpy())[:, None, :], random_harm.shape)})
        self.save_tokens(f"layer_{number:02d}", sample, {"base_loss": anchor, "route_loss": routed,
            "random_loss": random_loss, "observed_loss": observed_loss, "gaussian_seeds": np.array(seeds),
            "route_energy": energy, "random_realized_energy": actual_norms,
            "old_support": old_supports[number].numpy(), "new_support": current.support.numpy()})

    def save_tokens(self, name, sample, columns):
        destination = self.output / self.engine / name / "sequences"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / f"sequence_{sample['sequence']:06d}.npz",
            sequence=sample["sequence"], group=sample["group"], valid=sample["valid"].numpy(), **columns)

    def progress(self, stage, completed, total=None):
        if self.announce:
            print(json.dumps({"engine": self.engine, "stage": stage,
                              "local_sequences_completed": completed,
                              "local_sequences_total": len(self.samples) if total is None else total}), flush=True)

    def write_groups(self, name, table, gather):
        parts = gather(table.rows)
        if parts is None:
            return
        groups, counts, columns = DocumentSums.merge(parts)
        destination = self.output / self.engine / name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "groups.npz", group=np.array(groups), N=counts, **columns)
        if name == "task":
            summary = crossed_summary(columns, counts, self.protocol["statistics"], multiplicity=3)
            summary["cancellation"] = cancellation_description(summary)
            (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
            (self.output / "protocol.json").write_text(json.dumps(self.protocol, indent=2))

    def write(self, measurement, gather=lambda value: [value]):
        tables = {f"layer_{n:02d}": table for n, table in self.directions.items()}
        for name, table in tables.items():
            self.write_groups(name, table, gather)
        parts = gather(self.timings)
        if parts is not None:
            (self.output / self.engine / "timing.json").write_text(json.dumps(parts, indent=2))


def write_engineering_report(output, engines, *, protocol):
    """Can also run on Mac using only groups.npz and the copied protocol."""
    output = Path(output)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    results = {}
    lines = ["# 真实一步更新：任务损害与方向特殊性", "",
        "工程尺度采用用户指定的‘抵消一步收益’：U<0、V>0、U+V>0。U 是旧支持集反事实的当次收益，",
        "不等同于另一种训练算法可实现的长期收益。所有区间条件于这一次更新和所选文本分布。",
        "cpu_* 实现用于软件测试，不作为 H20 实验证据。损失单位为 nats/有效 token；范数偏差比例无量纲。", ""]
    for engine in engines:
        results[engine] = {}
        lines += [f"## {engine}", "", "| 比较 | 指标 | 点估计 | 区间 | 正态近似 MDE |",
                  "|---|---|---|---|---|"]
        names = ["task", *[f"layer_{number:02d}" for number in protocol["direction"]["layers"]]]
        for name in names:
            with np.load(output / engine / name / "groups.npz") as data:
                columns = {key: data[key] for key in data.files if key not in ("group", "N")}
                summary = crossed_summary(columns, data["N"], protocol["statistics"],
                    multiplicity=3 if name == "task" else len(protocol["direction"]["layers"]))
            results[engine][name] = summary
            metrics = ("U", "V", "loss_change", "sham_old", "sham_new") if name == "task" else (
                "route_harm", "random_harm", "specificity", "observed_vs_route", "suffix_vs_full")
            for metric in metrics:
                stat = summary["statistics"][metric]
                lo, hi = stat["interval"]
                lines.append(f"| {name} | {metric} | {stat['estimate']:.8g} | [{lo:.8g}, {hi:.8g}] | {stat['normal_approximation_mde']:.5g} |")
            if name == "task":
                decision = cancellation_description(summary)
                summary["cancellation"] = decision
                summary["resolution_projection"] = resolution_projection(summary, decision["pointwise_continuous_gain"], protocol["statistics"])
                for metric in ("U", "V", "loss_change"):
                    stat = summary["statistics"][metric]
                    stat["relative_perplexity_change"] = float(np.expm1(stat["estimate"]))
                    stat["relative_perplexity_change_interval"] = np.expm1(stat["interval"]).tolist()
            else:
                energy = summary["statistics"]["route_energy"]["estimate"]
                mismatch = summary["statistics"]["random_absolute_energy_mismatch"]["estimate"]
                summary["realized_norm_comparison"] = {
                    "absolute_energy_mismatch_over_route_energy": mismatch / energy if energy > 0 else None,
                    "note": "requested per-token norm is equal in FP64; injection casting changes realized norms"}
                lines.append(f"| {name} | 随机注入绝对能量误差 / 路由能量 | {mismatch / energy:.6g} | — | — |"
                             if energy > 0 else f"| {name} | 路由注入能量为零，范数比未定义 | — | — | — |")
        task = results[engine]["task"]
        lines += ["", task["cancellation"]["description"] + "。",
            f"有效位置 {task['valid_positions']}；文档组 {task['document_groups']}；主比较真实执行重复 {task['execution_repeats']}。",
            "空干预的区间、同状态执行重复与方向范数舍入差须一起解释；它们没有被从主效应相减。", ""]
    lines += ["## 解释范围", "",
        "主比较的三个符号使用 Bonferroni 同时区间；方向特殊性在所列层之间校正。其他诊断沿用同样的保守区间，",
        "不声称表中所有指标构成一个统一的同时置信集合。随机正负号先配成一组，再对文档、执行重复和 Gaussian 组交叉重采样。",
        "分轴标准误与样本数投影在 effects.json；MDE 是已观测方差下的正态近似，不是事前功效保证，也不覆盖未知路径偏差。", "",
        "specificity>0 表示从同一旧路由表示出发，正向真实路由差比随机方向平均更有害。单层读数不能相加恢复全网 V。",
        "BF16 注入舍入后的范数保存在逐序列文件；FP32 参考和 BF16 执行分别报告。observed_vs_route 是本层重建差，",
        "suffix_vs_full 是已捕获自然表示接续与完整前向的差，均是实现比较，不是普遍误差上界。", "",
        "本报告只回答第一、二步；Loss Spike 与 μP 跨宽度迁移需要配对训练分支，不能由本报告推断根因。"]
    (output / "effects.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    (output / "report.md").write_text("\n".join(lines) + "\n")
