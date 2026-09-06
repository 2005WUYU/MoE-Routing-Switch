"""One-update loss comparison with six real forwards per numerical implementation.

Only tensors are copied during a forward. Expert reference work and comparisons
run after it returns. Full logits live on CPU for one sequence and are discarded
after blockwise comparisons; files contain losses and scalar diagnostics only.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from moe_study.measure import ResultWriter, ScalarTable, batch


@dataclass
class RNGSnapshot:
    python: object
    numpy: tuple
    cpu: torch.Tensor
    cuda: list | None
    tracker: object
    tracker_states: dict | None

    @classmethod
    def capture(cls, tracker=None):
        return cls(random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                   tracker, deepcopy(tracker.get_states()) if tracker is not None else None)

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        if self.tracker is not None:
            self.tracker.set_states(deepcopy(self.tracker_states))

    @contextmanager
    def replay(self):
        outside = self.capture(self.tracker)
        self.restore()
        try:
            yield
        finally:
            outside.restore()


@torch.no_grad()
def logit_difference(first, second, device="cpu", block_size=32):
    """KL(softmax(first) || softmax(second)) and centered RMS of rounded logits.

    Float64 reduction avoids losing small differences between log probabilities.
    Values are not clamped, and no numerical floor is subtracted.
    """
    first, second = first.reshape(-1, first.shape[-1]), second.reshape(-1, second.shape[-1])
    kl, rms = [], []
    for start in range(0, len(first), block_size):
        a = first[start:start + block_size].to(device=device, dtype=torch.float64)
        b = second[start:start + block_size].to(device=device, dtype=torch.float64)
        log_p, log_q = a.log_softmax(-1), b.log_softmax(-1)
        kl.append((log_p.exp() * (log_p - log_q)).sum(-1).cpu())
        difference = a - b
        difference -= difference.mean(-1, keepdim=True)
        rms.append(difference.square().mean(-1).sqrt().cpu())
    return {"KL": torch.cat(kl), "centered_logits_RMS": torch.cat(rms)}


def support_difference(first, second):
    return torch.stack([(first[n].sort(-1).values != second[n].sort(-1).values).any(-1)
                        for n in first]).double().mean(0)


@dataclass
class LayerCapture:
    residual: torch.Tensor
    hidden: torch.Tensor
    scores: torch.Tensor
    support: torch.Tensor
    output: torch.Tensor


@dataclass
class ForwardCapture:
    network: object
    layers: dict
    modules: dict


def squared_difference(first, second):
    return (second.double() - first.double()).square().mean(-1)


def layer_comparison(first, second, prefix):
    """Return joint per-position data, without attributing repeat energy to J."""
    fields = {key: [] for key in ("residual_difference_energy", "router_input_difference_energy",
                                  "score_max_difference", "output_difference_energy", "support_changed")}
    for number, a in first.items():
        b = second[number]
        fields["residual_difference_energy"].append(squared_difference(a.residual, b.residual))
        fields["router_input_difference_energy"].append(squared_difference(a.hidden, b.hidden))
        fields["score_max_difference"].append((b.scores.double() - a.scores.double()).abs().max(-1).values)
        fields["output_difference_energy"].append(squared_difference(a.output, b.output))
        fields["support_changed"].append((a.support.sort(-1).values != b.support.sort(-1).values).any(-1))
    return {f"{prefix}_{key}": torch.stack(values).numpy() for key, values in fields.items()}


def layer_values(layers, prefix):
    fields = {key: [] for key in ("support", "margin", "router_input_energy", "residual_energy")}
    for layer in layers.values():
        k = layer.support.shape[-1]
        top = layer.scores.topk(k + 1, dim=-1).values.double()
        fields["support"].append(layer.support)
        fields["margin"].append(top[:, k - 1] - top[:, k])
        fields["router_input_energy"].append(layer.hidden.double().square().mean(-1))
        fields["residual_energy"].append(layer.residual.double().square().mean(-1))
    return {f"{prefix}_{key}": torch.stack(values).numpy() for key, values in fields.items()}


@dataclass
class Baseline:
    first_loss: torch.Tensor
    second_loss: torch.Tensor
    supports: dict
    layers: dict
    columns: dict
    diagnostics: dict


class CausalMeasurement:
    def __init__(self, samples, diagnostic_ids, output, engine, rng, device="cpu", logit_block_size=32, announce=True):
        self.samples, self.diagnostic_ids = samples, set(diagnostic_ids)
        self.output, self.engine, self.rng = Path(output), engine, rng
        self.device, self.logit_block_size = device, logit_block_size
        self.announce = announce
        self.old = {}
        self.network = ScalarTable()
        self.timings = []

    @torch.no_grad()
    def forward(self, model, sample, condition, repeat, supports=None):
        layers, modules = {}, {}

        def observe(number, layer, residual, hidden, scores, chosen, output):
            layers[number] = LayerCapture(*[value.detach().cpu().clone() for value in
                                           (residual, hidden, scores, chosen.to(torch.int32), output)])
            modules[number] = layer

        previous_mode = model.training
        started = time.perf_counter()
        with self.rng.replay():
            model.eval()
            try:
                result = model.causal_forward(**batch(sample, self.device), supports=supports,
                    diagnostic=observe if sample["sequence"] in self.diagnostic_ids else None)
            finally:
                model.train(previous_mode)
        result.losses = result.losses.detach().cpu().reshape(-1).double()
        self.timings.append({"sequence": sample["sequence"], "condition": condition, "repeat": repeat,
                             "seconds": time.perf_counter() - started})
        return ForwardCapture(result, layers, modules)

    def logits(self, first, second, prefix):
        return {f"{prefix}_{name}": value for name, value in logit_difference(
            first.network.logits, second.network.logits, self.device, self.logit_block_size).items()}

    def capture_old(self, model):
        for index, sample in enumerate(self.samples):
            first = self.forward(model, sample, "N0", 1)
            second = self.forward(model, sample, "N0", 2)
            columns = self.logits(first, second, "repeat_N0")
            columns["repeat_N0_route_disagreement"] = support_difference(first.network.supports, second.network.supports)
            diagnostics = {}
            if first.layers:
                diagnostics.update(layer_values(first.layers, "N0"))
                diagnostics.update(layer_values(second.layers, "N0_repeat"))
                diagnostics.update(layer_comparison(first.layers, second.layers, "repeat_N0"))
            self.old[sample["sequence"]] = Baseline(first.network.losses, second.network.losses,
                first.network.supports, first.layers, columns, diagnostics)
            # Baselines own no GPU modules and no vocabulary-sized logits.
            del first, second
            self.progress("old", index + 1)

    @torch.no_grad()
    def reference_jump(self, captured, old):
        values, errors = [], []
        for number, current in captured.layers.items():
            layer = captured.modules[number]
            hidden = current.hidden.to(self.device)
            previous_support = old.supports[number].to(self.device)
            support = current.support.to(self.device)
            fixed, natural = layer.reference_routes(hidden, previous_support, support)
            values.append((natural.double() - fixed.double()).square().mean(-1).cpu())
            errors.append((natural.double() - current.output.to(self.device).double()).square().mean(-1).cpu())
        return {"J_local_reference": torch.stack(values).numpy(),
                "new_reference_vs_execution_energy": torch.stack(errors).numpy()}

    def measure_new(self, model):
        for index, sample in enumerate(self.samples):
            old = self.old.pop(sample["sequence"])
            # The same A0 from the first N0 is used in both counterfactual repeats.
            n1 = self.forward(model, sample, "N1", 1)
            f1 = self.forward(model, sample, "F1", 1, old.supports)
            f2 = self.forward(model, sample, "F1", 2, old.supports)
            n2 = self.forward(model, sample, "N1", 2)
            columns = dict(old.columns)
            columns.update(L0_1=old.first_loss, L0_2=old.second_loss,
                           L1_1=n1.network.losses, L1_2=n2.network.losses,
                           L10_1=f1.network.losses, L10_2=f2.network.losses)
            for repeat in (1, 2):
                columns[f"U_{repeat}"] = columns[f"L10_{repeat}"] - columns[f"L0_{repeat}"]
                columns[f"V_{repeat}"] = columns[f"L1_{repeat}"] - columns[f"L10_{repeat}"]
                columns[f"loss_change_{repeat}"] = columns[f"L1_{repeat}"] - columns[f"L0_{repeat}"]
                for name in ("U", "V", "loss_change"):
                    value = columns[f"{name}_{repeat}"]
                    columns[f"{name}_{repeat}_positive"] = value.clamp_min(0)
                    columns[f"{name}_{repeat}_negative"] = (-value).clamp_min(0)
            for name in ("U", "V", "loss_change"):
                columns[name] = (columns[f"{name}_1"] + columns[f"{name}_2"]) / 2
            for condition, name in (("N0", "L0"), ("N1", "L1"), ("F1", "L10")):
                difference = columns[f"{name}_2"] - columns[f"{name}_1"]
                columns[f"repeat_{condition}_loss"] = difference
                columns[f"repeat_{condition}_positive"] = difference.clamp_min(0)
                columns[f"repeat_{condition}_negative"] = (-difference).clamp_min(0)
            columns.update(self.logits(n1, f1, "route_1"))
            columns.update(self.logits(n2, f2, "route_2"))
            columns.update(self.logits(n1, n2, "repeat_N1"))
            columns.update(self.logits(f1, f2, "repeat_F1"))
            columns["repeat_N1_route_disagreement"] = support_difference(n1.network.supports, n2.network.supports)
            columns["repeat_F1_route_disagreement"] = support_difference(f1.network.supports, f2.network.supports)
            columns["updated_route_disagreement"] = support_difference(old.supports, n1.network.supports)
            columns["position"] = torch.arange(len(sample["valid"]))
            self.network.append(sample, columns)
            if old.layers:
                diagnostics = old.diagnostics
                diagnostics.update(layer_values(n1.layers, "N1"))
                diagnostics.update(layer_values(n2.layers, "N1_repeat"))
                diagnostics.update(layer_values(f1.layers, "F1"))
                diagnostics.update(layer_comparison(old.layers, n1.layers, "update"))
                diagnostics.update(layer_comparison(n1.layers, n2.layers, "repeat_N1"))
                diagnostics.update(layer_comparison(f1.layers, f2.layers, "repeat_F1"))
                # Every rank replays its aligned diagnostic sequence after all four forwards.
                diagnostics.update(self.reference_jump(n1, old))
                destination = self.output / self.engine / "diagnostics"
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination / f"sequence_{sample['sequence']:06d}.npz",
                    layer=np.array(list(old.layers)), sequence=sample["sequence"], group=sample["group"],
                    position=np.arange(len(sample["valid"])), valid=sample["valid"].numpy(), **diagnostics)
            del n1, n2, f1, f2, old
            self.progress("new", index + 1)

    def progress(self, stage, completed):
        if self.announce:
            print(json.dumps({"engine": self.engine, "stage": stage,
                              "local_sequences_completed": completed, "local_sequences_total": len(self.samples)}), flush=True)

    def write(self, measurement, gather=lambda value: [value]):
        ResultWriter(self.output, measurement, gather).write(self.engine, {"network": self.network}, True)
        parts = gather(self.timings)
        if parts is not None:
            path = self.output / self.engine / "forward_timing.json"
            path.write_text(json.dumps({"rank_timings": parts}, indent=2))


def write_causal_report(output, engines):
    """Report signed effects and repetition separately; never label a run passed."""
    output = Path(output)
    effects = {}
    lines = ["# 一次真实更新的路由反事实比较", "",
             "V = L1 − L10；正值表示自然新路由比旧支持集反事实的损失高。两个条件使用同一个更新后参数端点。",
             "区间仅表示文档组输入抽样不确定性；两个执行重复不能估计完整的执行方差或训练间方差。", "",
             "| 实现 | 指标 | nats/有效 token | 输入抽样区间 |", "|---|---|---|---|"]
    for engine in engines:
        root = output / engine / "network"
        summary = json.loads((root / "summary.json").read_text())
        rows = [json.loads(line) for line in (root / "groups.jsonl").read_text().splitlines()]
        quantiles = {metric: np.quantile([row[metric] / row["N"] for row in rows], [.01, .1, .5, .9, .99]).tolist()
                     for metric in ("V_1", "V_2", "repeat_N0_loss", "repeat_N1_loss", "repeat_F1_loss")}
        effects[engine] = {**summary, "document_mean_quantiles": quantiles,
                           "quantile_probabilities": [.01, .1, .5, .9, .99], "execution_repeats": 2}
        for metric in ("V_1", "V_2", "U", "loss_change", "repeat_N0_loss", "repeat_N1_loss", "repeat_F1_loss"):
            value = summary["statistics"][metric]
            lo, hi = value["interval"]
            lines.append(f"| {engine} | {metric} | {value['estimate']:.8g} | [{lo:.8g}, {hi:.8g}] |")
    lines += ["", "每个实现的 network/tokens.npz 保留逐位置损失、正负贡献、KL 和中心化 logits RMS；",
              "diagnostics/ 保留完整长度子集的逐层联合标量。旧支持集取各实现首次 N0，重复中不替换。",
              "", "区间跨零表示尚未分辨，不表示效应为零；FP32 的效应不能直接替代原生 BF16 的训练结论。",
              "本次只测一个真实更新附近的即时效果，不据此判断长期稳定性、学习率敏感性或 μP 迁移。"]
    (output / "effects.json").write_text(json.dumps(effects, ensure_ascii=False, indent=2, allow_nan=False))
    (output / "report.md").write_text("\n".join(lines) + "\n")
