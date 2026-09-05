"""Capture X[t-1], perform a real update in train.py, then measure X[t]."""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from moe_study.metrics import boundary_terms, loss_terms, same_norm_direction, support_changes, update_terms
from moe_study.reference.network_fp32 import NetworkOutput
from moe_study.state import preserve_rng
from moe_study.statistics import GroupTable, bootstrap_indices, energy_tail, paired_bootstrap


def batch(sample: dict, device) -> dict:
    return {name: sample[name].to(device)[None] for name in ("tokens", "labels", "valid")}


@dataclass
class OldLayer:
    hidden: Tensor
    execution: Tensor
    reference: Tensor
    support: Tensor
    scores: Tensor
    repeat_energy: Tensor | None = None


@dataclass
class OldSequence:
    layers: dict[int, OldLayer]
    losses: Tensor
    repeated_losses: Tensor


@dataclass
class CapturedState:
    sequences: list[OldSequence]
    routers: dict[int, Tensor]

    def network_outputs(self):
        return [NetworkOutput(sequence.losses, sequence.losses.new_zeros(()),
                              {number: layer.support for number, layer in sequence.layers.items()})
                for sequence in self.sequences]


@dataclass
class ScalarTable:
    groups: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    sequences: list[np.ndarray] = field(default_factory=list)
    columns: dict[str, list[np.ndarray]] = field(default_factory=dict)
    incoming_ranks: list[np.ndarray] = field(default_factory=list)

    def append(self, sample: dict, values: dict[str, Tensor]):
        valid = sample["valid"].numpy().reshape(-1)
        self.groups.append(np.repeat(sample["group"], len(valid)))
        self.masks.append(valid)
        self.sequences.append(np.repeat(sample["sequence"], len(valid)))
        for key, value in values.items():
            self.columns.setdefault(key, []).append(value.detach().cpu().numpy().reshape(-1))

    def arrays(self):
        return np.concatenate(self.groups), np.concatenate(self.masks), {
            key: np.concatenate(values) for key, values in self.columns.items()
        }


class ResultWriter:
    """Consume one measurement's scalar tables; never persist high-dimensional activations.

    Distributed callers can pass a gather callable to pool one layer at a time
    on rank 0. It returns None on the other ranks, which continue the next layer.
    """
    def __init__(self, root: Path, measurement: dict, gather=lambda value: [value]):
        self.root, self.measurement, self.gather = root, measurement, gather

    def write(self, tag: str, tables: dict[str, ScalarTable], retain_tokens: bool, sequence_limit=None):
        for name, table in tables.items():
            groups, valid, columns = table.arrays()
            sequences = np.concatenate(table.sequences)
            if sequence_limit is not None:
                valid = valid & (sequences < sequence_limit)
            if table.incoming_ranks:
                ranks = np.concatenate(table.incoming_ranks)[valid]
                rank_histogram = np.bincount(ranks[ranks > 0].astype(int), minlength=129)
            else:
                rank_histogram = None
            parts = self.gather((groups, valid, columns, sequences, rank_histogram))
            if parts is None:
                continue
            groups = np.concatenate([part[0] for part in parts])
            valid = np.concatenate([part[1] for part in parts])
            columns = {key: np.concatenate([part[2][key] for part in parts]) for key in parts[0][2]}
            destination = self.root / tag / name
            destination.mkdir(parents=True, exist_ok=True)
            grouped = GroupTable.from_tokens(groups, valid, columns)
            indices = bootstrap_indices(len(grouped.group_ids), self.measurement["bootstrap_repeats"],
                                        self.measurement["bootstrap_seed"])
            summary = {"valid_positions": int(valid.sum()), "document_groups": len(grouped.group_ids),
                       "statistics": paired_bootstrap(grouped, indices, self.measurement["confidence"]),
                       "interval_scope": "input sampling conditional on this trajectory and text distribution"}
            if "J" in columns:
                switched = columns["switched"][valid].astype(bool)
                summary["tail"] = energy_tail(columns["J"][valid], switched)
                summary["event_document_groups"] = len(np.unique(groups[valid][switched]))
                summary["replacements_histogram"] = np.bincount(columns["replacements"][valid].astype(int)).tolist()
                summary["incoming_old_rank_histogram"] = np.sum([part[4] for part in parts], axis=0).tolist()
            (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
            with (destination / "groups.jsonl").open("w") as output:
                for row in grouped.rows():
                    output.write(json.dumps(row, allow_nan=False) + "\n")
            if retain_tokens and "J" in columns:
                # 10 bytes/position/layer before ZIP: exact J plus count/rank scalars.
                # Full S/J/C/T and boundary moments live in FP64 group records.
                np.savez_compressed(destination / "tokens.npz", J=columns["J"].astype(np.float64),
                                    replacements=columns["replacements"].astype(np.uint8),
                                    incoming_rank_max=columns["incoming_rank_max"].astype(np.uint8))
            if name == "network":
                np.savez_compressed(destination / "tokens.npz", valid=valid, groups=groups,
                                    sequence=np.concatenate([p[3] for p in parts]), **columns)


class PairedMeasurement:
    def __init__(self, model, samples: list[dict], config: dict, device="cpu"):
        self.model, self.samples, self.config, self.device = model, samples, config, device

    @torch.no_grad()
    def capture(self, repeat=True) -> CapturedState:
        sequences, routers = [], {}
        with preserve_rng():
            self.model.eval()
            for sample in self.samples:
                layers = {}

                def observe(number, layer, hidden, scores, support, output):
                    reference = layer.reference_routes(hidden, support, support)[0]
                    layers[number] = OldLayer(hidden.detach().cpu(), output.detach().cpu(),
                                              reference.cpu(), support.cpu().to(torch.int32), scores.cpu())
                    if number not in routers:
                        routers[number] = layer.router_weight.detach().cpu().clone()

                actual = self.model(**batch(sample, self.device), observer=observe)

                def observe_repeat(number, layer, hidden, scores, support, output):
                    layers[number].repeat_energy = (output.cpu().double() - layers[number].execution.double()).square().mean(-1)

                if repeat:
                    repeated = self.model(**batch(sample, self.device), observer=observe_repeat)
                else:
                    repeated = actual
                    for layer in layers.values():
                        layer.repeat_energy = torch.zeros(layer.hidden.shape[0], dtype=torch.float64)
                sequences.append(OldSequence(layers, actual.losses.cpu().reshape(-1), repeated.losses.cpu().reshape(-1)))
        return CapturedState(sequences, routers)

    def baseline_tables(self, captured: CapturedState) -> dict[str, ScalarTable]:
        """α=0 reuses the captured FP32 forward: all three states are the same object."""
        tables = {"network": ScalarTable()}
        for sample, sequence in zip(self.samples, captured.sequences):
            tables["network"].append(sample, loss_terms(sequence.losses, sequence.losses, sequence.losses))
            for number, record in sequence.layers.items():
                values = update_terms(record.reference, record.reference, record.reference, record.hidden)
                zero = torch.zeros_like(values["J"])
                router = captured.routers[number]
                values.update(boundary_terms(record.hidden, record.hidden, router, router, record.scores, record.support.shape[-1]))
                values.update({name: zero for name in ("J_squared", "J_switched", "J_unchanged", "switched", "replacements", "multi_exchange",
                    "incoming_rank_max", "execution_repeat_energy", "fp32_error_energy_sum", "fp32_error_samples")})
                table = tables.setdefault(f"fp32_reference/layer_{number:02d}", ScalarTable())
                table.append(sample, values)
                table.incoming_ranks.append(np.zeros(record.support.shape, dtype=np.int32))
        return tables

    @torch.no_grad()
    def finish(self, captured: CapturedState, release_old=True) -> dict[str, ScalarTable]:
        tables = {"network": ScalarTable()}
        self.latest_outputs = []
        with preserve_rng():
            self.model.eval()
            for sample, old in zip(self.samples, captured.sequences):
                old_supports = {number: record.support for number, record in old.layers.items()}

                def observe(number, layer, hidden, scores, support, output):
                    record = old.layers[number]
                    old_support = record.support.to(hidden.device)
                    old_hidden = record.hidden.to(hidden.device)
                    old_router = captured.routers[number].to(hidden.device)
                    old_scores = record.scores.to(hidden.device)
                    changes = support_changes(old_support, support, old_scores)
                    boundary = boundary_terms(old_hidden, hidden, old_router, layer.router_weight, old_scores, support.shape[-1])
                    middle_exec = layer.execution_route(hidden, old_support)
                    middle_ref, new_ref = layer.reference_routes(hidden, old_support, support)
                    sample_count = self.config["fp64_sample_tokens_per_layer"]
                    rows = torch.arange(min(sample_count, hidden.shape[0]), device=hidden.device)
                    mid64, new64 = layer.reference_routes(hidden[rows], old_support[rows], support[rows], torch.float64)
                    error = ((new_ref[rows].double() - middle_ref[rows].double()) - (new64 - mid64)).square().mean(-1)
                    # Store numerical samples with their own count, separate from full-population J.
                    error_sum = torch.zeros(hidden.shape[0], device=hidden.device, dtype=torch.float64)
                    error_count = torch.zeros_like(error_sum)
                    sample_valid = sample["valid"][rows.cpu()].to(hidden.device)
                    error_sum[rows], error_count[rows] = error * sample_valid, sample_valid.double()
                    shared = {"switched": changes["switched"], "replacements": changes["replacements"],
                              "multi_exchange": changes["replacements"] > 1,
                              "incoming_rank_max": changes["incoming_rank"].max(-1).values,
                              "execution_repeat_energy": record.repeat_energy.to(hidden.device),
                              "fp32_error_energy_sum": error_sum, "fp32_error_samples": error_count, **boundary}
                    for precision, y0, y10, y1 in (
                        ("execution", record.execution, middle_exec, output),
                        ("fp32_reference", record.reference, middle_ref, new_ref),
                    ):
                        values = update_terms(y0.to(hidden.device), y10, y1, hidden)
                        values.update(shared)
                        values["J_squared"] = values["J"].square()
                        values["J_switched"] = values["J"] * changes["switched"]
                        values["J_unchanged"] = values["J"] * ~changes["switched"]
                        table = tables.setdefault(f"{precision}/layer_{number:02d}", ScalarTable())
                        table.append(sample, values)
                        table.incoming_ranks.append(changes["incoming_rank"].cpu().numpy())
                    # Only scalar arrays outlive this callback. Old vectors are released now.
                    if release_old:
                        del old.layers[number]

                actual = self.model(**batch(sample, self.device), observer=observe)
                self.latest_outputs.append(NetworkOutput(actual.losses.cpu(), actual.balance_loss.cpu(), actual.supports))
                fixed = self.model(**batch(sample, self.device), supports=old_supports)
                values = loss_terms(old.losses, fixed.losses.cpu().reshape(-1), actual.losses.cpu().reshape(-1))
                values["repeat_loss_difference"] = old.repeated_losses.double() - old.losses.double()
                tables["network"].append(sample, values)
        return tables


@torch.no_grad()
def interventions(model, samples, old_supports, config: dict, seed: int, device="cpu", baseline_outputs=None) -> dict[str, ScalarTable]:
    tables = {}
    choices = [(layer, None) for layer in config["single_layer_interventions"]]
    choices += [(config["random_direction_layer"], repeat) for repeat in range(config["random_direction_repeats"])]
    with preserve_rng():
        model.eval()
        baseline_outputs = baseline_outputs if baseline_outputs is not None else [model(**batch(sample, device)) for sample in samples]
        for number, repeat in choices:
            name = f"layer_{number:02d}/" + ("old_support" if repeat is None else f"random_{repeat}")
            table = ScalarTable()
            for sample, support, actual in zip(samples, old_supports, baseline_outputs):
                # Seeding by sequence makes the directions independent of rank placement.
                generator = torch.Generator(device=device).manual_seed(seed + 1000003 * (repeat or 0) + sample["sequence"])

                def replace(layer_number, layer, hidden, selected, output):
                    if layer_number != number:
                        return output
                    old_output = layer.execution_route(hidden, support[number].to(device))
                    replacement = old_output.float() - output.float()
                    return old_output if repeat is None else (output.float() + same_norm_direction(replacement, generator)).to(output.dtype)

                replaced = model(**batch(sample, device), intervention=replace)
                table.append(sample, {"actual_loss": actual.losses.reshape(-1), "replaced_loss": replaced.losses.reshape(-1),
                                      "replacement_loss_difference": replaced.losses.cpu().reshape(-1).double() - actual.losses.cpu().reshape(-1).double()})
            tables[name] = table
    return tables


def main():
    parser = argparse.ArgumentParser(description="Show the experiment's paired-update work list (no training).")
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.experiment.read_text())
    measurement = config["measurement"]
    rows = []
    for step in range(1, config["experiment"]["total_steps"] + 1):
        large = step in measurement["large_steps"]
        window = any(a <= step <= b for a, b in measurement["consecutive_windows"])
        if large or window:
            rows.append({"old_step": step - 1, "new_step": step,
                         "sequences": measurement["large_sequences"] if large else measurement["window_sequences"],
                         "window": window})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
