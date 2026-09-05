"""Paired document-cluster statistics conditional on one training trajectory."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class GroupTable:
    group_ids: np.ndarray
    counts: np.ndarray
    sums: dict[str, np.ndarray]

    @classmethod
    def from_tokens(cls, groups: np.ndarray, valid: np.ndarray, columns: Mapping[str, np.ndarray]):
        ids, membership = np.unique(groups[valid], return_inverse=True)
        counts = np.bincount(membership, minlength=len(ids)).astype(np.float64)
        sums = {
            name: np.bincount(membership, weights=np.asarray(values)[valid], minlength=len(ids))
            for name, values in columns.items()
        }
        return cls(ids, counts, sums)

    def total(self) -> dict[str, float]:
        return {"N": float(self.counts.sum()), **{k: float(v.sum()) for k, v in self.sums.items()}}

    def rows(self) -> list[dict]:
        return [
            {"group": str(group), "N": int(self.counts[i]), **{k: float(v[i]) for k, v in self.sums.items()}}
            for i, group in enumerate(self.group_ids)
        ]


def derived(total: Mapping[str, float]) -> dict[str, float | None]:
    """Recompute ratios from pooled sums, never average document-level ratios."""
    result = {k: v / total["N"] for k, v in total.items() if k != "N"}
    if "J" in total:
        result["conditional_J"] = total["J_switched"] / total["switched"] if total["switched"] else None
        result["r_h"] = math.sqrt(total["J"] / total["H"]) if total["H"] else None
        result["J_over_T"] = total["J"] / total["T"] if total["T"] else None
    if "V" in total:
        result["ppl_ratio"] = math.exp(result["V"])
    if "fp32_error_samples" in total:
        result["fp32_error_rms"] = math.sqrt(total["fp32_error_energy_sum"] / total["fp32_error_samples"]) if total["fp32_error_samples"] else None
        result["execution_repeat_rms"] = math.sqrt(total["execution_repeat_energy"] / total["N"])
    return result


def bootstrap_indices(group_count: int, repeats: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(group_count, size=(repeats, group_count))


def paired_bootstrap(table: GroupTable, indices: np.ndarray, confidence: float = 0.95) -> dict:
    """Reuse these same indices for every paired alpha/precision/endpoint table.

    No-event resamples leave conditional amplitudes undefined. Their number is
    reported, and the conditional interval uses only defined resamples.
    """
    estimates = derived(table.total())
    # Bootstrap multiplicities pool all additive statistics in one matrix product.
    weights = np.zeros((len(indices), len(table.group_ids)), dtype=np.float64)
    np.add.at(weights, (np.arange(len(indices))[:, None], indices), 1)
    names = list(table.sums)
    matrix = np.column_stack([table.counts, *(table.sums[name] for name in names)])
    pooled = weights @ matrix
    counts = pooled[:, 0]
    totals = {name: pooled[:, i + 1] for i, name in enumerate(names)}
    draws = {name: values / counts for name, values in totals.items()}

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)

    if "J" in totals:
        draws["conditional_J"] = ratio(totals["J_switched"], totals["switched"])
        draws["r_h"] = np.sqrt(ratio(totals["J"], totals["H"]))
        draws["J_over_T"] = ratio(totals["J"], totals["T"])
    if "V" in totals:
        draws["ppl_ratio"] = np.exp(draws["V"])
    if "fp32_error_samples" in totals:
        draws["fp32_error_rms"] = np.sqrt(ratio(totals["fp32_error_energy_sum"], totals["fp32_error_samples"]))
        draws["execution_repeat_rms"] = np.sqrt(totals["execution_repeat_energy"] / counts)
    draws = {name: values[~np.isnan(values)] for name, values in draws.items()}
    quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]
    return {
        name: {
            "estimate": estimate,
            "interval": np.quantile(draws[name], quantiles).tolist() if len(draws[name]) else None,
            "defined_resamples": len(draws[name]),
        }
        for name, estimate in estimates.items()
    }


def energy_tail(energy: np.ndarray, switched: np.ndarray, top_counts=(1, 10, 100)) -> dict:
    """Exact empirical tails; retain token scalars separately for final-step/alpha data."""
    events = np.asarray(energy)[np.asarray(switched, dtype=bool)]
    total = float(np.sum(energy, dtype=np.float64))
    largest = np.sort(events)[::-1]
    return {
        "events": int(len(events)),
        "conditional_quantiles": dict(zip(
            ("q50", "q90", "q99", "q999"),
            np.quantile(events, [0.5, 0.9, 0.99, 0.999]).tolist(),
        )) if len(events) else None,
        "largest_event_energy_share": {
            str(n): float(largest[:n].sum()) / total if total else None for n in top_counts
        },
    }


def adjacent_slopes(alphas: list[float], energies: list[float], envelopes: list[tuple]) -> list[dict]:
    rows = []
    for a0, a1, j0, j1, e0, e1 in zip(alphas, alphas[1:], energies, energies[1:], envelopes, envelopes[1:]):
        resolved = a0 > 0 and j0 > 0 and j1 > 0 and (e0[1] < e1[0] or e1[1] < e0[0])
        slope = math.log(j1 / j0) / math.log(a1 / a0) if resolved else None
        rows.append({"alpha0": a0, "alpha1": a1, "energy_slope": slope,
                     "rms_slope": slope / 2 if slope is not None else None})
    return rows
