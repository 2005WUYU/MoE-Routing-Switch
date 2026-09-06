"""Crossed document/execution/direction uncertainty and effect-size resolution.

Statistical descriptions do not control execution. Antithetic signs are averaged
inside each Gaussian draw, and are never counted as independent directions.
"""

from statistics import NormalDist
import math

import numpy as np


def bootstrap_weights(size, repeats, generator, normalized):
    indices = generator.integers(size, size=(repeats, size))
    weights = np.zeros((repeats, size), dtype=np.float64)
    np.add.at(weights, (np.arange(repeats)[:, None], indices), 1)
    return weights / size if normalized else weights


def crossed_summary(columns, counts, options, multiplicity=1):
    """Each column has shape [execution, Gaussian-pair, document] and stores sums.

    All metrics share the same draws. Pool token sums/counts within resampled
    documents; average executions and Gaussian pairs without multiplying N.
    """
    shape = next(iter(columns.values())).shape
    executions, directions, documents = shape
    draws_count = options["bootstrap_repeats"]
    generator = np.random.default_rng(options["seed"])
    e = bootstrap_weights(executions, draws_count, generator, True)
    d = bootstrap_weights(directions, draws_count, generator, True)
    g = bootstrap_weights(documents, draws_count, generator, False)
    denominator = g @ counts
    alpha = (1 - options["confidence"]) / multiplicity
    q = [alpha / 2, 1 - alpha / 2]
    normal_factor = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(options["design_power"])
    result = {}
    for name, value in columns.items():
        combined = np.einsum("br,rdg,bd,bg->b", e, value, d, g, optimize=True) / denominator
        by_document = (g @ value.mean((0, 1))) / denominator
        by_execution = e @ (value.mean(1).sum(-1) / counts.sum())
        by_direction = d @ (value.mean(0).sum(-1) / counts.sum())
        standard_errors = {axis: float(draws.std(ddof=1)) for axis, draws in (
            ("combined", combined), ("documents_only", by_document),
            ("executions_only", by_execution), ("directions_only", by_direction))}
        result[name] = {"estimate": float(value.mean((0, 1)).sum() / counts.sum()),
            "interval": np.quantile(combined, q).tolist(), "standard_errors": standard_errors,
            "normal_approximation_mde": normal_factor * standard_errors["combined"]}
    return {"valid_positions": int(counts.sum()), "document_groups": documents,
            "execution_repeats": executions, "gaussian_pairs": directions,
            "confidence": options["confidence"], "multiplicity": multiplicity,
            "statistics": result,
            "scope": "crossed resampling conditional on these parameters, documents and observed execution paths; no training-run uncertainty"}


def cancellation_description(summary):
    stats = summary["statistics"]
    conditions = {"continuous_gain_resolved": stats["U"]["interval"][1] < 0,
                  "routing_harm_resolved": stats["V"]["interval"][0] > 0,
                  "net_gain_cancelled_resolved": stats["loss_change"]["interval"][0] > 0}
    gain = -stats["U"]["estimate"]
    control_extent = max(abs(endpoint) for name in ("sham_old", "sham_new") for endpoint in stats[name]["interval"])
    if all(conditions.values()):
        description = "抵消一步收益的三个符号得到区间支持"
    elif stats["U"]["interval"][0] >= 0:
        description = "连续项没有提供正收益，这一步不具备所选工程尺度的收益基准"
    elif conditions["continuous_gain_resolved"] and stats["loss_change"]["interval"][1] < 0:
        description = "本次区间排除了路由把这一步连续收益全部抵消；不等于排除部分增损或罕见损害"
    else:
        description = "尚未同时分辨抵消一步收益所需的三个符号"
    return {"criterion": "cancel_continuous_gain", "conditions": conditions,
            "pointwise_continuous_gain": gain, "observed_sham_interval_extent": control_extent,
            "V_interval_above_observed_sham_extent": stats["V"]["interval"][0] > control_extent,
            "description": description,
            "note": "空干预只度量已观测实现差，不是所有路由干预路径偏差的统一上界；不作噪声相减"}


def resolution_projection(summary, target, options, metric="V"):
    """Approximate one-axis sample projections; other axes and bias stay present."""
    alpha = (1 - options["confidence"]) / summary["multiplicity"]
    factor = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(options["design_power"])
    errors = summary["statistics"][metric]["standard_errors"]
    sizes = {"documents_only": summary["document_groups"], "executions_only": summary["execution_repeats"],
             "directions_only": summary["gaussian_pairs"]}
    return {"target_nats_per_token": target,
            "approximate_axis_counts": {name: math.ceil(size * (factor * errors[name] / target) ** 2) if errors[name] > 0 else None
                                        for name, size in sizes.items()} if target > 0 else None,
            "assumptions": "normal approximation; independent new units of the same distribution; only the named variance component shrinks; path bias is not included"}
