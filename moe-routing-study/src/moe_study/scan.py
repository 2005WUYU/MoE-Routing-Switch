"""Independent FP32 network interpolation of a single real master-parameter update."""

from pathlib import Path
import json

import torch

from moe_study.measure import PairedMeasurement, ResultWriter, ScalarTable, batch
from moe_study.reference.expert_fp32 import ieee_fp32
from moe_study.state import preserve_rng


@torch.no_grad()
def run_scan(model, load_alpha, samples, measurement: dict, writer: ResultWriter, device="cpu",
             production_endpoints=None, can_start=lambda: True):
    """load_alpha always constructs θ0 + α Δθ in this independent model.

    Production AdamW and parameters are owned by the caller and remain separate.
    A scheduling callback only ends work between complete alpha blocks.
    """
    completed = []
    with preserve_rng(), ieee_fp32(torch.device(device).type):
        load_alpha(0.0)
        pair = PairedMeasurement(model, samples, measurement, device)
        baseline = pair.capture(repeat=False)
        for alpha in measurement["alphas"]:
            if not can_start():
                break
            if alpha == 0:
                tables = pair.baseline_tables(baseline)
            else:
                load_alpha(alpha)
                tables = pair.finish(baseline, release_old=False)
            # The entire network is FP32; one local-reference table suffices here.
            tables = {name: table for name, table in tables.items() if not name.startswith("execution/")}
            writer.write(f"alpha_{alpha:g}", tables, retain_tokens=True)
            if production_endpoints is not None and alpha in (0.0, 1.0):
                endpoint = ScalarTable()
                production = production_endpoints[int(alpha)]
                for index, (sample, actual_production) in enumerate(zip(samples, production)):
                    if alpha == 0:
                        old = baseline.sequences[index]
                        from moe_study.reference.network_fp32 import NetworkOutput
                        actual_fp32 = NetworkOutput(old.losses, old.losses.new_zeros(()), {n: layer.support for n, layer in old.layers.items()})
                    else:
                        actual_fp32 = pair.latest_outputs[index]
                    route_disagreement = torch.stack([
                        (actual_fp32.supports[layer].sort(-1).values != actual_production.supports[layer].sort(-1).values).any(-1)
                        for layer in actual_fp32.supports
                    ]).float().mean(0)
                    endpoint.append(sample, {
                        "fp32_minus_production_loss": actual_fp32.losses.cpu().reshape(-1).double() - actual_production.losses.cpu().reshape(-1).double(),
                        "route_disagreement_layer_mean": route_disagreement,
                    })
                writer.write(f"endpoint_{int(alpha)}", {"network": endpoint}, retain_tokens=False)
            completed.append(alpha)
    return completed
