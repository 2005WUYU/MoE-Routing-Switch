import json
from pathlib import Path

import numpy as np
import torch

from moe_study.adapters.distributed_reference import split_qkv
from moe_study.config import RunConfig
from moe_study.measure import PairedMeasurement, ResultWriter, ScalarTable, interventions
from moe_study.reference.network_fp32 import QwenConfig, QwenReference
from moe_study.scan import run_scan
from moe_study.state import interpolate_parameters, snapshot_parameters
from moe_study.train import development_samples

ROOT = Path(__file__).resolve().parents[1]


def test_qkv_conversion_preserves_grouped_query_head_order():
    config = QwenConfig(hidden_size=6, num_hidden_layers=1, num_attention_heads=8,
                        num_key_value_heads=2, head_dim=4, num_experts=4,
                        num_experts_per_tok=2, moe_intermediate_size=4, vocab_size=8)
    query = torch.arange(32 * 6).reshape(8, 4, 6)
    key = torch.arange(8 * 6).reshape(2, 4, 6) + 1000
    value = torch.arange(8 * 6).reshape(2, 4, 6) + 2000
    packed = torch.cat([torch.cat([query[g * 4:g * 4 + 4], key[g:g + 1], value[g:g + 1]]) for g in range(2)]).reshape(-1, 6)
    actual = split_qkv(packed, config)
    for result, expected in zip(actual, (query, key, value)):
        assert torch.equal(result, expected.reshape(-1, 6))


def test_window_uses_preselected_sequences_from_large_measurement(tmp_path):
    table = ScalarTable()
    for index, energy in enumerate([1., 2., 100., 200.]):
        sample = {"sequence": index, "group": f"doc-{index}", "valid": torch.ones(2, dtype=torch.bool)}
        table.append(sample, {"V": torch.full((2,), energy)})
    measurement = {"bootstrap_repeats": 10, "bootstrap_seed": 0, "confidence": .95}
    writer = ResultWriter(tmp_path, measurement)
    writer.write("window", {"network": table}, False, sequence_limit=2)
    summary = json.loads((tmp_path / "window/network/summary.json").read_text())
    assert summary["valid_positions"] == 4
    assert summary["statistics"]["V"]["estimate"] == 1.5


def test_fp32_scan_uses_15_full_forwards_and_its_own_zero_support(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(77)
    config = RunConfig.read(ROOT / "examples/cpu_experiment.yaml", ROOT / "examples/cpu_machine.yaml", ROOT / "examples/cpu_schedule.yaml")
    measurement = config.experiment["measurement"]
    model = QwenReference(QwenConfig(**config.experiment["development_model"]))
    samples = development_samples(1, 8, 32, 22, "scan")
    old = snapshot_parameters(model)
    new = {name: value + .01 for name, value in old.items()}
    calls = []
    handle = model.register_forward_hook(lambda module, inputs, result: calls.append(result.supports))
    writer = ResultWriter(tmp_path, measurement)
    run_scan(model, lambda a: model.load_state_dict(interpolate_parameters(old, new, a)), samples, measurement, writer)
    handle.remove()
    assert len(calls) == 15
    for path in (tmp_path / "alpha_0/fp32_reference").glob("*/tokens.npz"):
        with np.load(path) as data:
            assert (data["J"] == 0).all()
            assert (data["replacements"] == 0).all()


def test_changed_execution_scores_use_captured_old_expert_ranks():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    config = QwenConfig(4, 1, 2, 1, 4, 4, 2, 4, 8)
    model = QwenReference(config)
    samples = development_samples(1, 4, 8, 9, "rank")
    options = {"fp64_sample_tokens_per_layer": 2}
    pair = PairedMeasurement(model, samples, options)
    captured = pair.capture()
    old_scores = captured.sequences[0].layers[1].scores.clone()
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.mul_(-1)
    tables = pair.finish(captured)
    changes = tables["fp32_reference/layer_01"].arrays()[2]
    assert (changes["replacements"] == 2).all()
    assert (changes["incoming_rank_max"] == 4).all()


def test_interventions_reuse_baseline_and_need_seven_full_forwards():
    torch.set_num_threads(1)
    config = RunConfig.read(ROOT / "examples/cpu_experiment.yaml", ROOT / "examples/cpu_machine.yaml", ROOT / "examples/cpu_schedule.yaml")
    model = QwenReference(QwenConfig(**config.experiment["development_model"]))
    samples = development_samples(1, 8, 32, 22, "intervention")
    pair = PairedMeasurement(model, samples, config.experiment["measurement"])
    captured = pair.capture()
    baseline = captured.network_outputs()
    supports = [baseline[0].supports]
    calls = []
    handle = model.register_forward_hook(lambda module, inputs, result: calls.append(1))
    interventions(model, samples, supports, config.experiment["measurement"], 0, baseline_outputs=baseline)
    handle.remove()
    assert len(calls) == 7
