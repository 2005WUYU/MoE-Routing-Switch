import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from moe_study.causal_measure import RNGSnapshot
from moe_study.engineering import run_engineering, workload
from moe_study.engineering_measure import DocumentSums, EngineeringMeasurement, gaussian_direction, write_engineering_report
from moe_study.engineering_statistics import cancellation_description, crossed_summary
from moe_study.measure import batch
from moe_study.state import CPUCheckpoint, Progress, snapshot_parameters
from moe_study.train import cpu_update, development_samples, make_optimizer
from test_causal_measurement import small_setup
from test_causal_adapter import NativeModel
from moe_study.adapters.megatron_qwen import MegatronEvaluation


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return yaml.safe_load((ROOT / "examples/cpu_engineering_chain.yaml").read_text())


def test_suffix_from_actual_residual_matches_full_network_including_last_layer(monkeypatch):
    _, reference, samples = small_setup()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    for model in (reference, MegatronEvaluation(NativeModel(), None)):
        sample = samples[0] if model is reference else development_samples(1, 8, 16, 7, "native")[0]
        model.eval()
        captured = {}

        def record(number, layer, residual, hidden, scores, chosen, output):
            captured[number] = (residual + output).clone()

        with torch.no_grad():
            full = model.causal_forward(**batch(sample, "cpu"), diagnostic=record)
            for number, hidden in captured.items():
                suffix = model.causal_suffix(number, hidden, **batch(sample, "cpu"))
                torch.testing.assert_close(suffix.losses.reshape(-1), full.losses.reshape(-1), atol=0, rtol=0)
            replay = model.causal_forward(**batch(sample, "cpu"))
        assert len(replay.supports) == len(captured)
        assert torch.equal(replay.losses, full.losses)


def test_zero_update_executes_real_shams_and_directions_without_changing_parameters(tmp_path):
    _, model, samples = small_setup()
    samples[1]["valid"][3:] = False
    before = snapshot_parameters(model)
    settings = protocol()
    pair = EngineeringMeasurement(samples, [], tmp_path, "test", RNGSnapshot.capture(), protocol=settings, announce=False)
    pair.capture_old(model)
    measure_direction = pair.measure_direction

    def task_is_persisted_before_directions(*args):
        summary = json.loads((tmp_path / "test/task/summary.json").read_text())
        assert summary["valid_positions"] == 11
        assert sum("layer" not in row for row in pair.timings) == 2 * 5 * 3
        return measure_direction(*args)

    pair.measure_direction = task_is_persisted_before_directions
    pair.measure_new(model)
    pair.write({})
    write_engineering_report(tmp_path, ["test"], protocol=settings)
    with np.load(tmp_path / "test/task/groups.npz") as data:
        assert data["N"].sum() == 11
        for name in ("U", "V", "loss_change", "sham_old", "sham_new"):
            np.testing.assert_array_equal(data[name], 0)
        assert data["V"].shape == (3, 1, 1)
    with np.load(tmp_path / "test/layer_01/sequences/sequence_000001.npz") as data:
        np.testing.assert_array_equal(data["route_energy"], 0)
        assert data["random_loss"].shape == (2, 3, 2, 8)
    full_calls = [row for row in pair.timings if "layer" not in row]
    suffix_calls = [row for row in pair.timings if "layer" in row]
    assert len(full_calls) == 2 * 5 * 3
    assert len(suffix_calls) == 2 * 3 * 2 * (3 + 2 * 3)
    assert model.training
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_direction_is_forward_from_common_anchor_and_reports_realized_norms(tmp_path):
    _, model, samples = small_setup()
    settings = protocol()
    pair = EngineeringMeasurement(samples, [], tmp_path, "test", RNGSnapshot.capture(), protocol=settings, announce=False)
    pair.capture_old(model)
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.mul_(-1)
    pair.measure_new(model)
    pair.write({})
    with np.load(tmp_path / "test/layer_01/sequences/sequence_000000.npz") as data:
        assert data["route_energy"].max() > 0
        expected = data["route_loss"][:, None] - data["random_loss"].mean(2)
        # All real repetitions must agree on this deterministic CPU model.
        np.testing.assert_array_equal(expected[0], expected[1])
        np.testing.assert_allclose(data["random_realized_energy"],
            np.broadcast_to(data["route_energy"], (3, 2, 8)), rtol=2e-5, atol=1e-12)
        assert np.max(np.abs(data["observed_loss"] - data["route_loss"])) < 2e-6
    with np.load(tmp_path / "test/layer_01/groups.npz") as data:
        assert data["specificity"].shape == (2, 3, 1)
        np.testing.assert_allclose(data["specificity"], data["route_harm"] - data["random_harm"], atol=1e-14)


def test_same_norm_random_control_distinguishes_known_harmful_direction():
    delta = torch.zeros(100, 32, dtype=torch.float64)
    delta[:, 0] = 0.05
    delta[-1] = 0
    random = gaussian_direction(delta, 99)
    torch.testing.assert_close(delta.norm(dim=-1), random.norm(dim=-1), atol=1e-15, rtol=0)
    assert torch.equal(random[-1], torch.zeros(32))
    # Known downstream loss: the true direction raises the first logit of a
    # competing class. Antithetic isotropic changes have much smaller cost.
    def loss(value):
        return torch.logaddexp(torch.zeros(len(value)), value[:, 0])
    route_harm = loss(delta) - loss(torch.zeros_like(delta))
    random_harm = (loss(random) + loss(-random)) / 2 - loss(torch.zeros_like(delta))
    assert (route_harm - random_harm).mean() > 0.02


def test_crossed_bootstrap_preserves_token_weighting_and_direction_uncertainty():
    options = protocol()["statistics"]
    counts = np.array([1, 9])
    values = np.array([[[1., 18.], [3., 36.], [9., 90.]], [[2., 27.], [5., 45.], [11., 108.]]])
    result = crossed_summary({"specificity": values}, counts, options)
    stat = result["statistics"]["specificity"]
    assert stat["estimate"] == values.mean((0, 1)).sum() / 10
    errors = stat["standard_errors"]
    assert all(value > 0 for value in errors.values())
    assert result["valid_positions"] == 10
    assert result["gaussian_pairs"] == 3


def test_cancellation_requires_all_three_signs_and_separates_exclusion():
    def summary(u, v, total):
        return {"statistics": {name: {"estimate": value, "interval": [value - 0.001, value + 0.001]}
            for name, value in {"U": u, "V": v, "loss_change": total, "sham_old": 0, "sham_new": 0}.items()}}
    harmful = cancellation_description(summary(-.01, .03, .02))
    assert all(harmful["conditions"].values())
    partial = cancellation_description(summary(-.03, .01, -.02))
    assert not all(partial["conditions"].values())
    assert "排除" in partial["description"]
    absent = cancellation_description(summary(.03, .01, .04))
    assert "不具备" in absent["description"]


def test_same_document_from_two_ranks_is_one_resampling_group():
    tables = [DocumentSums(), DocumentSums()]
    for table, amount in zip(tables, (1, 3)):
        table.append({"group": "same", "valid": torch.tensor([True, False])}, {"V": np.full((2, 3, 2), amount)})
    groups, counts, columns = DocumentSums.merge([table.rows for table in tables])
    assert groups == ["same"]
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(columns["V"], 4)


def test_engineering_resume_preserves_the_actual_next_optimizer_update(tmp_path):
    config, model, _ = small_setup()
    training = config.experiment["training"]
    samples = development_samples(8, 8, 32, config.experiment["experiment"]["seed"] + 1, "development-train")
    optimizer = make_optimizer(model, training)
    cpu_update(model, optimizer, samples[:4], training, config.learning_rate(1))
    source = tmp_path / "source"
    CPUCheckpoint(source).save(model, optimizer, Progress(1, 4))
    original = (source / "step_000001/state.pt").read_bytes()
    cpu_update(model, optimizer, samples[4:], training, config.learning_rate(2))
    expected = snapshot_parameters(model)
    args = SimpleNamespace(checkpoint=source, output=tmp_path / "new", sequences=2,
                           diagnostic_sequences=0, logit_block_size=32)
    settings = protocol()
    run_engineering(config, args, settings)
    actual = torch.load(args.output / "new_master_parameters.pt", weights_only=True)
    for name in expected:
        assert torch.equal(actual[name], expected[name])
    assert (source / "step_000001/state.pt").read_bytes() == original
    report = json.loads((args.output / "effects.json").read_text())
    assert report["cpu_execution"]["task"]["execution_repeats"] == 3
    assert workload(settings, 2, 3)["full_dataset_forwards"] == 30
