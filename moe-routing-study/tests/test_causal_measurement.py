from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from moe_study.adapters.distributed_reference import MasterSlice, MasterUpdate
from moe_study.causal import ordered_measurement_indices, run_cpu
from moe_study.causal_measure import CausalMeasurement, RNGSnapshot, logit_difference, write_causal_report
from moe_study.config import RunConfig
from moe_study.reference.network_fp32 import QwenConfig, QwenReference
from moe_study.state import CPUCheckpoint, Progress, snapshot_parameters
from moe_study.train import cpu_update, development_samples, make_optimizer

ROOT = Path(__file__).resolve().parents[1]


def small_setup():
    torch.set_num_threads(1)
    torch.manual_seed(501)
    config = RunConfig.read(ROOT / "examples/cpu_experiment.yaml", ROOT / "examples/cpu_machine.yaml",
                            ROOT / "examples/cpu_schedule.yaml")
    model = QwenReference(QwenConfig(**config.experiment["development_model"]))
    samples = development_samples(2, 8, 32, 12, "causal")
    return config, model, samples


def test_rng_replay_restores_external_and_tracker_states():
    class Tracker:
        def __init__(self):
            self.states = {"model-parallel": torch.tensor([7])}

        def get_states(self):
            return self.states

        def set_states(self, value):
            self.states = value

    tracker = Tracker()
    replay = RNGSnapshot.capture(tracker)
    draws = []
    for _ in range(2):
        outside = RNGSnapshot.capture(tracker)
        with replay.replay():
            draws.append((random.random(), np.random.rand(), torch.rand(3)))
            tracker.states["model-parallel"].add_(1)
        assert tracker.states["model-parallel"].item() == 7
        assert torch.equal(torch.get_rng_state(), outside.cpu)
    assert draws[0][:2] == draws[1][:2]
    assert torch.equal(draws[0][2], draws[1][2])


def test_logit_comparison_ignores_constant_shift_and_matches_distribution_kl():
    a = torch.tensor([[1., 2., 4.], [-3., 0., 1.]], dtype=torch.float64)
    same = logit_difference(a, a + 10, block_size=1)
    torch.testing.assert_close(same["KL"], torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
    assert same["centered_logits_RMS"].eq(0).all()
    b = a + torch.tensor([0., .01, -.02])
    actual = logit_difference(a, b, block_size=1)
    expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=a),
                                                torch.distributions.Categorical(logits=b))
    torch.testing.assert_close(actual["KL"], expected, atol=1e-14, rtol=1e-8)


def test_six_real_forwards_reuse_first_support_and_leave_parameters_untouched(tmp_path):
    config, model, samples = small_setup()
    active = False
    calls = []
    original = model.causal_forward

    def forward(**kwargs):
        nonlocal active
        active = True
        calls.append(kwargs.get("supports"))
        result = original(**kwargs)
        active = False
        return result

    model.causal_forward = forward
    for layer in model.layers:
        reference = layer.mlp.reference_routes

        def outside_forward(*args, reference=reference, **kwargs):
            assert not active
            return reference(*args, **kwargs)

        layer.mlp.reference_routes = outside_forward
    pair = CausalMeasurement(samples, [0], tmp_path, "test", RNGSnapshot.capture())
    pair.capture_old(model)
    saved_support = deepcopy(pair.old[0].supports)
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.mul_(-1)
    new_parameters = snapshot_parameters(model)
    pair.measure_new(model)
    assert len(calls) == 6 * len(samples)
    # Four old forwards; then N1/F1/F1/N1 on the first sequence.
    for position in (5, 6):
        for layer in saved_support:
            assert torch.equal(calls[position][layer], saved_support[layer])
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, new_parameters[name])
    columns = pair.network.arrays()[2]
    np.testing.assert_array_equal(columns["repeat_N0_loss"], 0)
    np.testing.assert_array_equal(columns["repeat_N1_loss"], 0)
    np.testing.assert_array_equal(columns["repeat_F1_loss"], 0)
    np.testing.assert_allclose(columns["U_1"] + columns["V_1"], columns["loss_change_1"], atol=1e-14)
    assert np.any(columns["updated_route_disagreement"] > 0)
    with np.load(tmp_path / "test/diagnostics/sequence_000000.npz") as diagnostic:
        assert diagnostic["J_local_reference"].max() > 0
        assert diagnostic["repeat_N1_output_difference_energy"].max() == 0
        assert diagnostic["N0_support"].shape == (3, 8, 2)
    pair.write(config.experiment["measurement"])
    write_causal_report(tmp_path, ["test"])
    summary = json.loads((tmp_path / "effects.json").read_text())["test"]
    assert summary["valid_positions"] == 16
    assert summary["execution_repeats"] == 2


def test_no_update_gives_zero_effect_with_actual_repeats(tmp_path):
    config, model, samples = small_setup()
    samples[1]["valid"][3:] = False
    pair = CausalMeasurement(samples, [0], tmp_path, "test", RNGSnapshot.capture())
    pair.capture_old(model)
    pair.measure_new(model)
    columns = pair.network.arrays()[2]
    for metric in ("V_1", "V_2", "U", "loss_change", "route_1_KL", "route_2_centered_logits_RMS"):
        np.testing.assert_array_equal(columns[metric], 0)
    pair.write(config.experiment["measurement"])
    summary = json.loads((tmp_path / "test/network/summary.json").read_text())
    assert summary["valid_positions"] == 11


def test_diagnostic_order_aligns_collectives_without_changing_sample_ids():
    index = [{"valid_positions": value} for value in [2, 8, 3, 8, 8, 8, 4, 1]]
    order, diagnostic = ordered_measurement_indices(index, 8, 4, 8)
    assert diagnostic == [1, 3, 4, 5]
    assert sorted(order) == list(range(8))
    assert all(order[rank] in diagnostic for rank in range(4))
    assert all(len(order[rank::4]) == 2 for rank in range(4))


def test_master_endpoints_read_stored_values_without_interpolation(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    new = torch.tensor([1.0])
    child = SimpleNamespace(_get_main_param_and_optimizer_states=lambda p: {"param": new})
    update = MasterUpdate.__new__(MasterUpdate)
    update.model = model
    update.slices = {model.weight: MasterSlice(0, 1, torch.tensor([1e8]), child, model.weight)}
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, group: None)
    assert update.at("weight", 0, None, None).item() == 1e8
    assert update.at("weight", 1, None, None).item() == 1.0


def test_cpu_resume_updates_once_and_preserves_source_checkpoint(tmp_path):
    config, model, _ = small_setup()
    training = config.experiment["training"]
    seed = config.experiment["experiment"]["seed"]
    train_samples = development_samples(8, 8, 32, seed + 1, "development-train")
    optimizer = make_optimizer(model, training)
    cpu_update(model, optimizer, train_samples[:4], training, config.learning_rate(1))
    source = tmp_path / "source"
    CPUCheckpoint(source).save(model, optimizer, Progress(1, 4))
    original = (source / "step_000001/state.pt").read_bytes()
    cpu_update(model, optimizer, train_samples[4:], training, config.learning_rate(2))
    expected = snapshot_parameters(model)
    args = SimpleNamespace(checkpoint=source, output=tmp_path / "new", sequences=2,
                           diagnostic_sequences=1, logit_block_size=4)
    run_cpu(config, args)
    actual = torch.load(args.output / "new_master_parameters.pt", weights_only=True)
    for name in expected:
        assert torch.equal(actual[name], expected[name])
    assert (source / "step_000001/state.pt").read_bytes() == original
    metadata = json.loads((args.output / "run.json").read_text())
    assert metadata["resumed_step"] == 1 and metadata["final_step"] == 2
    assert metadata["full_dataset_forwards"] == 12
