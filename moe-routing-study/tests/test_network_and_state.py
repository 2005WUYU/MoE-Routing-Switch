import copy
from pathlib import Path

import torch

from moe_study.config import RunConfig
from moe_study.measure import PairedMeasurement, batch, interventions
from moe_study.reference.network_fp32 import QwenConfig, QwenReference
from moe_study.state import CPUCheckpoint, Progress, interpolate_parameters, snapshot_parameters
from moe_study.train import cpu_update, development_samples, make_optimizer

ROOT = Path(__file__).resolve().parents[1]


def setup_development():
    torch.set_num_threads(1)
    torch.manual_seed(101)
    config = RunConfig.read(ROOT / "examples/cpu_experiment.yaml", ROOT / "examples/cpu_machine.yaml", ROOT / "examples/cpu_schedule.yaml")
    model = QwenReference(QwenConfig(**config.experiment["development_model"]))
    samples = development_samples(4, 8, 32, 5, "test")
    return model, samples, config


def test_full_network_counterfactual_recomputes_downstream_inputs():
    model, samples, config = setup_development()
    sample = samples[0]
    actual = model(**batch(sample, "cpu"))
    forced = {layer: (ids + 1) % 4 for layer, ids in actual.supports.items()}
    inputs = []
    model(**batch(sample, "cpu"), observer=lambda n, l, h, s, a, y: inputs.append(h.detach().clone()))
    changed = []
    model(**batch(sample, "cpu"), supports=forced, observer=lambda n, l, h, s, a, y: changed.append(h.detach().clone()))
    torch.testing.assert_close(inputs[0], changed[0])
    assert not torch.equal(inputs[1], changed[1])
    assert model.layers[0].self_attn.q_proj.out_features == 32
    assert model.config.hidden_size == 16


def test_paired_no_update_zero_and_measurements_preserve_rng():
    model, samples, config = setup_development()
    pair = PairedMeasurement(model, samples, config.experiment["measurement"])
    rng = torch.get_rng_state()
    old = pair.capture()
    tables = pair.finish(old)
    assert torch.equal(rng, torch.get_rng_state())
    for name, table in tables.items():
        columns = table.arrays()[2]
        if "J" in columns:
            assert (columns["J"] == 0).all()
            assert (columns["T"] == 0).all()
        else:
            assert (columns["V"] == 0).all()


def test_checkpoint_resume_matches_uninterrupted_adamw(tmp_path):
    model, samples, config = setup_development()
    training = config.experiment["training"]
    optimizer = make_optimizer(model, training)
    cpu_update(model, optimizer, samples[:2], training, config.learning_rate(1))
    checkpoint = CPUCheckpoint(tmp_path)
    checkpoint.save(model, optimizer, Progress(1, 2))
    cpu_update(model, optimizer, samples[2:], training, config.learning_rate(2))
    expected = snapshot_parameters(model)
    resumed, _, _ = setup_development()
    resumed_optimizer = make_optimizer(resumed, training)
    progress = checkpoint.load(resumed, resumed_optimizer)
    assert progress == Progress(1, 2)
    cpu_update(resumed, resumed_optimizer, samples[2:], training, config.learning_rate(2))
    for name, parameter in resumed.named_parameters():
        assert torch.equal(parameter, expected[name])
    checkpoint.save(resumed, resumed_optimizer, Progress(2, 4))
    assert not (tmp_path / "step_000001").exists()


def test_interpolation_starts_from_original_master_and_does_not_touch_optimizer():
    old = {"w": torch.tensor([1.0001, 2.0002])}
    new = {"w": torch.tensor([1.0003, 2.0006])}
    first = interpolate_parameters(old, new, .125)["w"]
    interpolate_parameters(old, new, 1.)
    assert torch.equal(interpolate_parameters(old, new, .125)["w"], first)
    assert torch.equal(old["w"], torch.tensor([1.0001, 2.0002]))
