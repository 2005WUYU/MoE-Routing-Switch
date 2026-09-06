"""Resume one optimizer update and compare repeated natural/old-support forwards."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import yaml

from moe_study.causal_measure import CausalMeasurement, RNGSnapshot, write_causal_report
from moe_study.config import RunConfig


def ordered_measurement_indices(index, count, diagnostic_count, length):
    """Place full-length diagnostic sequences in complete distributed blocks.

    Original sequence IDs are retained. With 64 ranks, use counts divisible by 64
    so every rank participates in each diagnostic EP collective.
    """
    diagnostic = [i for i in range(count) if index[i]["valid_positions"] == length][:diagnostic_count]
    selected = set(diagnostic)
    return diagnostic + [i for i in range(count) if i not in selected], diagnostic


def run_cpu(config, args):
    """Exercise the same repeated-forward controller on a saved small CPU model."""
    from moe_study.reference.network_fp32 import QwenConfig, QwenReference
    from moe_study.state import CPUCheckpoint, snapshot_parameters
    from moe_study.train import cpu_update, development_samples, make_optimizer

    torch.set_num_threads(config.machine["machine"]["cpu_threads"])
    architecture = QwenConfig(**config.experiment["development_model"])
    model = QwenReference(architecture)
    training = config.experiment["training"]
    optimizer = make_optimizer(model, training)
    progress = CPUCheckpoint(args.checkpoint).load(model, optimizer)
    step = progress.step + 1
    seed = config.experiment["experiment"]["seed"]
    rng = RNGSnapshot.capture()
    with rng.replay():
        reference = deepcopy(model)
    samples = development_samples(args.sequences, training["sequence_length"], architecture.vocab_size,
                                  seed + 2, "development-measurement")
    diagnostic = [sample["sequence"] for sample in samples[:args.diagnostic_sequences]]
    engines = ["cpu_execution", "cpu_reference"]
    pairs = [CausalMeasurement(samples, diagnostic, args.output, engine, rng, "cpu", args.logit_block_size)
             for engine in engines]
    pairs[0].capture_old(model)
    pairs[1].capture_old(reference)
    train_samples = development_samples(step * training["global_sequences_per_step"], training["sequence_length"],
                                        architecture.vocab_size, seed + 1, "development-train")
    start = progress.consumed_sequences
    record = cpu_update(model, optimizer, train_samples[start:start + training["global_sequences_per_step"]],
                        training, config.learning_rate(step))
    progress.step, progress.consumed_sequences = step, start + training["global_sequences_per_step"]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot_parameters(model), args.output / "new_master_parameters.pt")
    pairs[0].measure_new(model)
    reference.load_state_dict(model.state_dict())
    pairs[1].measure_new(reference)
    for pair in pairs:
        pair.write(config.experiment["measurement"])
    (args.output / "run.json").write_text(json.dumps({"purpose": "CPU software development, not H20 evidence",
        "resumed_step": step - 1, "final_step": step, "training": record, "engines": engines,
        "full_dataset_forwards": 12, "checkpoint_source": str(args.checkpoint),
        "config": config.expanded()}, indent=2))
    write_causal_report(args.output, engines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("machine", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Existing checkpoints directory, read for resume")
    parser.add_argument("--output", required=True, type=Path, help="New measurement directory")
    parser.add_argument("--sequences", type=int, default=1024)
    parser.add_argument("--diagnostic-sequences", type=int, default=64)
    parser.add_argument("--logit-block-size", type=int, default=32)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = RunConfig(yaml.safe_load(args.experiment.read_text()), yaml.safe_load(args.machine.read_text()),
                       {"segments": {"CAUSAL": {"resume": True, "final_analysis": False}}})
    if args.print_config:
        print(json.dumps({"config": config.expanded(), "layout": config.layout(),
            "checkpoint": str(args.checkpoint), "output": str(args.output), "optimizer_steps": 1,
            "full_dataset_forwards": 12, "sequences": args.sequences,
            "diagnostic_sequences": args.diagnostic_sequences}, indent=2))
        return
    if config.machine["machine"]["backend"] == "cpu_development":
        run_cpu(config, args)
    else:
        from moe_study.adapters.causal_train import run
        run(config, args)


if __name__ == "__main__":
    main()
