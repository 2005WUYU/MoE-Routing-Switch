"""Explicit CPU development and Megatron cluster entry points."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from moe_study.config import RunConfig
from moe_study.measure import PairedMeasurement, ResultWriter, batch, interventions
from moe_study.reference.network_fp32 import QwenConfig, QwenReference
from moe_study.scan import run_scan
from moe_study.state import CPUCheckpoint, Progress, interpolate_parameters, preserve_rng, record_code, snapshot_parameters


def development_samples(count: int, length: int, vocabulary: int, seed: int, prefix: str) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randint(vocabulary, (count, length + 1), generator=generator)
    return [{"tokens": row[:-1], "labels": row[1:], "valid": torch.ones(length, dtype=torch.bool),
             "group": f"{prefix}-{index // 2}", "sequence": index} for index, row in enumerate(rows)]


def make_optimizer(model, training: dict):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": training["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=training["learning_rate"], betas=training["betas"], eps=training["epsilon"])


def cpu_update(model, optimizer, samples, training: dict, learning_rate: float) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    loss_sum, auxiliary_sum = 0.0, 0.0
    for sample in samples:
        result = model(**batch(sample, "cpu"), balance_coefficient=training["load_balance_coefficient"])
        loss = result.losses[:, sample["valid"]].mean()
        ((loss + result.balance_loss) / len(samples)).backward()
        loss_sum += loss.item() / len(samples)
        auxiliary_sum += result.balance_loss.item() / len(samples)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip_norm"]))
    optimizer.step()
    return {"lm_loss": loss_sum, "balance_loss": auxiliary_sum, "learning_rate": learning_rate,
            "grad_norm_before_clip": grad_norm,
            "gradient_scale": min(1.0, training["grad_clip_norm"] / (grad_norm + 1e-6))}


def run_cpu(config: RunConfig, segment_name: str, output: Path):
    """Small, genuine AdamW updates solely for software development on a Mac CPU."""
    torch.set_num_threads(config.machine["machine"]["cpu_threads"])
    seed = config.experiment["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_config = QwenConfig(**config.experiment["development_model"])
    model = QwenReference(model_config)
    training, measurement = config.experiment["training"], config.experiment["measurement"]
    optimizer = make_optimizer(model, training)
    checkpoint = CPUCheckpoint(output / "checkpoints")
    segment = config.segment(segment_name)
    progress = checkpoint.load(model, optimizer) if segment["resume"] else Progress()
    total_steps = config.experiment["experiment"]["total_steps"]
    batch_size = training["global_sequences_per_step"]
    training_samples = development_samples(total_steps * batch_size, training["sequence_length"],
                                           model_config.vocab_size, seed + 1, "development-train")
    measurement_samples = development_samples(measurement["large_sequences"], training["sequence_length"],
                                              model_config.vocab_size, seed + 2, "development-measurement")
    writer = ResultWriter(output / "measurements", measurement)
    output.mkdir(parents=True, exist_ok=True)
    if not segment["resume"]:
        (output / "training.jsonl").write_text("")
    (output / f"run_{segment_name}.json").write_text(json.dumps({
        "purpose": "CPU software development; not H20 scientific results", "config": config.expanded(),
        "code": record_code(output), "torch_version": torch.__version__, "segment": segment_name,
        "resumed_step": progress.step,
    }, indent=2))
    final_step = measurement["interpolation_update"][1]
    for step in range(progress.step + 1, segment["end_step"] + 1):
        started = time.perf_counter()
        count = config.measurement_sequences(step)
        if count:
            pair = PairedMeasurement(model, measurement_samples[:count], measurement)
            old = pair.capture()
        if step == final_step and segment["final_analysis"]:
            master0 = snapshot_parameters(model)
            old_supports = [{number: layer.support for number, layer in seq.layers.items()} for seq in old.sequences]
            endpoint0 = old.network_outputs()
        start = progress.consumed_sequences
        record = cpu_update(model, optimizer, training_samples[start:start + batch_size], training, config.learning_rate(step))
        progress.step, progress.consumed_sequences = step, start + batch_size
        if count:
            tables = pair.finish(old)
            writer.write(f"step_{step:06d}", tables, retain_tokens=step == final_step)
            if any(a <= step <= b for a, b in measurement["consecutive_windows"]):
                writer.write(f"window_step_{step:06d}", tables, False, measurement["window_sequences"])
            del tables
        if step == final_step and segment["final_analysis"]:
            master1 = snapshot_parameters(model)
            endpoint1 = pair.latest_outputs
            writer.write("interventions", interventions(model, measurement_samples, old_supports, measurement, seed + 3,
                                                       baseline_outputs=endpoint1), False)
            with preserve_rng():
                reference = QwenReference(model_config).float()
            completed = run_scan(reference, lambda alpha: reference.load_state_dict(interpolate_parameters(master0, master1, alpha)),
                                 measurement_samples, measurement, writer, production_endpoints=[endpoint0, endpoint1])
            (output / "analysis_progress.json").write_text(json.dumps({"completed_alphas": completed,
                                                                       "requested_alphas": measurement["alphas"]}, indent=2))
        record.update(step=step, consumed_sequences=progress.consumed_sequences, elapsed_seconds=time.perf_counter() - started)
        with (output / "training.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    written = checkpoint.save(model, optimizer, progress)
    print(json.dumps({"completed": asdict(progress), "checkpoint_bytes": written}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("machine", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("segment")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = RunConfig.read(args.experiment, args.machine, args.schedule)
    if args.print_config:
        print(json.dumps({**config.expanded(), "layout": config.layout()}, indent=2))
        return
    backend = config.machine["machine"]["backend"]
    output = args.output or Path(config.machine["machine"]["project_root"]) / "outputs" / config.experiment["experiment"]["name"]
    if backend == "cpu_development":
        run_cpu(config, args.segment, output)
    else:
        from moe_study.adapters.megatron_qwen import run_megatron
        run_megatron(config, args.segment, output)


if __name__ == "__main__":
    main()
