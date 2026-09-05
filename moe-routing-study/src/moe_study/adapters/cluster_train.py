"""Cluster orchestration using native Bridge setup, optimizer step and checkpoints."""

from contextlib import ExitStack
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.distributed as dist

from moe_study.adapters.distributed_reference import MasterUpdate, create_reference, load_reference_master
from moe_study.adapters.megatron_qwen import MegatronEvaluation, build_bridge_config
from moe_study.data import TokenDataset
from moe_study.measure import PairedMeasurement, ResultWriter, batch, interventions
from moe_study.scan import run_scan
from moe_study.state import preserve_rng, record_code


def run(config, segment_name, output: Path):
    from megatron.bridge.data.utils import get_dataset_provider
    from megatron.bridge.training.checkpointing import save_checkpoint
    from megatron.bridge.training.config import runtime_config_update
    from megatron.bridge.training.setup import setup
    from megatron.bridge.training.state import GlobalState
    from megatron.bridge.training.train import checkpoint_and_decide_exit, train_step
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    segment = config.segment(segment_name)
    training, measurement = config.experiment["training"], config.experiment["measurement"]
    cfg, architecture = build_bridge_config(config, segment_name, output)
    if segment["resume"]:
        # Read the native saved position; the plan's nominal start is never used as progress.
        cfg.checkpoint.ckpt_step = int((output / "checkpoints/latest_checkpointed_iteration.txt").read_text())
    runtime_config_update(cfg)
    state = GlobalState()
    state.cfg = cfg
    runtime = setup(state, get_dataset_provider(cfg.dataset))
    model, optimizer, scheduler = runtime.model, runtime.optimizer, runtime.scheduler
    # This explicit BF16+DDP recipe has precisely these two wrappers.
    raw_model = model[0].module.module
    rank, world = dist.get_rank(), dist.get_world_size()
    ep_group = parallel_state.get_expert_model_parallel_group()
    dense_group = parallel_state.get_data_parallel_group()
    expert_data_group = parallel_state.get_expert_data_parallel_group()
    evaluation = MegatronEvaluation(raw_model, ep_group)
    scalar_group = dist.new_group(backend="gloo")

    def gather(value):
        pieces = [None] * world if rank == 0 else None
        dist.gather_object(value, pieces, dst=0, group=scalar_group)
        return pieces

    writer = ResultWriter(output / "measurements", measurement, gather)
    actual_groups = gather({"rank": rank, "ep": dist.get_process_group_ranks(ep_group),
                            "expert_dp": dist.get_process_group_ranks(expert_data_group),
                            "dp": dist.get_process_group_ranks(dense_group)})
    measurement_data = TokenDataset(config.experiment["data"]["root"], "measurement")
    all_local_samples = [measurement_data[index] for index in range(rank, measurement["large_sequences"], world)]
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / f"process_groups_{segment_name}.json").write_text(json.dumps(actual_groups, indent=2))
        (output / f"run_{segment_name}.json").write_text(json.dumps({
            "purpose": "Qwen3 continued language-model training; freshly initialized AdamW at X0",
            "config": config.expanded(), "code": record_code(output), "segment": segment_name,
            "resumed_step": state.train_state.step, "torch_version": torch.__version__,
            "layout": config.layout(),
        }, indent=2))
        print(json.dumps(config.layout()), flush=True)
    state.timers("interval-time", log_level=0).start(barrier=True)
    get_rerun_state_machine().current_iteration = state.train_state.step
    torch.cuda.reset_peak_memory_stats()
    segment_started = time.perf_counter()
    completed_alphas, completed_interventions = [], []

    def time_available():
        minutes = (time.time() - state.start_time) / 60
        expired = torch.tensor(int(minutes >= cfg.train.exit_duration_in_mins), device="cuda")
        dist.all_reduce(expired, op=dist.ReduceOp.MAX)
        return not bool(expired.item())

    def native_exit():
        return checkpoint_and_decide_exit(state, model, optimizer, scheduler,
            state.train_state.floating_point_operations_so_far, runtime.checkpointing_context, runtime.train_data_iterator)

    def forward_step(iterator, chunk):
        data = next(iterator)
        tokens, labels, valid = (data[key].cuda() for key in ("tokens", "labels", "valid"))
        positions = torch.arange(tokens.shape[-1], device="cuda")[None].expand_as(tokens)
        output = chunk(input_ids=tokens, position_ids=positions, attention_mask=None, labels=labels)

        def loss_function(losses):
            total = (losses.float() * valid).sum()
            count = valid.sum()
            # Core divides the loss sum by token count and microbatch count.
            return total, count, {"lm_loss": torch.stack((total.detach(), count))}

        return output, loss_function

    exited = False
    for step in range(state.train_state.step + 1, segment["end_step"] + 1):
        if native_exit():
            exited = True
            break
        count = config.measurement_sequences(step)
        local_samples = [sample for sample in all_local_samples if sample["sequence"] < count]
        paired_started = time.perf_counter()
        if count:
            pair = PairedMeasurement(evaluation, local_samples, measurement, "cuda")
            captured = pair.capture()
        final_step = step == measurement["interpolation_update"][1] and segment["final_analysis"]
        if final_step:
            master_update = MasterUpdate(raw_model, optimizer)
            old_supports = [{number: layer.support for number, layer in sequence.layers.items()} for sequence in captured.sequences]
            endpoint0 = captured.network_outputs()
        capture_seconds = time.perf_counter() - paired_started
        for chunk in model:
            chunk.train()
        # Choose the LR for the upcoming update explicitly; the first update is lr/32.
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate(step) * group.get("lr_mult", 1.0)
        load_counts = torch.zeros(architecture.num_hidden_layers, architecture.num_experts, device="cuda", dtype=torch.int64)
        with ExitStack() as hooks:
            for number, layer in enumerate(raw_model.decoder.layers):
                def record_load(router, inputs, result, number=number):
                    # Full activation recomputation's first forward has gradients disabled.
                    # Count it once; backward recomputation has gradients enabled.
                    if not torch.is_grad_enabled():
                        load_counts[number] += result[1].sum(0)
                handle = layer.mlp.router.register_forward_hook(record_load)
                hooks.callback(handle.remove)
            started = time.perf_counter()
            result = train_step(forward_step, runtime.train_data_iterator, model, optimizer, scheduler,
                                state, get_forward_backward_func())
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
        losses, skipped, _, _, _, grad_norm, _ = result
        state.train_state.step = step
        state.train_state.consumed_train_samples += training["global_sequences_per_step"]
        dist.all_reduce(load_counts, group=dense_group)
        if rank == 0:
            with (output / "training.jsonl").open("a") as log:
                log.write(json.dumps({"step": step, "consumed_sequences": state.train_state.consumed_train_samples,
                    "lm_loss": float(losses["lm_loss"]), "learning_rate": config.learning_rate(step),
                    "grad_norm_before_clip": grad_norm, "gradient_scale": min(1.0, training["grad_clip_norm"] / (grad_norm + 1e-6)),
                    "native_skipped_update": skipped, "train_seconds": train_seconds,
                    "capture_seconds": capture_seconds, "expert_tokens": load_counts.cpu().tolist(),
                    "empty_expert_count": int((load_counts == 0).sum())}) + "\n")
        paired_started = time.perf_counter()
        if count:
            tables = pair.finish(captured)
            writer.write(f"step_{step:06d}", tables, retain_tokens=final_step)
            if any(a <= step <= b for a, b in measurement["consecutive_windows"]):
                writer.write(f"window_step_{step:06d}", tables, False, measurement["window_sequences"])
            del tables
        if final_step:
            endpoint1 = pair.latest_outputs
            if time_available():
                intervention_tables = interventions(evaluation, local_samples, old_supports, measurement,
                                                   config.experiment["experiment"]["seed"] + 3, "cuda", endpoint1)
                writer.write("interventions", intervention_tables, False)
                completed_interventions = list(intervention_tables)
            if time_available():
                with preserve_rng():
                    reference = create_reference(architecture, ep_group)
                completed_alphas = run_scan(reference,
                    lambda alpha: load_reference_master(reference, master_update, alpha, dense_group, expert_data_group),
                    local_samples, measurement, writer, "cuda", [endpoint0, endpoint1], time_available)
                del reference
            del master_update
        if rank == 0:
            with (output / "measurement_timing.jsonl").open("a") as log:
                log.write(json.dumps({"step": step, "post_update_seconds": time.perf_counter() - paired_started}) + "\n")
    if not exited:
        save_checkpoint(state, model, optimizer, scheduler, state.train_state.floating_point_operations_so_far,
                        runtime.checkpointing_context, train_data_iterator=runtime.train_data_iterator)
    local_resource = {"rank": rank, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    resources = gather(local_resource)
    if rank == 0:
        # One post-run disk accounting pass; no directory scans inside training.
        bytes_written = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        (output / f"resources_{segment_name}.json").write_text(json.dumps({"ranks": resources,
            "elapsed_after_setup_seconds": time.perf_counter() - segment_started,
            "elapsed_including_setup_seconds": time.time() - state.start_time,
            "final_step": state.train_state.step, "run_directory_bytes_at_exit": bytes_written,
            "disk_peak": "not sampled; report checkpoint write overlap separately"}, indent=2))
        (output / f"analysis_progress_{segment_name}.json").write_text(json.dumps({
            "completed_alphas": completed_alphas, "requested_alphas": measurement["alphas"],
            "completed_interventions": completed_interventions, "final_step": state.train_state.step}, indent=2))
    dist.barrier()
    dist.destroy_process_group()
