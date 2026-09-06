"""Bridge 0.2 / Core 0.15: resume once, collect six forwards in each precision."""

from contextlib import ExitStack
import json
import time

import torch
import torch.distributed as dist

from moe_study.adapters.distributed_reference import MasterUpdate, create_reference, load_reference_master
from moe_study.adapters.megatron_qwen import MegatronEvaluation, build_bridge_config
from moe_study.causal import ordered_measurement_indices
from moe_study.causal_measure import CausalMeasurement, RNGSnapshot, write_causal_report
from moe_study.data import TokenDataset
from moe_study.reference.expert_fp32 import ieee_fp32
from moe_study.state import record_code


def run(config, args):
    from megatron.bridge.data.utils import get_dataset_provider
    from megatron.bridge.training.config import runtime_config_update
    from megatron.bridge.training.setup import setup
    from megatron.bridge.training.state import GlobalState
    from megatron.bridge.training.train import train_step
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.rerun_state_machine import get_rerun_state_machine
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    started = time.perf_counter()
    training = config.experiment["training"]
    cfg, architecture = build_bridge_config(config, "CAUSAL", args.output)
    cfg.checkpoint.load = str(args.checkpoint)
    cfg.checkpoint.ckpt_step = int((args.checkpoint / "latest_checkpointed_iteration.txt").read_text())
    # Preserve the original scheduler definition while allowing the next batch.
    cfg.train.train_iters = cfg.checkpoint.ckpt_step + 1
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.scheduler.use_checkpoint_opt_param_scheduler = True
    runtime_config_update(cfg)
    state = GlobalState()
    state.cfg = cfg
    runtime = setup(state, get_dataset_provider(cfg.dataset))
    model, optimizer, scheduler = runtime.model, runtime.optimizer, runtime.scheduler
    raw_model = model[0].module.module
    rank, world = dist.get_rank(), dist.get_world_size()
    ep = parallel_state.get_expert_model_parallel_group()
    dense = parallel_state.get_data_parallel_group()
    expert_dp = parallel_state.get_expert_data_parallel_group()
    scalar_group = dist.new_group(backend="gloo")

    def gather(value):
        pieces = [None] * world if rank == 0 else None
        dist.gather_object(value, pieces, dst=0, group=scalar_group)
        return pieces

    args.output.mkdir(parents=True, exist_ok=True)
    step = state.train_state.step + 1
    rng = RNGSnapshot.capture(get_cuda_rng_tracker())
    torch.cuda.reset_peak_memory_stats()
    data = TokenDataset(config.experiment["data"]["root"], "measurement")
    order, diagnostic = ordered_measurement_indices(data.index, args.sequences, args.diagnostic_sequences, data.length)
    samples = [data[index] for index in order[rank::world]]
    engines = ["bf16_execution", "fp32_reference"]
    pairs = [CausalMeasurement(samples, diagnostic, args.output, engine, rng, "cuda", args.logit_block_size, rank == 0)
             for engine in engines]
    groups = gather({"rank": rank, "ep": dist.get_process_group_ranks(ep),
                     "dense": dist.get_process_group_ranks(dense), "expert_dp": dist.get_process_group_ranks(expert_dp)})
    if rank == 0:
        (args.output / "run.json").write_text(json.dumps({"purpose": "One real update with repeated route counterfactuals",
            "resumed_step": step - 1, "requested_step": step, "checkpoint_source": str(args.checkpoint),
            "engines": engines, "full_dataset_forwards": 12, "sequence_order": order,
            "diagnostic_sequences": diagnostic, "config": config.expanded(), "process_groups": groups,
            "torch_version": torch.__version__, "code": record_code(args.output),
            "old_support": "first N0 in each engine, reused for both F1 repeats",
            "new_order_per_sequence": ["N1_1", "F1_1", "F1_2", "N1_2"]}, indent=2))
    evaluation = MegatronEvaluation(raw_model, ep)
    master = MasterUpdate(raw_model, optimizer)
    pairs[0].capture_old(evaluation)
    with rng.replay(), ieee_fp32("cuda"):
        reference = create_reference(architecture, ep)
        load_reference_master(reference, master, 0, dense, expert_dp)
        pairs[1].capture_old(reference)
    del reference
    torch.cuda.empty_cache()

    def forward_step(iterator, chunk):
        sample = next(iterator)
        tokens, labels, valid = (sample[name].cuda() for name in ("tokens", "labels", "valid"))
        positions = torch.arange(tokens.shape[-1], device="cuda")[None].expand_as(tokens)
        losses = chunk(input_ids=tokens, position_ids=positions, attention_mask=None, labels=labels)

        def loss_function(values):
            total, count = (values.float() * valid).sum(), valid.sum()
            return total, count, {"lm_loss": torch.stack((total.detach(), count))}

        return losses, loss_function

    for chunk in model:
        chunk.train()
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate(step) * group.get("lr_mult", 1.0)
    state.timers("interval-time", log_level=0).start(barrier=True)
    get_rerun_state_machine().current_iteration = state.train_state.step
    load_counts = torch.zeros(architecture.num_hidden_layers, architecture.num_experts, device="cuda", dtype=torch.int64)
    with ExitStack() as hooks:
        for number, layer in enumerate(raw_model.decoder.layers):
            def record_load(router, inputs, result, number=number):
                if not torch.is_grad_enabled():
                    load_counts[number] += result[1].sum(0)
            hooks.callback(layer.mlp.router.register_forward_hook(record_load).remove)
        train_started = time.perf_counter()
        result = train_step(forward_step, runtime.train_data_iterator, model, optimizer, scheduler,
                            state, get_forward_backward_func())
        torch.cuda.synchronize()
    losses, skipped, _, _, _, grad_norm, _ = result
    state.train_state.step = step
    state.train_state.consumed_train_samples += training["global_sequences_per_step"]
    dist.all_reduce(load_counts, group=dense)
    if rank == 0:
        (args.output / "training.json").write_text(json.dumps({"step": step,
            "consumed_sequences": state.train_state.consumed_train_samples,
            "lm_loss": float(losses["lm_loss"]), "learning_rate": config.learning_rate(step),
            "native_skipped_update": skipped, "grad_norm_before_clip": grad_norm,
            "train_seconds": time.perf_counter() - train_started, "expert_tokens": load_counts.cpu().tolist()}, indent=2))
    master.save_new_endpoint(args.output / "new_master_shards", dense, expert_dp)
    pairs[0].measure_new(evaluation)
    pairs[0].write(config.experiment["measurement"], gather)
    with rng.replay(), ieee_fp32("cuda"):
        reference = create_reference(architecture, ep)
        load_reference_master(reference, master, 1, dense, expert_dp)
        pairs[1].measure_new(reference)
    pairs[1].write(config.experiment["measurement"], gather)
    resources = gather({"rank": rank, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    if rank == 0:
        write_causal_report(args.output, engines)
        (args.output / "completion.json").write_text(json.dumps({"final_step": step,
            "native_skipped_update": skipped, "engines": engines, "full_dataset_forwards": 12,
            "elapsed_seconds": time.perf_counter() - started, "resources": resources,
            "new_endpoint": "new_master_shards", "old_endpoint": str(args.checkpoint),
            "new_endpoint_scope": "FP32 model master shards; no new optimizer checkpoint"}, indent=2))
        print(f"One-update comparison written to {args.output}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
