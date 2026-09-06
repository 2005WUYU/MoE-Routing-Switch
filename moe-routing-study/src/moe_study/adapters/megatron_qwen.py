"""Bridge 0.2.0 / Core 0.15 Qwen3 binding; CUDA imports live in cluster functions.

Framework-specific work is limited to configuration, execution dispatch, and
model/state interfaces. The common measurement layer owns no Megatron imports.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from moe_study.adapters.distributed_reference import expert_parallel_routes
from moe_study.data import TokenDataset
from moe_study.reference.expert_fp32 import ExpertWeights, ieee_fp32
from moe_study.reference.network_fp32 import NetworkOutput, QwenConfig
from moe_study.routing import selected_gates


def build_bridge_config(config, segment_name, output):
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_30b_a3b_pretrain_config
    from megatron.bridge.training.config import DatasetProvider

    @dataclass
    class PackedDatasetProvider(DatasetProvider):
        root: str = ""
        dataloader_type: str = "single"

        def build_datasets(self, context):
            return TokenDataset(self.root, "train"), None, None

    training = config.experiment["training"]
    experiment = config.experiment["experiment"]
    machine = config.machine["machine"]
    source = experiment["model_source"]
    segment = config.segment(segment_name)
    metadata = json.loads((Path(source) / "config.json").read_text())
    cfg = qwen3_30b_a3b_pretrain_config(
        hf_path=source, dir=str(output.parent), name=output.name,
        tensor_model_parallel_size=machine["tensor_parallel"],
        pipeline_model_parallel_size=machine["pipeline_parallel"],
        context_parallel_size=machine["context_parallel"],
        expert_model_parallel_size=machine["expert_parallel"],
        expert_tensor_parallel_size=machine["expert_tensor_parallel"],
        sequence_parallel=False, enable_recompute=True, mock=False,
        train_iters=experiment["total_steps"], global_batch_size=training["global_sequences_per_step"],
        micro_batch_size=machine["microbatch_sequences_per_gpu"], seq_length=training["sequence_length"],
        lr=training["learning_rate"], min_lr=0.0, lr_warmup_iters=training["warmup_steps"],
        eval_interval=experiment["total_steps"] + 1, save_interval=experiment["total_steps"] + 1,
    )
    # Register HF weight loading on first construction, before Adam state exists.
    # Resume jobs read the native model+optimizer checkpoint instead.
    if not segment["resume"]:
        bridge = AutoBridge.from_hf_pretrained(source)
        cfg.model.register_pre_wrap_hook(lambda models: _load_initial_weights(bridge, models))
    cfg.model.kv_channels = metadata["head_dim"]
    cfg.model.hidden_dropout = 0.0
    cfg.model.attention_dropout = 0.0
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_router_pre_softmax = False  # selected-softmax, not dense-softmax mass
    cfg.model.moe_router_score_function = "softmax"
    cfg.model.moe_router_load_balancing_type = "seq_aux_loss"
    # Core adds the auxiliary autograd contribution at every layer; average layers here.
    cfg.model.moe_aux_loss_coeff = training["load_balance_coefficient"] / metadata["num_hidden_layers"]
    cfg.model.moe_router_enable_expert_bias = False
    cfg.model.moe_expert_capacity_factor = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_input_jitter_eps = None
    cfg.model.moe_z_loss_coeff = None
    cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2 = training["betas"]
    cfg.optimizer.adam_eps = training["epsilon"]
    cfg.optimizer.weight_decay = training["weight_decay"]
    cfg.optimizer.clip_grad = training["grad_clip_norm"]
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.bf16 = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.comm_overlap = None
    cfg.rerun_state_machine.rerun_mode = "disabled"
    cfg.rerun_state_machine.check_for_nan_in_loss = False
    cfg.scheduler.lr_decay_style = "constant"
    cfg.scheduler.lr_decay_iters = experiment["total_steps"]
    cfg.scheduler.start_weight_decay = training["weight_decay"]
    cfg.scheduler.end_weight_decay = training["weight_decay"]
    cfg.dataset = PackedDatasetProvider(root=config.experiment["data"]["root"], num_workers=0)
    cfg.rng.seed = experiment["seed"]
    cfg.train.eval_iters = 0
    hours, minutes, seconds = map(int, machine["walltime_per_job"].split(":"))
    cfg.train.exit_duration_in_mins = hours * 60 + minutes + seconds / 60 - machine["normal_exit_reserve_minutes"]
    cfg.train.check_weight_hash_across_dp_replicas_interval = None
    cfg.checkpoint.save = str(output / "checkpoints")
    cfg.checkpoint.load = str(output / "checkpoints") if segment["resume"] else None
    cfg.checkpoint.save_interval = None
    cfg.checkpoint.most_recent_k = 1
    cfg.checkpoint.async_save = False
    cfg.checkpoint.save_optim = cfg.checkpoint.load_optim = True
    cfg.checkpoint.save_rng = cfg.checkpoint.load_rng = True
    cfg.checkpoint.finetune = False
    cfg.logger.log_interval = 1
    return cfg, QwenConfig(**{name: metadata[name] for name in QwenConfig.__dataclass_fields__})


def _load_initial_weights(bridge, models):
    bridge.load_hf_weights(models)
    return models


class MegatronLayer:
    def __init__(self, module, group):
        self.module, self.group = module, group
        self.k = module.router.topk
        self.offset = dist.get_rank(group) * module.num_local_experts

    @property
    def router_weight(self):
        return self.module.router.weight

    def expert_weights(self):
        experts = self.module.experts
        result = []
        for number in range(self.module.num_local_experts):
            gate, up = experts.get_parameter(f"linear_fc1.weight{number}").chunk(2, dim=0)
            down = experts.get_parameter(f"linear_fc2.weight{number}")
            result.append(ExpertWeights(gate, up, down))
        return result

    def execution_route(self, hidden, support):
        """Replay native TE experts and dispatcher, preserving the production path."""
        module = self.module
        shaped = hidden.reshape(-1, 1, hidden.shape[-1])
        scores = F.linear(hidden.float(), self.router_weight.float())
        probabilities = torch.zeros_like(scores).scatter_(-1, support.long(), selected_gates(scores, support))
        mapping = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, support.long(), True)
        dispatched, probabilities = module.token_dispatcher.dispatch_preprocess(shaped, mapping, probabilities)
        dispatched, probabilities = module.dispatch(dispatched, probabilities)
        output, _ = module.routed_experts_compute(dispatched, probabilities, shaped)
        return module.combine(output, None).reshape_as(hidden)

    def reference_routes(self, hidden, old, new, dtype=torch.float32):
        with ieee_fp32(hidden.device.type):
            scores = F.linear(hidden.to(dtype), self.router_weight.to(dtype))
            return expert_parallel_routes(hidden, scores, old, new, self.expert_weights(), self.offset, self.group, dtype)


class MegatronEvaluation:
    """Evaluation hooks are scoped to one forward and removed before the next update."""
    def __init__(self, raw_model, group):
        self.model = raw_model
        self.layers = {number: MegatronLayer(layer.mlp, group)
                       for number, layer in enumerate(raw_model.decoder.layers, start=1)}

    def eval(self):
        self.model.eval()

    @property
    def training(self):
        return self.model.training

    def train(self, mode=True):
        self.model.train(mode)

    def __call__(self, tokens, labels, valid, supports=None, observer=None, intervention=None):
        forced, chosen, scores_by_layer = supports or {}, {}, {}
        with ExitStack() as hooks:
            for number, layer in self.layers.items():
                def router_hook(router, inputs, result, number=number):
                    scores = router.gating(inputs[0]).reshape(-1, router.config.num_moe_experts)
                    scores_by_layer[number] = scores
                    probabilities, mapping = result
                    if number in forced:
                        ids = forced[number].to(tokens.device).long()
                        probabilities = torch.zeros_like(scores).scatter_(-1, ids, selected_gates(scores, ids))
                        mapping = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, True)
                    chosen[number] = mapping.nonzero(as_tuple=True)[1].reshape(-1, router.topk)
                    return probabilities, mapping

                def output_hook(module, inputs, result, number=number, layer=layer):
                    output, bias = result
                    hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
                    flat = output.reshape_as(hidden)
                    if observer is not None:
                        observer(number, layer, hidden, scores_by_layer[number], chosen[number], flat)
                    if intervention is not None:
                        flat = intervention(number, layer, hidden, chosen[number], flat)
                    return flat.reshape_as(output), bias

                handle = layer.module.router.register_forward_hook(router_hook)
                hooks.callback(handle.remove)
                handle = layer.module.register_forward_hook(output_hook)
                hooks.callback(handle.remove)
            positions = torch.arange(tokens.shape[-1], device=tokens.device)[None].expand_as(tokens)
            losses = self.model(input_ids=tokens, position_ids=positions, attention_mask=None, labels=labels)
        return NetworkOutput(losses, losses.new_zeros(()), {number: ids.cpu().to(torch.int32) for number, ids in chosen.items()})

    def causal_forward(self, tokens, labels, valid, supports=None, diagnostic=None, diagnostic_layers=None, return_logits=True):
        """Observe the real gating call; perform no expert replay inside this forward.

        TP=PP=1 in this experiment, so the output head exposes the full vocabulary.
        When requested, all conditions copy logits with the same output hook.
        """
        forced, chosen, scores, residuals, captured_logits = supports or {}, {}, {}, {}, []
        with ExitStack() as hooks:
            for number, layer in self.layers.items():
                router = layer.module.router
                gating = router.gating

                def observed_gating(hidden, number=number, gating=gating):
                    result = gating(hidden)
                    scores[number] = result.reshape(-1, result.shape[-1])
                    return result

                router.gating = observed_gating
                hooks.callback(setattr, router, "gating", gating)

                def choose(router, inputs, result, number=number):
                    probabilities, mapping = result
                    if number in forced:
                        ids = forced[number].to(tokens.device).long()
                        # Match Core's descending-score selected softmax order. The
                        # captured routing map stores IDs in expert order, not score order.
                        order = scores[number].gather(-1, ids).argsort(-1, descending=True)
                        ids = ids.gather(-1, order)
                        probabilities = torch.zeros_like(scores[number]).scatter_(-1, ids, selected_gates(scores[number], ids))
                        mapping = torch.zeros_like(mapping).scatter_(-1, ids, True)
                    chosen[number] = mapping.nonzero(as_tuple=True)[1].reshape(-1, router.topk)
                    return probabilities, mapping

                hooks.callback(router.register_forward_hook(choose).remove)
                if diagnostic is not None and (diagnostic_layers is None or number in diagnostic_layers):
                    decoder = self.model.decoder.layers[number - 1]

                    def before_norm(module, inputs, number=number):
                        residuals[number] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu().clone()

                    def after_moe(module, inputs, result, number=number, layer=layer):
                        hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
                        diagnostic(number, layer, residuals[number], hidden, scores[number], chosen[number], result[0].reshape_as(hidden))

                    hooks.callback(decoder.pre_mlp_layernorm.register_forward_pre_hook(before_norm).remove)
                    hooks.callback(layer.module.register_forward_hook(after_moe).remove)

            def head_output(module, inputs, result):
                captured_logits.append(result[0].detach().transpose(0, 1).contiguous().cpu())

            if return_logits:
                hooks.callback(self.model.output_layer.register_forward_hook(head_output).remove)
            positions = torch.arange(tokens.shape[-1], device=tokens.device)[None].expand_as(tokens)
            losses = self.model(input_ids=tokens, position_ids=positions, attention_mask=None, labels=labels)
        return NetworkOutput(losses.detach().cpu(), losses.new_zeros(()).cpu(),
                             {number: ids.cpu().to(torch.int32) for number, ids in chosen.items()},
                             captured_logits[0] if return_logits else None)

    def causal_suffix(self, layer_number, hidden, tokens, labels, valid):
        """Core 0.15 eval path iterates decoder.layers; retain original layer IDs.

        decoder_input bypasses embedding while preserving positions/RoPE. No
        training parameter is changed. The original layer list is restored on return.
        """
        decoder = self.model.decoder
        layers = decoder.layers
        with ExitStack() as context:
            context.callback(setattr, decoder, "layers", layers)
            decoder.layers = torch.nn.ModuleList(list(layers)[layer_number:])
            positions = torch.arange(tokens.shape[-1], device=tokens.device)[None].expand_as(tokens)
            losses = self.model(input_ids=tokens, position_ids=positions, attention_mask=None, labels=labels,
                                decoder_input=hidden.reshape(-1, 1, hidden.shape[-1]))
        return NetworkOutput(losses.detach().cpu(), losses.new_zeros(()).cpu(), {})


def run_megatron(config, segment_name, output):
    # Importing the project or using --print-config on Mac never imports CUDA/Bridge.
    from moe_study.adapters.cluster_train import run
    run(config, segment_name, output)
