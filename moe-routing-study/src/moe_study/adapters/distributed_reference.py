"""EP-local FP32/FP64 replay and FP32 master reconstruction for Core 0.15.

    Rank order within an EP group determines contiguous global expert IDs.
    Reconstruction reduces optimizer shards within the parameter's own DP group.
    It never derives the update from rounded BF16 execution parameters.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from moe_study.reference.expert_fp32 import ExpertWeights, ieee_fp32
from moe_study.reference.network_fp32 import Expert, QwenReference
from moe_study.routing import route_union


def gather_tensor(value, group):
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, value.contiguous(), group=group)
    return torch.cat(gathered, dim=0)


def expert_parallel_routes(hidden, scores, old, new, expert_weights, offset, group, dtype):
    """Gather a sequence block inside EP; each owner computes its expert union once.

    The union masks are global expert IDs. Non-owned branches contribute exact
    zeros to the EP sum. No expert weights cross the EP group.
    """
    count = hidden.shape[0]
    h = gather_tensor(hidden.to(dtype), group)
    z = gather_tensor(scores.to(dtype), group)
    a0, a1 = gather_tensor(old.long(), group), gather_tensor(new.long(), group)

    def owned_expert(number, inputs):
        if offset <= number < offset + len(expert_weights):
            return expert_weights[number - offset].evaluate(inputs, dtype)
        return torch.zeros_like(inputs)

    middle, actual = route_union(h, z, a0, a1, owned_expert)
    dist.all_reduce(middle, group=group)
    dist.all_reduce(actual, group=group)
    start = dist.get_rank(group) * count
    return middle[start:start + count], actual[start:start + count]


class ShardedSparseReference(nn.Module):
    def __init__(self, config, group):
        super().__init__()
        self.group = group
        self.k = config.num_experts_per_tok
        local_count = config.num_experts // dist.get_world_size(group)
        self.offset = dist.get_rank(group) * local_count
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(local_count)])

    @property
    def router_weight(self):
        return self.gate.weight

    def reference_routes(self, hidden, old, new, dtype=torch.float32):
        with ieee_fp32(hidden.device.type):
            scores = F.linear(hidden.to(dtype), self.gate.weight.to(dtype))
            return expert_parallel_routes(hidden, scores, old, new, [e.weights() for e in self.experts], self.offset, self.group, dtype)

    def execution_route(self, hidden, support):
        return self.reference_routes(hidden, support, support)[0]


def create_reference(config, group):
    """Allocate only the local expert shard plus replicated dense FP32 layers."""
    with torch.device("meta"):
        model = QwenReference(config)
        for layer in model.layers:
            layer.mlp = ShardedSparseReference(config, group)
    return model.to_empty(device="cuda").float().eval()


@dataclass
class MasterSlice:
    start: int
    end: int
    old: torch.Tensor
    optimizer: object
    parameter: nn.Parameter


class MasterUpdate:
    def __init__(self, model, optimizer):
        self.model = model
        self.slices = {}
        # With EP>1, Core returns ChainedOptimizer for dense/expert parameter groups.
        for child in optimizer.chained_optimizers:
            for parameter in child.model_param_gbuf_map:
                span = child._get_model_param_range_map(parameter)["param"]
                master = child._get_main_param_and_optimizer_states(parameter)["param"]
                self.slices[parameter] = MasterSlice(span.start, span.end, master.detach().cpu().clone(), child, parameter)

    def at(self, name, alpha, dense_group, expert_data_group):
        parameter = self.model.get_parameter(name)
        full = torch.zeros(parameter.shape, device=parameter.device, dtype=torch.float32)
        if parameter in self.slices:
            piece = self.slices[parameter]
            new = piece.optimizer._get_main_param_and_optimizer_states(parameter)["param"]
            old = piece.old.to(parameter.device)
            # Endpoints are the stored masters themselves, without subtract/add rounding.
            value = old if alpha == 0 else new if alpha == 1 else old + alpha * (new - old)
            full.view(-1)[piece.start:piece.end] = value
        group = expert_data_group if ".mlp.experts." in name else dense_group
        dist.all_reduce(full, group=group)
        return full

    def save_new_endpoint(self, destination, dense_group, expert_data_group):
        """Persist only uniquely owned FP32 shards, not a second Adam checkpoint."""
        destination.mkdir(parents=True, exist_ok=True)
        names = {parameter: name for name, parameter in self.model.named_parameters()}
        parameters = {}
        for parameter, piece in self.slices.items():
            master = piece.optimizer._get_main_param_and_optimizer_states(parameter)["param"]
            parameters[names[parameter]] = {"shape": list(parameter.shape), "execution_dtype": str(parameter.dtype),
                "start": piece.start, "end": piece.end, "master": master.detach().cpu().clone()}
        torch.save({"rank": dist.get_rank(), "dense_group": dist.get_process_group_ranks(dense_group),
                    "expert_data_group": dist.get_process_group_ranks(expert_data_group), "parameters": parameters},
                   destination / f"rank_{dist.get_rank():05d}.pt")


def split_qkv(packed, config):
    """Megatron groups Q heads with their K/V head; retain Q width 4096 for Qwen."""
    q_per_group = config.num_attention_heads // config.num_key_value_heads
    grouped = packed.view(config.num_key_value_heads, q_per_group + 2, config.head_dim, config.hidden_size)
    q = grouped[:, :q_per_group].reshape(-1, config.hidden_size)
    k = grouped[:, q_per_group].reshape(-1, config.hidden_size)
    v = grouped[:, q_per_group + 1].reshape(-1, config.hidden_size)
    return q, k, v


@torch.no_grad()
def load_reference_master(reference, update, alpha, dense_group, expert_data_group):
    def value(name):
        return update.at(name, alpha, dense_group, expert_data_group)

    reference.embed_tokens.weight.copy_(value("embedding.word_embeddings.weight"))
    reference.lm_head.weight.copy_(value("output_layer.weight"))
    reference.norm.weight.copy_(value("decoder.final_layernorm.weight"))
    for number, layer in enumerate(reference.layers):
        prefix = f"decoder.layers.{number}."
        layer.input_layernorm.weight.copy_(value(prefix + "self_attention.linear_qkv.layer_norm_weight"))
        layer.post_attention_layernorm.weight.copy_(value(prefix + "pre_mlp_layernorm.weight"))
        layer.mlp.gate.weight.copy_(value(prefix + "mlp.router.weight"))
        layer.self_attn.q_norm.weight.copy_(value(prefix + "self_attention.q_layernorm.weight"))
        layer.self_attn.k_norm.weight.copy_(value(prefix + "self_attention.k_layernorm.weight"))
        layer.self_attn.o_proj.weight.copy_(value(prefix + "self_attention.linear_proj.weight"))
        q, k, v = split_qkv(value(prefix + "self_attention.linear_qkv.weight"), reference.config)
        layer.self_attn.q_proj.weight.copy_(q)
        layer.self_attn.k_proj.weight.copy_(k)
        layer.self_attn.v_proj.weight.copy_(v)
        for expert_id, expert in enumerate(layer.mlp.experts):
            gate, up = value(prefix + f"mlp.experts.linear_fc1.weight{expert_id}").chunk(2, dim=0)
            expert.gate_proj.weight.copy_(gate)
            expert.up_proj.weight.copy_(up)
            expert.down_proj.weight.copy_(value(prefix + f"mlp.experts.linear_fc2.weight{expert_id}"))
