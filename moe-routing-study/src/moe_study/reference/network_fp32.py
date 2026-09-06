"""Readable full-network Qwen3-MoE reference, also used for small CPU development.

Dimensions come from config; Q width = num_attention_heads * head_dim, which
need not equal hidden_size. No KV cache, shared experts, dropout, or token drop.
The cluster adapter supplies sharded experts to the same layer-level interface.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from moe_study.reference.expert_fp32 import ExpertWeights, ieee_fp32
from moe_study.routing import route, route_union, select_support, sequence_balance_loss


@dataclass
class QwenConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, hidden: Tensor) -> Tensor:
        value = hidden.float()
        return (value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.epsilon)).to(hidden.dtype) * self.weight


def rotary(hidden: Tensor, theta: float) -> Tensor:
    length, width = hidden.shape[-2:]
    frequency = theta ** (-torch.arange(0, width, 2, device=hidden.device, dtype=torch.float32) / width)
    phase = torch.arange(length, device=hidden.device, dtype=torch.float32)[:, None] * frequency[None, :]
    phase = torch.cat((phase, phase), dim=-1)
    first, second = hidden.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return hidden * phase.cos() + rotated * phase.sin()


class Attention(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        config = self.config
        q = self.q_norm(self.q_proj(hidden).view(batch, length, config.num_attention_heads, config.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden).view(batch, length, config.num_key_value_heads, config.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
        q, k = rotary(q, config.rope_theta), rotary(k, config.rope_theta)
        repeats = config.num_attention_heads // config.num_key_value_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        # Math SDPA keeps FP32 arithmetic and avoids fused low-precision attention.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


class Expert(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))

    def weights(self) -> ExpertWeights:
        return ExpertWeights(self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)


class SparseLayer(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])

    @property
    def router_weight(self):
        return self.gate.weight

    def execution_route(self, hidden, support):
        scores = F.linear(hidden.float(), self.gate.weight.float())
        return route(hidden, scores, support, lambda e, h: self.experts[e](h))

    def reference_routes(self, hidden, old, new, dtype=torch.float32):
        with ieee_fp32(hidden.device.type):
            hidden = hidden.to(dtype)
            scores = F.linear(hidden, self.gate.weight.to(dtype))
            return route_union(hidden, scores, old, new, lambda e, h: self.experts[e].weights().evaluate(h, dtype))


class DecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SparseLayer(config)


@dataclass
class NetworkOutput:
    losses: Tensor
    balance_loss: Tensor
    supports: dict[int, Tensor]
    logits: Tensor | None = None


class QwenReference(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens, labels, valid, supports=None, observer=None, intervention=None, balance_coefficient=0.0,
                diagnostic=None, return_logits=False):
        forced = supports or {}
        hidden = self.embed_tokens(tokens)
        selected, balance = {}, []
        for number, layer in enumerate(self.layers, start=1):
            hidden = hidden + layer.self_attn(layer.input_layernorm(hidden))
            router_input = layer.post_attention_layernorm(hidden).reshape(-1, self.config.hidden_size)
            scores = F.linear(router_input.float(), layer.mlp.router_weight.float())
            support = forced[number].to(hidden.device) if number in forced else select_support(scores, layer.mlp.k)
            output = layer.mlp.execution_route(router_input, support)
            selected[number] = support.detach().cpu().to(torch.int32)
            if diagnostic is not None:
                diagnostic(number, layer.mlp, hidden.reshape(-1, self.config.hidden_size),
                           router_input, scores, support, output)
            if observer is not None:
                observer(number, layer.mlp, router_input, scores, support, output)
            if intervention is not None:
                output = intervention(number, layer.mlp, router_input, support, output)
            hidden = hidden + output.reshape_as(hidden)
            if balance_coefficient:
                balance.append(sequence_balance_loss(scores.reshape(*valid.shape, -1),
                                                     support.reshape(*valid.shape, -1), valid, balance_coefficient))
        logits = self.lm_head(self.norm(hidden))
        losses = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1), reduction="none").reshape_as(labels)
        auxiliary = torch.stack(balance).mean() if balance else hidden.new_zeros(())
        return NetworkOutput(losses, auxiliary, selected, logits.detach().cpu() if return_logits else None)

    def causal_forward(self, tokens, labels, valid, supports=None, diagnostic=None):
        return self(tokens, labels, valid, supports=supports, diagnostic=diagnostic, return_logits=True)
