"""Complete selected-normalized routes, with one evaluation per token/expert pair.

Tensor convention: hidden [tokens, width], scores [tokens, experts], supports
[tokens, K]. Support order has no mathematical meaning. Reference aggregation
uses ascending expert IDs so a permutation of the same support gives exact zero D.
"""

from collections.abc import Callable

import torch
from torch import Tensor

ExpertFunction = Callable[[int, Tensor], Tensor]


def select_support(scores: Tensor, k: int) -> Tensor:
    """CPU/reference tie convention: lower expert ID first; execution records its own IDs."""
    return scores.argsort(dim=-1, descending=True, stable=True)[..., :k]


def selected_gates(scores: Tensor, support: Tensor) -> Tensor:
    """Recompute softmax on the supplied set using the current scores."""
    return scores.gather(-1, support.long()).softmax(dim=-1)


def route_union(
    hidden: Tensor, scores: Tensor, old: Tensor, new: Tensor, expert: ExpertFunction
) -> tuple[Tensor, Tensor]:
    """Return F(h, θ; old), F(h, θ; new), computing only their expert union.

    No [tokens, all_experts, width] tensor is materialized. Each expert sees the
    same token block for both routes. The caller controls FP32/FP64 arithmetic.
    """
    old = old.long().sort(dim=-1).values
    new = new.long().sort(dim=-1).values
    gates_old = selected_gates(scores, old)
    gates_new = selected_gates(scores, new)
    output_old = torch.zeros_like(hidden)
    output_new = torch.zeros_like(hidden)
    for expert_id in torch.cat((old, new), dim=-1).unique(sorted=True).tolist():
        in_old = old == expert_id
        in_new = new == expert_id
        rows = (in_old.any(dim=-1) | in_new.any(dim=-1)).nonzero(as_tuple=True)[0]
        values = expert(expert_id, hidden[rows])
        weight_old = (gates_old[rows] * in_old[rows]).sum(dim=-1, keepdim=True)
        weight_new = (gates_new[rows] * in_new[rows]).sum(dim=-1, keepdim=True)
        output_old[rows] += values * weight_old
        output_new[rows] += values * weight_new
    return output_old, output_new


def route(hidden: Tensor, scores: Tensor, support: Tensor, expert: ExpertFunction) -> Tensor:
    return route_union(hidden, scores, support, support, expert)[0]


def sequence_balance_loss(
    scores: Tensor, support: Tensor, valid: Tensor, coefficient: float
) -> Tensor:
    """Equation (5) for [batch, sequence, experts], averaged over sequences.

    The caller averages layers. Support counts are constants for differentiation;
    the dense router probabilities retain their task-independent gradient path.
    """
    probability = scores.float().softmax(dim=-1)
    mask = valid.to(probability.dtype)
    count = mask.sum(dim=-1, keepdim=True)
    assignment = torch.zeros_like(probability).scatter_(-1, support.long(), 1)
    frequency = (assignment * mask[..., None]).sum(dim=-2) / (count * support.shape[-1])
    mean_probability = (probability * mask[..., None]).sum(dim=-2) / count
    return coefficient * scores.shape[-1] * (frequency * mean_probability).sum(dim=-1).mean()
