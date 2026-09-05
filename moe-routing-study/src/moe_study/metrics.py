"""Per-token mathematics. No file I/O, scheduling, or experiment decisions."""

import math

import torch
from torch import Tensor


def update_terms(y0: Tensor, y10: Tensor, y1: Tensor, h1: Tensor) -> dict[str, Tensor]:
    """Cast each output to FP64 *before* subtraction (equation 1)."""
    remainder = y10.double() - y0.double()
    jump = y1.double() - y10.double()
    total = y1.double() - y0.double()
    return {
        "S": remainder.square().mean(dim=-1),
        "J": jump.square().mean(dim=-1),
        "C": (remainder * jump).mean(dim=-1),
        "T": total.square().mean(dim=-1),
        "H": h1.double().square().mean(dim=-1),
    }


def support_changes(old: Tensor, new: Tensor, old_scores: Tensor) -> dict[str, Tensor]:
    """Full K-set replacements and the incoming experts' actual old ranks (1-based)."""
    incoming = ~(new[..., :, None] == old[..., None, :]).any(dim=-1)
    replacements = incoming.sum(dim=-1)
    old_ranks = old_scores.argsort(dim=-1, descending=True, stable=True).argsort(dim=-1) + 1
    incoming_ranks = old_ranks.gather(-1, new.long())
    return {
        "switched": replacements > 0,
        "replacements": replacements,
        "incoming_rank": torch.where(incoming, incoming_ranks, 0),
    }


def boundary_terms(
    h0: Tensor, h1: Tensor, w0: Tensor, w1: Tensor, scores0: Tensor, k: int
) -> dict[str, Tensor]:
    """Fixed old K/(K+1) pair, with all three terms of equation (3).

    Full scores determine IDs; dot products use FP64 on supplied weight/input
    values. Router GEMM rounding can be reported separately from this identity.
    """
    order = scores0.argsort(dim=-1, descending=True, stable=True)
    i, j = order[..., k - 1], order[..., k]
    a0 = w0[i].double() - w0[j].double()
    a1 = w1[i].double() - w1[j].double()
    da, dh = a1 - a0, h1.double() - h0.double()
    router = (da * h0.double()).sum(dim=-1)
    upstream = (a0 * dh).sum(dim=-1)
    cross = (da * dh).sum(dim=-1)
    return {
        "margin0": (a0 * h0.double()).sum(dim=-1),
        "margin1": (a1 * h1.double()).sum(dim=-1),
        "margin_router": router,
        "margin_upstream": upstream,
        "margin_cross": cross,
    }


def loss_terms(loss0: Tensor, loss10: Tensor, loss1: Tensor) -> dict[str, Tensor]:
    return {
        "L0": loss0.double(), "L10": loss10.double(), "L1": loss1.double(),
        "U": loss10.double() - loss0.double(),
        "V": loss1.double() - loss10.double(),
        "loss_change": loss1.double() - loss0.double(),
    }


def empirical_energy_envelope(energy: float, error_rms: float) -> tuple[float, float]:
    """Equations (6–7); empirical error RMS is not a universal error bound."""
    rms = math.sqrt(energy)
    return max(0.0, rms - error_rms) ** 2, (rms + error_rms) ** 2


def same_norm_direction(replacement: Tensor, generator: torch.Generator) -> Tensor:
    """Direction control for the actual replacement y_old_support - y_actual."""
    direction = torch.randn(
        replacement.shape, device=replacement.device, dtype=torch.float32, generator=generator
    )
    norm = replacement.float().norm(dim=-1, keepdim=True)
    return direction / direction.norm(dim=-1, keepdim=True) * norm
