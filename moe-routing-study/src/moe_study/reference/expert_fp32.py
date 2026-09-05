"""Unfused SwiGLU references on the *executed* weight values, cast before GEMM."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@contextmanager
def ieee_fp32(device_type: str):
    """Use IEEE FP32 GEMM and disable autocast; restore the surrounding settings."""
    old_global = torch.backends.fp32_precision
    old_matmul = torch.backends.cuda.matmul.fp32_precision
    old_cudnn = torch.backends.cudnn.fp32_precision
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.fp32_precision = old_global
        torch.backends.cuda.matmul.fp32_precision = old_matmul
        torch.backends.cudnn.fp32_precision = old_cudnn


@dataclass
class ExpertWeights:
    gate: Tensor
    up: Tensor
    down: Tensor

    def evaluate(self, hidden: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
        hidden = hidden.to(dtype)
        gate = F.linear(hidden, self.gate.to(dtype))
        up = F.linear(hidden, self.up.to(dtype))
        return F.linear(F.silu(gate) * up, self.down.to(dtype))


def expert_function(weights: list[ExpertWeights], dtype: torch.dtype = torch.float32):
    def evaluate(expert_id: int, hidden: Tensor) -> Tensor:
        return weights[expert_id].evaluate(hidden, dtype)

    return evaluate
