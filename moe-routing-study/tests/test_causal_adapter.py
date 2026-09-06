from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from moe_study.adapters.distributed_reference import ShardedSparseReference
from moe_study.adapters.megatron_qwen import MegatronEvaluation
from moe_study.causal_measure import CausalMeasurement, RNGSnapshot
from moe_study.measure import batch
from moe_study.reference.network_fp32 import QwenConfig, QwenReference
from moe_study.train import development_samples


class NativeRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 8))
        self.config = SimpleNamespace(num_moe_experts=4)
        self.topk, self.gating_calls = 2, 0

    def gating(self, hidden):
        self.gating_calls += 1
        return F.linear(hidden, self.weight)

    def forward(self, hidden):
        scores = self.gating(hidden).reshape(-1, 4)
        selected, ids = scores.topk(2, dim=-1)
        return (torch.zeros_like(scores).scatter_(-1, ids, selected.softmax(-1)),
                torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, True))


class NativeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = NativeRouter()
        self.num_local_experts = 4

    def forward(self, hidden):
        probabilities, _ = self.router(hidden)
        scale = (probabilities * torch.arange(1, 5)).sum(-1).reshape(*hidden.shape[:-1], 1)
        return hidden * scale, None


class NativeHead(nn.Linear):
    def forward(self, hidden):
        return super().forward(hidden), None


class NativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.decoder.layers:
            layer.pre_mlp_layernorm = nn.LayerNorm(8)
            layer.mlp = NativeMoE()
        self.output_layer = NativeHead(8, 16, bias=False)

    def forward(self, input_ids, position_ids, attention_mask, labels):
        hidden = self.embedding(input_ids).transpose(0, 1)
        for layer in self.decoder.layers:
            hidden = hidden + layer.mlp(layer.pre_mlp_layernorm(hidden))[0]
        logits = self.output_layer(hidden)[0].transpose(0, 1)
        return F.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1), reduction="none").reshape_as(labels)


def test_native_hooks_capture_actual_gating_once_and_restore_the_model(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    raw = NativeModel()
    model = MegatronEvaluation(raw, None)
    sample = development_samples(1, 4, 16, 9, "native")[0]
    records = []
    with torch.no_grad():
        natural = model.causal_forward(**batch(sample, "cpu"), diagnostic=lambda *values: records.append(values))
        fixed = model.causal_forward(**batch(sample, "cpu"), supports=natural.supports,
                                     diagnostic=lambda *values: records.append(values))
    assert torch.equal(natural.losses, fixed.losses)
    assert torch.equal(natural.logits, fixed.logits)
    assert len(records) == 4
    for layer in raw.decoder.layers:
        assert layer.mlp.router.gating_calls == 2
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in raw.modules())
    # The diagnostic residual is before layer normalization, not its normalized output.
    assert not torch.equal(records[0][2], records[0][3])


def _ep_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.manual_seed(78)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    config = QwenConfig(8, 2, 2, 1, 4, 4, 2, 8, 16)
    full = QwenReference(config)
    sharded = QwenReference(config)
    sharded.load_state_dict(full.state_dict())
    for a, b in zip(full.layers, sharded.layers):
        sparse = ShardedSparseReference(config, dist.group.WORLD)
        sparse.gate.load_state_dict(a.mlp.gate.state_dict())
        for local, expert in enumerate(sparse.experts):
            expert.load_state_dict(a.mlp.experts[rank * 2 + local].state_dict())
        b.mlp = sparse
    sample = development_samples(2, 4, 16, 62, "ep")[rank]
    pair = CausalMeasurement([sample], [0, 1], output, "ep", RNGSnapshot.capture(), announce=False)
    pair.capture_old(sharded)
    supports = pair.old[rank].supports
    with torch.no_grad():
        for model in (full, sharded):
            model.layers[0].mlp.gate.weight.mul_(-1)
        expected = (full(**batch(sample, "cpu")).losses.double()
                    - full(**batch(sample, "cpu"), supports=supports).losses.double()).reshape(-1)
    pair.measure_new(sharded)
    np.testing.assert_allclose(pair.network.arrays()[2]["V_1"], expected.numpy(), atol=2e-6, rtol=0)
    dist.destroy_process_group()


def test_two_rank_expert_parallel_counterfactual_matches_full_model(tmp_path):
    mp.spawn(_ep_worker, args=(f"file://{tmp_path}/rendezvous", tmp_path), nprocs=2, join=True)
    for rank in range(2):
        with np.load(tmp_path / f"ep/diagnostics/sequence_{rank:06d}.npz") as diagnostic:
            assert diagnostic["J_local_reference"].max() > 0
