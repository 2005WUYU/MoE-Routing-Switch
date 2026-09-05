import itertools

import torch

from moe_study.reference.expert_fp32 import ExpertWeights
from moe_study.routing import route_union, selected_gates, sequence_balance_loss


def dense_oracle(scores, expert_values, support):
    rows = []
    for token, ids in enumerate(support.tolist()):
        denominator = sum(scores[token, expert].exp() for expert in ids)
        terms = [scores[token, expert].exp() / denominator * expert_values[token, expert] for expert in ids]
        rows.append(sum(terms))
    return torch.stack(rows)


def test_union_matches_independent_all_expert_sum_and_evaluates_once():
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(3, 5, dtype=torch.float64, generator=generator)
    scores = torch.randn(3, 6, dtype=torch.float64, generator=generator)
    matrices = torch.randn(6, 5, 5, dtype=torch.float64, generator=generator)
    old = torch.tensor([[0, 1, 2], [0, 3, 4], [1, 2, 5]])
    new = torch.tensor([[3, 4, 5], [0, 3, 5], [5, 2, 1]])
    seen = []

    def expert(number, inputs):
        for row in inputs:
            token = (hidden == row).all(-1).nonzero().item()
            seen.append((token, number))
        return inputs @ matrices[number]

    middle, actual = route_union(hidden, scores, old, new, expert)
    all_outputs = torch.stack([hidden @ matrix for matrix in matrices], dim=1)
    torch.testing.assert_close(middle, dense_oracle(scores, all_outputs, old), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual, dense_oracle(scores, all_outputs, new), rtol=1e-13, atol=1e-13)
    assert len(seen) == len(set(seen)) == sum(len(set(a + b)) for a, b in zip(old.tolist(), new.tolist()))
    assert torch.equal(middle[2], actual[2])


def test_constant_experts_analytic_selected_normalization():
    hidden = torch.zeros(1, 2, dtype=torch.float64)
    scores = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    constants = torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]], dtype=torch.float64)
    middle, actual = route_union(hidden, scores, torch.tensor([[0, 1]]), torch.tensor([[1, 2]]),
                                 lambda e, h: constants[e].expand_as(h))
    torch.testing.assert_close(middle, torch.tensor([[3., 5.]], dtype=torch.float64))
    expected = (constants[1] + torch.exp(torch.tensor(2.)) * constants[2]) / (1 + torch.exp(torch.tensor(2.)))
    torch.testing.assert_close(actual[0], expected, rtol=1e-7, atol=1e-7)


def test_identical_experts_can_switch_without_a_jump():
    hidden = torch.randn(4, 7)
    scores = torch.randn(4, 5)
    a = torch.tensor([[0, 1]]).expand(4, -1)
    b = torch.tensor([[3, 4]]).expand(4, -1)
    old, new = route_union(hidden, scores, a, b, lambda e, h: torch.ones_like(h))
    torch.testing.assert_close(old, new, rtol=1e-6, atol=1e-7)


def test_new_scores_recompute_gates_on_old_set():
    ids = torch.tensor([[0, 1]])
    old, new = torch.tensor([[0., 0., 0.]]), torch.tensor([[2., 0., 3.]])
    assert not torch.equal(selected_gates(old, ids), selected_gates(new, ids))
    torch.testing.assert_close(selected_gates(new, ids).sum(-1), torch.ones(1))


def test_expert_casts_bf16_weights_before_gemm():
    generator = torch.Generator().manual_seed(4)
    weights = [torch.randn(shape, generator=generator).bfloat16() for shape in ((6, 4), (6, 4), (4, 6))]
    hidden = torch.randn(3, 4, generator=generator).bfloat16()
    expert = ExpertWeights(*weights)
    actual = expert.evaluate(hidden)
    torch.testing.assert_close(actual.double(), expert.evaluate(hidden, torch.float64), rtol=2e-6, atol=1e-6)
    assert actual.dtype == torch.float32


def test_balance_is_mean_of_sequences_with_padding_and_layers_handled_by_caller():
    scores = torch.tensor([[[2., 0.], [0., 2.]], [[1., -1.], [100., -100.]]], requires_grad=True)
    ids = scores.topk(1, dim=-1).indices
    valid = torch.tensor([[True, True], [True, False]])
    actual = sequence_balance_loss(scores, ids, valid, .001)
    expected = (sequence_balance_loss(scores[:1], ids[:1], valid[:1], .001)
                + sequence_balance_loss(scores[1:], ids[1:], valid[1:], .001)) / 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad[1, 1].abs().sum() == 0
    assert scores.grad[1, 0].abs().sum() > 0
