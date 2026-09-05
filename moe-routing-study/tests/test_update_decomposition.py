import torch

from moe_study.metrics import boundary_terms, empirical_energy_envelope, loss_terms, same_norm_direction, support_changes, update_terms


def test_energy_and_loss_identities_include_negative_cross_term():
    y0 = torch.zeros(2, 3)
    middle = torch.ones(2, 3) * 2
    y1 = torch.ones(2, 3)
    terms = update_terms(y0, middle, y1, y1)
    torch.testing.assert_close(terms["T"], terms["S"] + terms["J"] + 2 * terms["C"])
    assert terms["C"].max() < 0
    losses = loss_terms(torch.tensor([2.]), torch.tensor([1.]), torch.tensor([3.]))
    torch.testing.assert_close(losses["loss_change"], losses["U"] + losses["V"])
    assert losses["U"] < 0 < losses["loss_change"]


def test_cast_before_subtraction_preserves_output_difference():
    y0 = torch.tensor([[1.0]], dtype=torch.float32)
    middle = torch.tensor([[1.0 + 2**-23]], dtype=torch.float32)
    y1 = torch.tensor([[1.0 + 2**-22]], dtype=torch.float32)
    assert update_terms(y0, middle, y1, y1)["J"].item() == 2**-46


def test_margin_uses_old_pair_and_keeps_cross_term():
    h0 = torch.tensor([[1., 2.]], dtype=torch.float64)
    h1 = torch.tensor([[2., 1.]], dtype=torch.float64)
    w0 = torch.tensor([[2., 0.], [0., .5], [-1., -1.]], dtype=torch.float64)
    w1 = w0 + torch.tensor([[-1.5, 0.], [.5, 1.], [0., 0.]], dtype=torch.float64)
    terms = boundary_terms(h0, h1, w0, w1, h0 @ w0.T, 1)
    assert terms["margin0"] > 0 > terms["margin1"]
    assert terms["margin_cross"] != 0
    torch.testing.assert_close(terms["margin1"] - terms["margin0"],
                               terms["margin_router"] + terms["margin_upstream"] + terms["margin_cross"])


def test_multiple_incoming_experts_use_all_old_scores():
    old, new = torch.tensor([[0, 1, 2]]), torch.tensor([[0, 4, 5]])
    scores = torch.tensor([[6., 5., 4., 3., 2., 1.]])
    terms = support_changes(old, new, scores)
    assert terms["replacements"].item() == 2
    assert terms["incoming_rank"].tolist() == [[0, 5, 6]]


def test_direction_norm_sign_and_training_rng_isolation():
    replacement = torch.tensor([[3., -4., 0.], [0., 0., 0.]])
    before = torch.get_rng_state()
    direction = same_norm_direction(replacement, torch.Generator().manual_seed(100))
    torch.testing.assert_close(direction.norm(dim=-1), replacement.norm(dim=-1))
    assert torch.equal(before, torch.get_rng_state())
    assert direction[1].count_nonzero() == 0


def test_empirical_envelope_is_not_noise_subtraction():
    assert empirical_energy_envelope(4., 1.) == (1., 9.)
    assert empirical_energy_envelope(.25, 1.) == (0., 2.25)
