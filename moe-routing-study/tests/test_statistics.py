import numpy as np

from moe_study.statistics import GroupTable, adjacent_slopes, bootstrap_indices, derived, energy_tail, paired_bootstrap


def test_document_pooling_weights_tokens_and_excludes_padding():
    groups = np.array(["a", "a", "a", "b", "b"])
    mask = np.array([1, 1, 1, 1, 0], dtype=bool)
    table = GroupTable.from_tokens(groups, mask, {"V": np.array([1., 1., 1., 9., 1000.])})
    assert table.counts.tolist() == [3, 1]
    assert derived(table.total())["V"] == 3
    indices = np.array([[0, 1], [1, 1], [0, 0]])
    result = paired_bootstrap(table, indices)
    assert result["V"]["estimate"] == 3
    assert result["V"]["interval"][1] < 10


def test_same_bootstrap_indices_preserve_paired_alpha_relation():
    table = GroupTable(np.array(["a", "b", "c"]), np.array([2., 4., 3.]), {"V": np.array([1., 2., 9.])})
    double = GroupTable(table.group_ids, table.counts, {"V": 2 * table.sums["V"]})
    indices = bootstrap_indices(3, 100, 13)
    a, b = paired_bootstrap(table, indices), paired_bootstrap(double, indices)
    np.testing.assert_allclose(b["V"]["interval"], 2 * np.array(a["V"]["interval"]))


def test_undefined_ratios_are_explicit_and_zero_events_stay_zero():
    values = derived({"N": 10., "J": 0., "J_switched": 0., "switched": 0., "H": 0., "T": 0.})
    assert values["J"] == 0
    assert values["conditional_J"] is values["r_h"] is values["J_over_T"] is None
    assert energy_tail(np.zeros(10), np.zeros(10, dtype=bool))["conditional_quantiles"] is None


def test_slopes_skip_zero_events_and_overlapping_numerical_ranges():
    rows = adjacent_slopes([0., .25, .5, 1.], [0., .25, .5, 1.], [(0., 0.), (.2, .3), (.4, .6), (.5, 1.5)])
    assert rows[0]["energy_slope"] is None
    assert rows[1]["energy_slope"] == 1
    assert rows[1]["rms_slope"] == .5
    assert rows[2]["energy_slope"] is None
