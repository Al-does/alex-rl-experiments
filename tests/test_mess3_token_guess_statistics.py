"""Tests for the token-guess seed-level statistics."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.mess3_token_guess_cycle_2.statistics import (
    compare,
    holm_adjust,
    seeds_for_power,
    summarise,
    t_test_power,
)

# Kelly cycle 3 belief-probe R², seeds 42/43/44, as recorded in each arm's
# results/<run_id>/condition_summary.json.
CYCLE_3_PPO = {42: 0.860618, 43: 0.932631, 44: 0.829447}
CYCLE_3_IQN = {42: 0.974336, 43: 0.975997, 44: 0.956050}


def test_summarise_reports_a_wider_interval_than_the_published_plus_minus():
    values = list(CYCLE_3_PPO.values())
    estimate = summarise(values)

    population_sd = float(np.std(values, ddof=0))
    assert estimate.sample_sd > population_sd

    # The published "±" was a population standard deviation. A 95% interval on
    # the mean of three seeds is several times wider.
    assert estimate.half_width > 2.0 * population_sd
    assert estimate.ci_low < estimate.mean < estimate.ci_high
    assert estimate.n == 3


def test_summarise_rejects_a_single_seed():
    with pytest.raises(ValueError):
        summarise([0.97])


def test_seeds_for_power_scales_with_the_squared_noise_to_signal_ratio():
    assert seeds_for_power(0.02, 0.01) < seeds_for_power(0.01, 0.01)
    assert seeds_for_power(0.01, 0.02) > seeds_for_power(0.01, 0.01)
    # An unpaired design needs more seeds than a paired one.
    assert seeds_for_power(0.01, 0.01, paired=False) > seeds_for_power(
        0.01, 0.01, paired=True
    )
    assert seeds_for_power(0.0, 0.01) >= 10_000


def test_power_rises_with_seeds_and_the_planner_agrees_with_it():
    difference, sample_sd = 0.010, 0.008
    powers = [t_test_power(n, difference, sample_sd) for n in (3, 6, 12, 24)]
    assert powers == sorted(powers)
    assert powers[0] < 0.8 < powers[-1]

    needed = seeds_for_power(difference, sample_sd)
    assert t_test_power(needed, difference, sample_sd) >= 0.8
    assert t_test_power(needed - 1, difference, sample_sd) < 0.8


def test_the_headline_ppo_versus_iqn_gap_is_unresolved_at_three_seeds():
    # Kelly cycle 3 reported PPO 0.8742 ± 0.0432 against IQN 0.9688 ± 0.0090,
    # which reads as decisive. Paired across the seeds both arms actually ran,
    # the interval still contains zero.
    comparison = compare(CYCLE_3_IQN, CYCLE_3_PPO)

    assert comparison.n == 3
    assert comparison.difference > 0.09
    assert not comparison.resolved
    assert comparison.ci_low < 0.0 < comparison.ci_high
    assert comparison.p_value > 0.05
    assert comparison.seeds_for_power > 3


def test_a_difference_smaller_than_its_noise_is_not_resolved_by_three_seeds():
    # Two conditions separated by less than their seed-to-seed spread.
    left = {42: 0.9720, 43: 0.9660, 44: 0.9770}
    right = {42: 0.9700, 43: 0.9700, 44: 0.9700}
    comparison = compare(left, right)

    assert abs(comparison.difference) < 0.005
    assert not comparison.resolved
    assert comparison.seeds_for_power > 3


def test_compare_requires_shared_seeds():
    with pytest.raises(ValueError):
        compare({42: 0.9}, {43: 0.8})


def test_holm_adjustment_is_monotonic_and_never_shrinks_a_p_value():
    raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
    adjusted = holm_adjust(raw)

    assert set(adjusted) == set(raw)
    for name, value in raw.items():
        assert adjusted[name] >= value
    ordered = sorted(raw, key=lambda name: raw[name])
    values = [adjusted[name] for name in ordered]
    assert values == sorted(values)
    # The smallest of four p-values is multiplied by four.
    assert adjusted["a"] == pytest.approx(0.004)
    assert holm_adjust({}) == {}
