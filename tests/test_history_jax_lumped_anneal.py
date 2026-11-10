"""Per-leaf entropy anneal schedules of the 4B lumped-baseline rerun."""

import pytest

from experiments.belief_quotient_history_jax_2026_11.reinforce import entropy_coefficient
from experiments.belief_quotient_history_jax_lumped_anneal4b_2026_11 import shared as belief
from experiments.reward_factors_history_jax_lumped_anneal4b_2026_11 import shared as reward

ROLLOUT = 19_456 * 1_024


@pytest.mark.parametrize("pkg", [belief, reward])
def test_windows_are_whole_rollouts_four_long_and_inside_budget(pkg):
    assert pkg.TOTAL_ENV_STEPS == 4_000_000_000
    for leaf, (start, end) in pkg.ANNEAL_WINDOWS.items():
        assert start % ROLLOUT == 0 and end % ROLLOUT == 0, leaf
        assert end - start == 4 * ROLLOUT, leaf
        assert 0 < start < end < pkg.TOTAL_ENV_STEPS, leaf


@pytest.mark.parametrize("pkg", [belief, reward])
def test_schedule_holds_then_anneals_to_zero(pkg):
    for leaf, (start, end) in pkg.ANNEAL_WINDOWS.items():
        sched = pkg.entropy_schedule(leaf)
        assert entropy_coefficient(sched, 0) == pytest.approx(0.015)
        assert entropy_coefficient(sched, start) == pytest.approx(0.015)
        assert entropy_coefficient(sched, (start + end) // 2) == pytest.approx(0.0075)
        assert entropy_coefficient(sched, end) == 0.0
        assert entropy_coefficient(sched, pkg.TOTAL_ENV_STEPS) == 0.0


def test_leaf_sets_match_parent_studies():
    assert set(belief.ANNEAL_WINDOWS) == {f"variant_{v}_{a}" for v in (1, 2, 3) for a in belief.base.ARCHITECTURES}
    assert set(reward.ANNEAL_WINDOWS) == {f"{c}_{a}" for c in reward.base.CONDITIONS for a in reward.base.ARCHITECTURES}
