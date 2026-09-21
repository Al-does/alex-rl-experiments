"""Tests for the re-analysis of the committed multi-seed token-guess results."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.mess3_token_guess_cycle_2.audit.experiment import (
    STUDIES,
    analyse_study,
    collect_study,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
REFERENCES = {
    "belief_r2_floor": 0.9668,
    "belief_r2_ceiling": 0.99888,
    "accuracy_floor": 0.6732,
    "accuracy_ceiling": 0.6883,
}


@pytest.mark.parametrize("study", STUDIES)
def test_every_audited_study_committed_three_seeds_per_arm(study):
    per_arm = collect_study(REPOSITORY_ROOT, study)

    assert per_arm, f"{study} committed no condition summaries"
    for arm, by_seed in per_arm.items():
        assert sorted(by_seed) == [42, 43, 44], arm


def test_the_kelly_cycle_3_orderings_do_not_survive_a_holm_correction():
    analysis = analyse_study(
        collect_study(REPOSITORY_ROOT, "mess_3_kelly_cycle_3"),
        REFERENCES,
    )
    comparisons = analysis["comparisons"]

    assert len(comparisons) == 6
    assert not any(values["resolved"] for values in comparisons.values())


def test_only_the_strongest_kelly_cycle_3_arm_clears_the_no_network_floor():
    analysis = analyse_study(
        collect_study(REPOSITORY_ROOT, "mess_3_kelly_cycle_3"),
        REFERENCES,
    )
    conditions = analysis["conditions"]

    assert conditions["conditional_decoupled_kelly_iqn"]["exceeds_no_network_floor"]
    for arm in ("ppo", "iqn", "conditional_decoupled_kelly_mean"):
        assert not conditions[arm]["exceeds_no_network_floor"], arm

    # Plain PPO sits far below a probe on the raw observations.
    assert conditions["ppo"]["fraction_of_usable_range"] < -1.0


def test_every_kelly_cycle_2_kelly_arm_falls_below_the_no_network_floor():
    analysis = analyse_study(
        collect_study(REPOSITORY_ROOT, "mess_3_kelly_cycle_2"),
        REFERENCES,
    )
    conditions = analysis["conditions"]

    for arm, values in conditions.items():
        expected = arm.startswith("correctness")
        assert values["exceeds_no_network_floor"] is expected, arm
