"""Exact single-factor design diagnostics for product variant-3 control."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from envs.hmm import stationary_distribution
from experiments.mess3_reward_state_action_symmetry_cycle_5.design import (
    EFFECT_SIZE,
    EXPECTED_ORACLE_POLICIES,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import MESS3_ALPHA
from experiments.two_factor_reward_state_SAC_cycle_2.task import factor_transition


GAMMA = 0.99


def _policy_transition(policy: tuple[int, int, int]) -> np.ndarray:
    transitions = [factor_transition(action) for action in range(3)]
    return np.stack([transitions[policy[state]][state] for state in range(3)])


def analytic_design_summary() -> dict[str, Any]:
    ranked = sorted(
        (
            float(stationary_distribution(_policy_transition(policy))[2]),
            policy,
        )
        for policy in itertools.product(range(3), repeat=3)
    )
    oracle_occupancy, oracle_policy = ranked[-1]
    if oracle_policy != EXPECTED_ORACLE_POLICIES[3]:
        raise AssertionError(
            f"variant-3 oracle {oracle_policy} != {EXPECTED_ORACLE_POLICIES[3]}"
        )
    return {
        "factor_action_semantics": "action-symmetry cycle 5 variant 3",
        "effect_size": EFFECT_SIZE,
        "emission_alpha": MESS3_ALPHA,
        "oracle_policy_by_state": list(oracle_policy),
        "oracle_single_factor_state_2_occupancy": oracle_occupancy,
        "runner_up_single_factor_state_2_occupancy": ranked[-2][0],
        "oracle_gap": oracle_occupancy - ranked[-2][0],
        "reward_both_oracle_mean_reward": 2.0 * oracle_occupancy,
        "reward_factor_1_oracle_mean_reward": oracle_occupancy,
    }
