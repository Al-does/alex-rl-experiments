"""Exact design diagnostics for the sticky-state cycle-4 baseline."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from envs.mess3.model import sticky_control_model
from experiments.mess3_reward_state_action_symmetry_cycle_4.task import (
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
    ActionSymmetryTask,
)


EFFECT_SIZE = 1.5
EXPECTED_ORACLE_POLICIES = {
    1: (POSITIVE_ACTION, POSITIVE_ACTION, POSITIVE_ACTION),
    2: (POSITIVE_ACTION, POSITIVE_ACTION, NOOP_ACTION),
    3: (POSITIVE_ACTION, NEGATIVE_ACTION, NOOP_ACTION),
}


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    system = np.asarray(transition, dtype=np.float64).T - np.eye(3)
    system[-1] = 1.0
    return np.linalg.solve(system, np.array([0.0, 0.0, 1.0]))


def _policy_transition(
    task: ActionSymmetryTask,
    policy: tuple[int, int, int],
) -> np.ndarray:
    return np.stack(
        [
            task.transition_matrix_for_action(policy[state])[state]
            for state in range(3)
        ]
    )


def _rank_policies(task: ActionSymmetryTask) -> list[tuple[float, tuple[int, ...]]]:
    ranked = [
        (
            float(_stationary_distribution(_policy_transition(task, policy))[2]),
            policy,
        )
        for policy in itertools.product(range(3), repeat=3)
    ]
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def analytic_design_summary() -> dict[str, Any]:
    """Return exact transition and full-state occupancy design values."""

    model = sticky_control_model(alpha=0.85)
    variants = {}
    tasks = {
        variant: ActionSymmetryTask(
            model=model,
            variant=variant,
            effect_size=EFFECT_SIZE,
        )
        for variant in (1, 2, 3)
    }
    for variant, task in tasks.items():
        ranked = _rank_policies(task)
        oracle_occupancy, oracle_policy = ranked[0]
        expected = EXPECTED_ORACLE_POLICIES[variant]
        if oracle_policy != expected:
            raise AssertionError(
                f"variant {variant} oracle {oracle_policy} != expected {expected}"
            )
        runner_up_occupancy = ranked[1][0]
        always_positive = (POSITIVE_ACTION,) * 3
        variants[f"variant_{variant}"] = {
            "oracle_policy": list(oracle_policy),
            "oracle_stationary_state_2": oracle_occupancy,
            "runner_up_stationary_state_2": runner_up_occupancy,
            "oracle_gap": oracle_occupancy - runner_up_occupancy,
            "always_positive_stationary_state_2": float(
                _stationary_distribution(
                    _policy_transition(task, always_positive)
                )[2]
            ),
        }

    variant_2 = tasks[2]
    noop = variant_2.transition_matrix_for_action(NOOP_ACTION)
    positive = variant_2.transition_matrix_for_action(POSITIVE_ACTION)
    outside_gain = float(positive[0, 2] - noop[0, 2])
    state_two_retention_gain = float(noop[2, 2] - positive[2, 2])
    threshold = outside_gain / (outside_gain + state_two_retention_gain)
    return {
        "baseline_transition_matrix": model.transition_matrix.tolist(),
        "effect_size": EFFECT_SIZE,
        "emission_alpha": 0.85,
        "variant_2_state_2_probabilities": {
            "noop": float(noop[2, 2]),
            "positive": float(positive[2, 2]),
            "negative": float(
                variant_2.transition_matrix_for_action(NEGATIVE_ACTION)[2, 2]
            ),
        },
        "variant_2_nonreward_positive_state_2_probability": float(
            positive[0, 2]
        ),
        "variant_2_one_step_noop_belief_threshold": threshold,
        "fully_observed": variants,
    }
