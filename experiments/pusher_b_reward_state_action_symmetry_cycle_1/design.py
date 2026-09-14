from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from envs.hmm import stationary_distribution
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.process import (
    EFFECT_SIZE,
    PRESETS,
    REWARD_STATES,
    pusher_b_model,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.task import (
    N_ACTIONS,
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
    ActionSymmetryTask,
)


MAX_CONTROLLED_TRANSITION_PROBABILITY = 0.96


def expected_oracle_policy(
    variant: int,
    reward_state: int,
) -> tuple[int, int, int]:
    if variant == 1:
        return (POSITIVE_ACTION,) * 3
    policy = [POSITIVE_ACTION] * 3
    policy[reward_state] = NOOP_ACTION
    if variant == 2:
        return tuple(policy)
    if variant != 3:
        raise ValueError("variant must be one of 1, 2, or 3")
    policy[(reward_state + 2) % 3] = NEGATIVE_ACTION
    return tuple(policy)


def controlled_kernels(
    preset: str,
    variant: int,
    reward_state: str,
    effect_size: float = EFFECT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    task = ActionSymmetryTask(
        model=pusher_b_model(preset),
        variant=variant,
        reward_state=reward_state,
        effect_size=effect_size,
    )
    transitions = np.stack(
        [
            task.transition_matrix_for_action(action)
            for action in range(N_ACTIONS)
        ]
    )
    edges = np.stack(
        [
            task.edge_transition_matrices_for_action(action)
            for action in range(N_ACTIONS)
        ]
    )
    return transitions, edges


def _policy_transition(
    transitions: np.ndarray,
    policy: tuple[int, int, int],
) -> np.ndarray:
    return np.stack(
        [transitions[policy[state], state] for state in range(3)]
    )


def _rank_policies(
    transitions: np.ndarray,
    reward_state: int,
) -> list[tuple[float, tuple[int, int, int]]]:
    ranked = [
        (
            float(
                stationary_distribution(
                    _policy_transition(transitions, policy)
                )[reward_state]
            ),
            policy,
        )
        for policy in itertools.product(range(N_ACTIONS), repeat=3)
    ]
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def condition_design_summary(
    preset: str,
    variant: int,
    reward_state: str,
) -> dict[str, Any]:
    model = pusher_b_model(preset)
    reward_state_index = model.state_labels.index(reward_state)
    transitions, edges = controlled_kernels(
        preset,
        variant,
        reward_state,
    )
    ranked = _rank_policies(transitions, reward_state_index)
    oracle_occupancy, oracle_policy = ranked[0]
    expected_policy = expected_oracle_policy(variant, reward_state_index)
    if oracle_policy != expected_policy:
        raise AssertionError(
            f"{preset}/{reward_state}/variant_{variant} oracle "
            f"{oracle_policy} != expected {expected_policy}"
        )
    maximum_probability = float(transitions.max())
    if maximum_probability >= MAX_CONTROLLED_TRANSITION_PROBABILITY:
        raise AssertionError(
            "action effect makes a controlled transition nearly deterministic"
        )
    baseline_occupancy = float(model.initial_distribution[reward_state_index])
    if oracle_occupancy <= baseline_occupancy:
        raise AssertionError("oracle policy does not improve reward occupancy")
    return {
        "preset": preset,
        "variant": variant,
        "reward_state": reward_state,
        "reward_state_index": reward_state_index,
        "effect_size": EFFECT_SIZE,
        "expected_oracle_policy": list(expected_policy),
        "oracle_policy": list(oracle_policy),
        "baseline_stationary_reward_occupancy": baseline_occupancy,
        "oracle_stationary_reward_occupancy": oracle_occupancy,
        "runner_up_stationary_reward_occupancy": ranked[1][0],
        "oracle_gap": oracle_occupancy - ranked[1][0],
        "minimum_controlled_transition_probability": float(
            transitions.min()
        ),
        "maximum_controlled_transition_probability": maximum_probability,
        "transition_matrices": transitions.tolist(),
        "edge_transition_matrices": edges.tolist(),
    }


def analytic_design_summary() -> dict[str, Any]:
    return {
        "study": "pusher_b_reward_state_action_symmetry_cycle_1",
        "effect_size": EFFECT_SIZE,
        "effect_size_selection": (
            "An odds tilt of 1.5 matches the established cycle-5 control "
            "ladder, gives the intended unique full-state oracle in all 12 "
            "conditions, and keeps every controlled destination probability "
            "below 0.96."
        ),
        "variant_semantics": {
            "variant_1": (
                "positive is optimal in every hidden state"
            ),
            "variant_2": (
                "noop is optimal in the rewarding state and positive is "
                "optimal in both non-rewarding states"
            ),
            "variant_3": (
                "noop is optimal in the rewarding state; following cyclic "
                "state order, positive and negative are optimal in the two "
                "non-rewarding states"
            ),
        },
        "conditions": [
            condition_design_summary(preset, variant, reward_state)
            for preset in PRESETS
            for reward_state in REWARD_STATES
            for variant in (1, 2, 3)
        ],
    }
