from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from envs.hmm import stationary_distribution
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.design import (
    MAX_CONTROLLED_TRANSITION_PROBABILITY,
    expected_oracle_policy,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.task import (
    N_ACTIONS,
    ActionSymmetryTask,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_2.process import (
    DESTINATION_EMISSION_MATRIX,
    EFFECT_SIZE,
    REWARD_STATE,
    TRANSITION_MATRIX,
    sticky_cycle_model,
)


VARIANTS = (2, 3)


def controlled_kernels(variant: int) -> tuple[np.ndarray, np.ndarray]:
    task = ActionSymmetryTask(
        model=sticky_cycle_model(),
        variant=variant,
        reward_state=REWARD_STATE,
        effect_size=EFFECT_SIZE,
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


def condition_design_summary(variant: int) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("variant must be 2 or 3")
    model = sticky_cycle_model()
    reward_state = model.state_labels.index(REWARD_STATE)
    transitions, edges = controlled_kernels(variant)
    ranked = sorted(
        [
            (
                float(
                    stationary_distribution(
                        _policy_transition(transitions, policy)
                    )[reward_state]
                ),
                policy,
            )
            for policy in itertools.product(range(N_ACTIONS), repeat=3)
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    oracle_occupancy, oracle_policy = ranked[0]
    expected_policy = expected_oracle_policy(variant, reward_state)
    if oracle_policy != expected_policy:
        raise AssertionError(
            f"variant_{variant} oracle {oracle_policy} != {expected_policy}"
        )
    maximum_probability = float(transitions.max())
    if maximum_probability >= MAX_CONTROLLED_TRANSITION_PROBABILITY:
        raise AssertionError("controlled transition is nearly deterministic")
    return {
        "variant": variant,
        "reward_state": REWARD_STATE,
        "reward_state_index": reward_state,
        "effect_size": EFFECT_SIZE,
        "expected_oracle_policy": list(expected_policy),
        "oracle_policy": list(oracle_policy),
        "baseline_stationary_reward_occupancy": float(
            model.initial_distribution[reward_state]
        ),
        "oracle_stationary_reward_occupancy": oracle_occupancy,
        "runner_up_stationary_reward_occupancy": ranked[1][0],
        "oracle_gap": oracle_occupancy - ranked[1][0],
        "minimum_controlled_transition_probability": float(
            transitions.min()
        ),
        "maximum_controlled_transition_probability": maximum_probability,
        "reference_transition_matrix": TRANSITION_MATRIX.tolist(),
        "destination_emission_matrix": (
            DESTINATION_EMISSION_MATRIX.tolist()
        ),
        "transition_matrices": transitions.tolist(),
        "edge_transition_matrices": edges.tolist(),
    }


def analytic_design_summary() -> dict[str, Any]:
    return {
        "study": "pusher_b_reward_state_action_symmetry_cycle_2",
        "hmm": (
            "a 90%-persistent three-state cycle with a 60/40 directional "
            "movement split and noisy destination-state emissions"
        ),
        "effect_size": EFFECT_SIZE,
        "reward_state": REWARD_STATE,
        "conditions": [
            condition_design_summary(variant) for variant in VARIANTS
        ],
    }
