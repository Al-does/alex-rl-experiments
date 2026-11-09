import itertools

import numpy as np

from envs.mess3.model import control_model
from experiments.mess3_reward_state_action_symmetry_cycle_5.design import (
    CYCLE_5_TRANSITION_MATRIX,
    EFFECT_SIZE,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.task import (
    ActionSymmetryTask,
)


EMISSION_ALPHA = 0.75
TRANSITION_RATE = 0.25
BASELINE_TRANSITION_MATRIX = (
    (0.9375, 0.0375, 0.025),
    (0.0375, 0.9375, 0.025),
    (0.075, 0.075, 0.85),
)
CONDITION_NAMES = {
    1: "Trivial Quotient",
    2: "Reward-State Quotient",
    3: "Full-State Quotient",
}
THREE_TOKEN_POLICIES = {
    2: (
        ((1, 1, 0), (1, 1, 0), (1, 0, 0)),
        ((1, 1, 0), (1, 1, 0), (0, 1, 0)),
        ((1, 1, 0), (1, 1, 0), (0, 0, 0)),
    ),
    3: (
        ((1, 1, 0), (1, 2, 0), (1, 0, 0)),
        ((1, 2, 0), (2, 2, 0), (0, 2, 0)),
        ((1, 0, 0), (0, 2, 0), (0, 0, 0)),
    ),
}


def model_kwargs():
    return {
        "alpha": EMISSION_ALPHA,
        "transition_matrix": [list(row) for row in BASELINE_TRANSITION_MATRIX],
    }


def controlled_model():
    return control_model(**model_kwargs())


def stationary_distribution(transition):
    system = np.asarray(transition, dtype=np.float64).T - np.eye(len(transition))
    system[-1] = 1.0
    target = np.zeros(len(transition))
    target[-1] = 1.0
    return np.linalg.solve(system, target)


def analytic_design_summary():
    expected = (
        (1 - TRANSITION_RATE) * np.eye(3)
        + TRANSITION_RATE * np.asarray(CYCLE_5_TRANSITION_MATRIX)
    )
    if not np.allclose(expected, BASELINE_TRANSITION_MATRIX, atol=1e-15):
        raise AssertionError("sticky baseline must be formed before action tilting")
    model = controlled_model()
    variants = {}
    for variant in CONDITION_NAMES:
        task = ActionSymmetryTask(model=model, variant=variant, effect_size=EFFECT_SIZE)
        transitions = np.stack([task.transition_matrix_for_action(a) for a in range(3)])
        memoryless = []
        privileged = []
        for policy in itertools.product(range(3), repeat=3):
            observable_transition = sum(
                model.emission_matrix[:, token, None] * transitions[action]
                for token, action in enumerate(policy)
            )
            memoryless.append(
                (float(stationary_distribution(observable_transition)[2]), policy)
            )
            state_transition = np.stack(
                [transitions[action, state] for state, action in enumerate(policy)]
            )
            privileged.append(
                (float(stationary_distribution(state_transition)[2]), policy)
            )
        memoryless_reward, memoryless_policy = max(memoryless)
        oracle_reward, oracle_policy = max(privileged)
        record = {
            "name": CONDITION_NAMES[variant],
            "best_deterministic_current_token_policy": list(memoryless_policy),
            "best_deterministic_current_token_stationary_reward": memoryless_reward,
            "privileged_state_policy": list(oracle_policy),
            "privileged_state_stationary_reward": oracle_reward,
        }
        if variant in THREE_TOKEN_POLICIES:
            policy = np.asarray(THREE_TOKEN_POLICIES[variant])
            transition = np.zeros((27, 27))
            for state, older, previous, token, next_state in itertools.product(
                range(3), repeat=5
            ):
                transition[
                    9 * state + 3 * older + previous,
                    9 * next_state + 3 * previous + token,
                ] = (
                    model.emission_matrix[state, token]
                    * transitions[policy[older, previous, token], state, next_state]
                )
            reward = float(stationary_distribution(transition)[18:].sum())
            record.update(
                {
                    "feasible_three_token_policy": policy.tolist(),
                    "feasible_three_token_stationary_reward": reward,
                    "three_token_gain_over_deterministic_current_token": (
                        reward - memoryless_reward
                    ),
                    "history_reference_scope": (
                        "fixed feasible three-token lookup; not a Bayes optimum "
                        "or a learned-model result"
                    ),
                }
            )
        variants[f"variant_{variant}"] = record
    return {
        "emission_alpha": EMISSION_ALPHA,
        "transition_rate": TRANSITION_RATE,
        "baseline_transition_matrix": model.transition_matrix.tolist(),
        "effect_size": EFFECT_SIZE,
        "tilt_order": "mix the baseline with identity, then tilt reward-state odds",
        "reference_scope": "long-run expected occupancy; not an empirical reward cap",
        "variants": variants,
    }
