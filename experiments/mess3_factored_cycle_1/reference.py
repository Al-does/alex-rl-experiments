"""Exact filters and deterministic pre-training audits for factored MESS3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from envs.hmm import factor_marginals, product_distribution
from experiments.mess3_factored_cycle_1.dynamics import (
    BASE_TRANSITION,
    GOAL_STATE,
    N_FACTOR_STATES,
    action_kernels,
    additive_matrix_residual,
    e2_action_transitions,
    reward_vector,
    stationary_distribution,
    value_iteration,
)


PRE_REGISTERED_THRESHOLDS = {
    "belief_demand_min": 0.015,
    "e2_incentive_point_max": 0.001,
    "e2_incentive_upper_95_max": 0.0015,
    "e2_visibility_min_nats": 0.003,
    "max_standard_error": 5e-4,
}

# These are design-document reference targets, not measurements made by this
# code. Full Monte Carlo verification intentionally remains a separate,
# expensive pre-training operation.
REFERENCE_TARGETS = {
    "e1_demand_gap": 0.0206,
    "e2_lambda_1_incentive": 0.0007,
    "e2_lambda_1_visibility_nats": 0.0041,
    "e2_belief_demand": 0.0184,
    "e3b_demand_gap": 0.0178,
    "e3c_demand_gap": 0.0205,
    "e4_demand_gap": 0.0213,
}


def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    total = values.sum(axis=-1, keepdims=True)
    if (total <= 0.0).any():
        raise ValueError("probability vector has zero normalizer")
    return values / total


def posterior_from_symbol(
    prior: np.ndarray,
    symbol: int,
    emission: np.ndarray,
) -> np.ndarray:
    """Condition a prior on one current symbol."""

    return normalize(
        np.asarray(prior, dtype=np.float64)
        * np.asarray(emission, dtype=np.float64)[:, int(symbol)]
    )


def aware_filter_update(
    belief: np.ndarray,
    action: int,
    next_symbol: int,
    *,
    kernels: np.ndarray,
    emission: np.ndarray,
) -> np.ndarray:
    """Predict through the selected kernel, then condition on the next symbol."""

    predicted = np.asarray(belief, dtype=np.float64) @ kernels[int(action)]
    return posterior_from_symbol(predicted, next_symbol, emission)


def qmdp_action(belief: np.ndarray, q_values: np.ndarray) -> int:
    """Select the lowest-index maximizer of expected fully observed Q."""

    scores = np.asarray(belief, dtype=np.float64) @ np.asarray(
        q_values,
        dtype=np.float64,
    )
    return int(np.argmax(scores))


def factor_targets(joint_beliefs: np.ndarray) -> dict[str, np.ndarray]:
    """Build probe targets using PR 35's public belief algebra."""

    first, second = factor_marginals(joint_beliefs, (3, 3))
    product = product_distribution((first, second))
    return {
        "joint": np.asarray(joint_beliefs, dtype=np.float64),
        "f1": first,
        "f2": second,
        "product": product,
        "joint_residual": np.asarray(joint_beliefs, dtype=np.float64) - product,
        "f2_goal_block": second[..., GOAL_STATE : GOAL_STATE + 1],
        "f2_within_n": (second[..., 0] - second[..., 1])[..., None],
    }


def coarse_e2_transition() -> np.ndarray:
    """Return E2's action-indexed two-state ``{N, G}`` kernels."""

    fine = e2_action_transitions()
    return np.stack(
        [
            np.array(
                [
                    [1.0 - matrix[0, GOAL_STATE], matrix[0, GOAL_STATE]],
                    [1.0 - matrix[GOAL_STATE, GOAL_STATE], matrix[GOAL_STATE, GOAL_STATE]],
                ],
                dtype=np.float64,
            )
            for matrix in fine
        ]
    )


def coarse_e2_emission(alpha2: float) -> np.ndarray:
    """Return ``P(1{x2=2} | N/G)`` as a two-symbol likelihood matrix."""

    if not 0.0 <= alpha2 <= 1.0:
        raise ValueError("alpha2 must lie in [0, 1]")
    p_goal_symbol_given_n = (1.0 - alpha2) / 2.0
    p_goal_symbol_given_g = alpha2
    return np.array(
        [
            [1.0 - p_goal_symbol_given_n, p_goal_symbol_given_n],
            [1.0 - p_goal_symbol_given_g, p_goal_symbol_given_g],
        ],
        dtype=np.float64,
    )


def coarse_e2_update(
    belief: np.ndarray,
    action: int,
    next_x2: int,
    *,
    alpha2: float,
) -> np.ndarray:
    """Advance the exactly s1-free binary E2 reference filter."""

    predicted = np.asarray(belief, dtype=np.float64) @ coarse_e2_transition()[
        int(action)
    ]
    binary_symbol = int(int(next_x2) == GOAL_STATE)
    return posterior_from_symbol(
        predicted,
        binary_symbol,
        coarse_e2_emission(alpha2),
    )


def e2_blind_transitions(coupling_lambda: float) -> np.ndarray:
    """Average E2's fine F2 kernels over autonomous F1 stationarity."""

    context_weights = stationary_distribution(BASE_TRANSITION)
    fine_actions = e2_action_transitions()
    from experiments.mess3_factored_cycle_1.dynamics import (
        modulate_within_non_goal,
    )

    return np.stack(
        [
            sum(
                context_weights[context]
                * modulate_within_non_goal(
                    matrix,
                    context_state=context,
                    coupling_lambda=coupling_lambda,
                )
                for context in range(N_FACTOR_STATES)
            )
            for matrix in fine_actions
        ]
    )


def e2_lumpability_audit(
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> dict[str, Any]:
    """Audit A1 across the registered dose-response sweep."""

    worst = 0.0
    by_lambda: dict[str, float] = {}
    for coupling_lambda in lambdas:
        kernels = action_kernels(
            "e2_tilt",
            coupling_lambda=coupling_lambda,
        )
        lambda_worst = 0.0
        for action_kernel in kernels:
            masses = np.empty((3, 3), dtype=np.float64)
            for first in range(3):
                for second in range(3):
                    source = 3 * first + second
                    masses[first, second] = action_kernel[
                        source,
                        GOAL_STATE::3,
                    ].sum()
            reference = masses[0]
            lambda_worst = max(
                lambda_worst,
                float(np.max(np.abs(masses - reference))),
                float(abs(reference[0] - reference[1])),
            )
        by_lambda[str(coupling_lambda)] = lambda_worst
        worst = max(worst, lambda_worst)
    return {
        "audit": "A1_lumpability",
        "passed": bool(worst <= 1e-12),
        "tolerance": 1e-12,
        "worst_deviation": worst,
        "by_lambda": by_lambda,
    }


def value_invariance_audit() -> dict[str, Any]:
    """Audit A2 for E2's sweep and E4's gauge symmetry."""

    conditions: dict[str, float] = {}
    for coupling_lambda in (0.0, 0.5, 1.0, 1.5, 2.0):
        solved = value_iteration(
            action_kernels("e2_tilt", coupling_lambda=coupling_lambda),
            reward_vector("f2_goal"),
        )
        grid = solved.value.reshape(3, 3)
        conditions[f"e2_lambda_{coupling_lambda}"] = float(
            np.max(np.ptp(grid, axis=0))
        )
    gauge = value_iteration(
        action_kernels("e4_gauge"),
        reward_vector("f2_goal"),
    )
    conditions["e4_gauge"] = float(
        np.max(np.ptp(gauge.value.reshape(3, 3), axis=0))
    )
    worst = max(conditions.values())
    return {
        "audit": "A2_value_invariance",
        "passed": bool(worst <= 1e-9),
        "tolerance": 1e-9,
        "worst_deviation": worst,
        "conditions": conditions,
    }


def e3_function_coupling_audit() -> dict[str, Any]:
    """Record fully observed value/policy structure for E3a/b/c."""

    product = action_kernels("product")
    diagonal = action_kernels("diagonal")
    additive = value_iteration(product, reward_vector("additive"))
    conjunctive = value_iteration(product, reward_vector("conjunctive"))
    shared = value_iteration(diagonal, reward_vector("additive"))

    def product_policy_factorizes(policy: np.ndarray) -> bool:
        actions = policy.reshape(3, 3)
        first = actions // 3
        second = actions % 3
        return bool(
            np.all(first == first[:, :1])
            and np.all(second == second[:1, :])
        )

    shared_grid = shared.policy.reshape(3, 3)
    shared_is_single_factor = bool(
        np.all(shared_grid == shared_grid[:, :1])
        or np.all(shared_grid == shared_grid[:1, :])
    )
    return {
        "e3a": {
            "value_nonadditive_residual": additive_matrix_residual(additive.value),
            "policy_factorizes": product_policy_factorizes(additive.policy),
        },
        "e3b": {
            "value_nonadditive_residual": additive_matrix_residual(
                conjunctive.value
            ),
            "policy_factorizes": product_policy_factorizes(conjunctive.policy),
        },
        "e3c": {
            "value_nonadditive_residual": additive_matrix_residual(shared.value),
            "policy_is_single_factor_rule": shared_is_single_factor,
            "policy_grid": shared_grid.tolist(),
        },
    }


def structural_audit_report() -> dict[str, Any]:
    """Run deterministic audits available without Monte Carlo reference jobs."""

    a1 = e2_lumpability_audit()
    a2 = value_invariance_audit()
    return {
        "status": "passed" if a1["passed"] and a2["passed"] else "failed",
        "audits": {
            "A1": a1,
            "A2": a2,
            "E3_function_coupling": e3_function_coupling_audit(),
        },
        "pre_registered_thresholds": PRE_REGISTERED_THRESHOLDS,
        "design_document_reference_targets": REFERENCE_TARGETS,
        "monte_carlo_audits": {
            "status": "not_run",
            "reason": (
                "A3-A6 require the registered 4096-chain reference campaign; "
                "a smoke run must not be presented as that acceptance test."
            ),
        },
        "e2_scope_note": (
            "The latent reward process is exactly lumpable. With the full "
            "three-symbol x2 channel, fine information slightly improves the "
            "block posterior, so E2 tests an approximate incentive-controlled "
            "quotient rather than exact belief-state bisimulation."
        ),
    }
