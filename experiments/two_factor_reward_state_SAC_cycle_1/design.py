"""Reference policies and the preregistered partial-observability audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.hmm import stationary_distribution
from envs.mess3.model import emission_matrix

from experiments.two_factor_reward_state_SAC_cycle_1.process import (
    MESS3_ALPHA,
    TRANSITION_MATRIX,
)
from experiments.two_factor_reward_state_SAC_cycle_1.task import shifted_transition


GAMMA = 0.99
AUDIT_CHAINS = 4_096
AUDIT_STEPS = 6_000
AUDIT_BURN_IN = 500
MINIMUM_DEMAND_GAP = 0.015
MAXIMUM_STANDARD_ERROR = 5e-4
REFERENCE_VALUES = {
    "fully_observed": 0.5555555555555556,
    "qmdp": 0.4439,
    "best_constant": 0.4233128834355829,
    "reactive": 0.4210,
}


def controlled_transitions() -> np.ndarray:
    """Return one transition matrix per factor-level shift."""

    return np.stack([shifted_transition(shift) for shift in range(3)])


def fully_observed_q_values(
    *,
    gamma: float = GAMMA,
    tolerance: float = 1e-13,
) -> np.ndarray:
    """Solve the fully observed discounted control problem by value iteration."""

    transitions = controlled_transitions()
    reward = np.array([0.0, 0.0, 1.0])
    value = np.zeros(3)
    for _ in range(100_000):
        q_values = reward[:, None] + gamma * np.einsum(
            "asj,j->sa",
            transitions,
            value,
        )
        updated = q_values.max(axis=1)
        if np.max(np.abs(updated - value)) <= tolerance:
            return q_values
        value = updated
    raise RuntimeError("fully observed value iteration did not converge")


def fully_observed_occupancy() -> float:
    """Return exact state-2 occupancy under the fully observed greedy policy."""

    transitions = controlled_transitions()
    policy = fully_observed_q_values().argmax(axis=1)
    controlled = np.stack(
        [transitions[policy[state], state] for state in range(3)]
    )
    return float(stationary_distribution(controlled)[2])


def constant_occupancies() -> np.ndarray:
    """Return exact state-2 occupancy for each fixed shift."""

    return np.array(
        [
            stationary_distribution(transition)[2]
            for transition in controlled_transitions()
        ]
    )


def _sample_rows(
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> np.ndarray:
    draws = rng.random(len(probabilities))
    return (draws[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)


def _simulate_policy(
    mode: str,
    *,
    seed: int,
    chains: int,
    steps: int,
    burn_in: int,
) -> tuple[float, float]:
    if mode not in {"qmdp", "reactive"}:
        raise ValueError("mode must be qmdp or reactive")
    if chains <= 1 or steps <= burn_in:
        raise ValueError("audit needs multiple chains and post-burn-in steps")

    rng = np.random.default_rng(seed)
    transitions = controlled_transitions()
    emissions = emission_matrix(MESS3_ALPHA)
    q_values = fully_observed_q_values()
    fully_observed_policy = q_values.argmax(axis=1)
    states = rng.integers(3, size=chains)
    tokens = _sample_rows(rng, emissions[states])
    beliefs = emissions[:, tokens].T
    beliefs /= beliefs.sum(axis=1, keepdims=True)
    occupancy = np.zeros(chains)

    for step in range(steps):
        actions = (
            (beliefs @ q_values).argmax(axis=1)
            if mode == "qmdp"
            else fully_observed_policy[tokens]
        )
        if step >= burn_in:
            occupancy += states == 2
        states = _sample_rows(rng, transitions[actions, states])
        tokens = _sample_rows(rng, emissions[states])
        beliefs = np.einsum(
            "ni,nij->nj",
            beliefs,
            transitions[actions],
        )
        beliefs *= emissions[:, tokens].T
        beliefs /= beliefs.sum(axis=1, keepdims=True)

    chain_means = occupancy / (steps - burn_in)
    return (
        float(chain_means.mean()),
        float(chain_means.std(ddof=1) / np.sqrt(chains)),
    )


def demand_audit(
    *,
    seed: int = 20260828,
    chains: int = AUDIT_CHAINS,
    steps: int = AUDIT_STEPS,
    burn_in: int = AUDIT_BURN_IN,
) -> dict[str, Any]:
    """Run the required QMDP-versus-memoryless regression audit."""

    qmdp, qmdp_se = _simulate_policy(
        "qmdp",
        seed=seed,
        chains=chains,
        steps=steps,
        burn_in=burn_in,
    )
    reactive, reactive_se = _simulate_policy(
        "reactive",
        seed=seed + 1,
        chains=chains,
        steps=steps,
        burn_in=burn_in,
    )
    constants = constant_occupancies()
    best_constant = float(constants.max())
    demand_gap = qmdp - max(reactive, best_constant)
    gap_se = float(np.hypot(qmdp_se, reactive_se))
    report = {
        "alpha": MESS3_ALPHA,
        "chains": chains,
        "steps": steps,
        "burn_in": burn_in,
        "fully_observed": fully_observed_occupancy(),
        "qmdp": qmdp,
        "qmdp_standard_error": qmdp_se,
        "constant_occupancies": constants.tolist(),
        "best_constant": best_constant,
        "reactive": reactive,
        "reactive_standard_error": reactive_se,
        "demand_gap": demand_gap,
        "demand_gap_standard_error": gap_se,
        "minimum_demand_gap": MINIMUM_DEMAND_GAP,
        "maximum_standard_error": MAXIMUM_STANDARD_ERROR,
    }
    if demand_gap < MINIMUM_DEMAND_GAP:
        raise AssertionError(f"demand gap {demand_gap:.6f} is below the audit floor")
    if gap_se > MAXIMUM_STANDARD_ERROR:
        raise AssertionError(f"demand-gap s.e. {gap_se:.6f} exceeds the audit ceiling")
    return report
