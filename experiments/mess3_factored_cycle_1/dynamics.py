"""Exact dynamics and finite-state control calculations for factored MESS3.

Joint states and symbols use mixed-radix order ``3 * first + second``.  The
static HMM is assembled by :func:`envs.hmm.factored_model`; this module owns
the experiment-specific action-conditioned kernels that the HMM task executes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


N_FACTOR_STATES = 3
N_JOINT_STATES = 9
GOAL_STATE = 2
E2_ATTRACT = 1.5
E2_REPEL = 2.5

BASE_TRANSITION = np.array(
    [
        [0.75, 0.15, 0.10],
        [0.15, 0.75, 0.10],
        [0.30, 0.30, 0.40],
    ],
    dtype=np.float64,
)
BASE_TRANSITION.setflags(write=False)


def _stochastic(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise ValueError(f"{name} must be a finite row-stochastic square matrix")
    return values


def shift_matrix(shift: int, *, size: int = N_FACTOR_STATES) -> np.ndarray:
    """Return the destination relabeling ``j -> j + shift (mod size)``."""

    if size <= 0:
        raise ValueError("size must be positive")
    return np.roll(np.eye(size, dtype=np.float64), int(shift) % size, axis=1)


def shifted_transition(
    shift: int,
    *,
    base: np.ndarray = BASE_TRANSITION,
) -> np.ndarray:
    """Apply a row-uniform cyclic shift after the baseline transition."""

    return _stochastic(base, name="base") @ shift_matrix(
        shift,
        size=np.asarray(base).shape[0],
    )


def tilt_transition(
    *,
    attract: float = 0.0,
    repel: float = 0.0,
    base: np.ndarray = BASE_TRANSITION,
) -> np.ndarray:
    """Apply E2's row-uniform destination-2 attract and stay-state repel."""

    matrix = _stochastic(base, name="base")
    if not np.isfinite(attract) or attract < 0.0:
        raise ValueError("attract must be finite and non-negative")
    if not np.isfinite(repel) or repel < 0.0:
        raise ValueError("repel must be finite and non-negative")
    log_weights = np.log(matrix)
    log_weights[:, GOAL_STATE] += float(attract)
    diagonal = np.arange(len(matrix))
    log_weights[diagonal, diagonal] -= float(repel)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=1, keepdims=True)


def e2_action_transitions(
    *,
    base: np.ndarray = BASE_TRANSITION,
    attract: float = E2_ATTRACT,
    repel: float = E2_REPEL,
) -> np.ndarray:
    """Return E2's ``{noop, attract, attract+repel}`` F2 menu."""

    return np.stack(
        [
            _stochastic(base, name="base"),
            tilt_transition(attract=attract, base=base),
            tilt_transition(attract=attract, repel=repel, base=base),
        ]
    )


def modulate_within_non_goal(
    transition: np.ndarray,
    *,
    context_state: int,
    coupling_lambda: float,
) -> np.ndarray:
    """Change only the destination-0:destination-1 odds for E2.

    Goal-block mass is preserved exactly.  The context contrast is ``(+1,
    -1, 0)`` for F1 states ``(0, 1, 2)``.
    """

    matrix = np.array(_stochastic(transition, name="transition"), copy=True)
    context = int(context_state)
    if not 0 <= context < N_FACTOR_STATES:
        raise ValueError("context_state must be 0, 1, or 2")
    if not np.isfinite(coupling_lambda) or coupling_lambda < 0.0:
        raise ValueError("coupling_lambda must be finite and non-negative")
    contrast = (1.0, -1.0, 0.0)[context]
    multiplier = np.exp(float(coupling_lambda) * contrast)
    non_goal_mass = matrix[:, :GOAL_STATE].sum(axis=1)
    odds = (matrix[:, 0] / matrix[:, 1]) * multiplier
    matrix[:, 0] = non_goal_mass * odds / (1.0 + odds)
    matrix[:, 1] = non_goal_mass / (1.0 + odds)
    return matrix


def _joint_from_source_conditioned_f2(
    f1_transition: np.ndarray,
    f2_by_source_f1: np.ndarray,
) -> np.ndarray:
    """Compose F1 dynamics with an F2 kernel selected by current F1 state."""

    first = _stochastic(f1_transition, name="f1_transition")
    second = np.asarray(f2_by_source_f1, dtype=np.float64)
    if second.shape != (N_FACTOR_STATES, N_FACTOR_STATES, N_FACTOR_STATES):
        raise ValueError("f2_by_source_f1 must have shape (3, 3, 3)")
    joint = np.empty((N_JOINT_STATES, N_JOINT_STATES), dtype=np.float64)
    for source_f1 in range(N_FACTOR_STATES):
        for source_f2 in range(N_FACTOR_STATES):
            source = N_FACTOR_STATES * source_f1 + source_f2
            joint[source] = np.outer(
                first[source_f1],
                second[source_f1, source_f2],
            ).reshape(-1)
    return _stochastic(joint, name="joint transition")


def product_shift_kernels(
    *,
    base: np.ndarray = BASE_TRANSITION,
) -> np.ndarray:
    """Return nine independent product-action kernels in ``3*a1+a2`` order."""

    shifts = [shifted_transition(index, base=base) for index in range(3)]
    return np.stack(
        [np.kron(shifts[first], shifts[second]) for first in range(3) for second in range(3)]
    )


def diagonal_shift_kernels(
    *,
    base: np.ndarray = BASE_TRANSITION,
) -> np.ndarray:
    """Return three shared-shift kernels acting on both factors."""

    return np.stack(
        [
            np.kron(
                shifted_transition(shift, base=base),
                shifted_transition(shift, base=base),
            )
            for shift in range(3)
        ]
    )


def e2_kernels(
    coupling_lambda: float,
    *,
    base: np.ndarray = BASE_TRANSITION,
    attract: float = E2_ATTRACT,
    repel: float = E2_REPEL,
) -> np.ndarray:
    """Return E2 kernels with current-F1 dependence confined within N."""

    f2_actions = e2_action_transitions(
        base=base,
        attract=attract,
        repel=repel,
    )
    kernels = []
    for action_transition in f2_actions:
        conditional = np.stack(
            [
                modulate_within_non_goal(
                    action_transition,
                    context_state=context,
                    coupling_lambda=coupling_lambda,
                )
                for context in range(N_FACTOR_STATES)
            ]
        )
        kernels.append(_joint_from_source_conditioned_f2(base, conditional))
    return np.stack(kernels)


def gauge_kernels(
    *,
    base: np.ndarray = BASE_TRANSITION,
) -> np.ndarray:
    """Return E4 kernels where current F1 rotates physical F2 action meaning."""

    kernels = []
    for action in range(N_FACTOR_STATES):
        conditional = np.stack(
            [
                shifted_transition((action + context) % 3, base=base)
                for context in range(N_FACTOR_STATES)
            ]
        )
        kernels.append(_joint_from_source_conditioned_f2(base, conditional))
    return np.stack(kernels)


def action_kernels(
    action_kind: str,
    *,
    coupling_lambda: float = 0.0,
    base: np.ndarray = BASE_TRANSITION,
) -> np.ndarray:
    """Build the action-conditioned joint kernels for one condition."""

    builders = {
        "product": lambda: product_shift_kernels(base=base),
        "diagonal": lambda: diagonal_shift_kernels(base=base),
        "e2_tilt": lambda: e2_kernels(coupling_lambda, base=base),
        "e4_gauge": lambda: gauge_kernels(base=base),
    }
    try:
        kernels = builders[action_kind]()
    except KeyError as error:
        raise ValueError(f"unknown action kind {action_kind!r}") from error
    if not np.allclose(kernels.sum(axis=-1), 1.0, atol=1e-12):
        raise AssertionError("constructed action kernels are not stochastic")
    return kernels


def split_joint_index(index: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(index, dtype=np.int64)
    return values // N_FACTOR_STATES, values % N_FACTOR_STATES


def reward_vector(reward_kind: str) -> np.ndarray:
    """Return current-state rewards for a named design condition."""

    states = np.arange(N_JOINT_STATES)
    first, second = split_joint_index(states)
    rewards = {
        "f1_goal": (first == GOAL_STATE).astype(np.float64),
        "f2_goal": (second == GOAL_STATE).astype(np.float64),
        "additive": (
            (first == GOAL_STATE).astype(np.float64)
            + (second == GOAL_STATE).astype(np.float64)
        ),
        "conjunctive": (
            (first == GOAL_STATE) & (second == GOAL_STATE)
        ).astype(np.float64),
    }
    try:
        return rewards[reward_kind]
    except KeyError as error:
        raise ValueError(f"unknown reward kind {reward_kind!r}") from error


@dataclass(frozen=True, slots=True)
class ValueIterationResult:
    value: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray
    iterations: int
    residual: float


def value_iteration(
    kernels: np.ndarray,
    rewards: Sequence[float],
    *,
    gamma: float = 0.99,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
) -> ValueIterationResult:
    """Solve the fully observed discounted MDP with lowest-index tie breaks."""

    transitions = np.asarray(kernels, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    if transitions.ndim != 3 or transitions.shape[1:] != (len(reward), len(reward)):
        raise ValueError("kernels and rewards have incompatible shapes")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    value = np.zeros(len(reward), dtype=np.float64)
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        q_values = reward[:, None] + gamma * np.einsum(
            "asj,j->sa",
            transitions,
            value,
        )
        updated = q_values.max(axis=1)
        residual = float(np.max(np.abs(updated - value)))
        value = updated
        if residual <= tolerance:
            break
    else:
        raise RuntimeError("value iteration did not converge")
    q_values = reward[:, None] + gamma * np.einsum(
        "asj,j->sa",
        transitions,
        value,
    )
    return ValueIterationResult(
        value=value,
        q_values=q_values,
        policy=q_values.argmax(axis=1),
        iterations=iteration,
        residual=residual,
    )


def additive_matrix_residual(values: np.ndarray) -> float:
    """Return relative residual after fitting ``constant + f(s1) + g(s2)``."""

    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    fitted = (
        matrix.mean(axis=1, keepdims=True)
        + matrix.mean(axis=0, keepdims=True)
        - matrix.mean()
    )
    centered = matrix - matrix.mean()
    denominator = float(np.square(centered).sum())
    if denominator == 0.0:
        return 0.0
    return float(np.square(matrix - fitted).sum() / denominator)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Return a normalized stationary row distribution."""

    matrix = _stochastic(transition, name="transition")
    system = np.vstack([matrix.T - np.eye(len(matrix)), np.ones(len(matrix))])
    target = np.concatenate([np.zeros(len(matrix)), np.ones(1)])
    solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    solution = np.clip(solution, 0.0, None)
    return solution / solution.sum()
