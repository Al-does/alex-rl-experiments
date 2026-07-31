"""Action-driven MESS3 dynamics and their two-factor decomposition.

The guessed token feeds back into the hidden dynamics as a cyclic shift of the
MESS3 state. Writing ``C`` for the forward shift permutation on ``Z_3``, one
guess ``a`` executes

    U(a) = T @ R(a),    R(a) = (1 - kappa) I + kappa C^a,

so guess ``0`` leaves the process alone, guess ``1`` rotates it one step, and
guess ``2`` rotates it two steps, each with probability ``kappa``.

Both ``T`` and every ``R(a)`` are circulant, so they commute and the executed
state factors exactly into two parallel parts,

    s_t = m_t + Phi_t  (mod 3),

where ``m`` is an untouched passive MESS3 chain and ``Phi`` is a Z_3 register
driven only by the agent's own guesses. The emitted token factors the same way,
``x_t = u_t + Phi_t``, with ``u`` the passive MESS3 token. This is the
composition studied by Shai et al. (arXiv:2602.02385), with the twist that the
policy - not the generator - drives the second factor.

``kappa`` interpolates between three qualitatively different regimes:

* ``kappa = 0``  - ``Phi`` never moves; the process is passive MESS3.
* ``0 < kappa < 1`` - ``Phi`` is a hidden walk; observing ``x`` only reveals the
  sum ``u + Phi``, so the two factors are correlated and a factored (product)
  belief is lossy.
* ``kappa = 1``  - ``Phi`` is a deterministic function of past guesses, so the
  joint belief is exactly a product state and factoring is lossless.
"""

from __future__ import annotations

import numpy as np

from envs.mess3.model import (
    N_STATES,
    N_TOKENS,
    PASSIVE_TRANSITION_MATRIX,
    emission_matrix,
)


def cyclic_shift_matrix(shift: int, size: int = N_STATES) -> np.ndarray:
    """Return the permutation moving probability mass forward by ``shift``."""

    if size <= 0:
        raise ValueError("size must be positive")
    return np.roll(np.eye(size, dtype=np.float64), int(shift) % size, axis=1)


def circulant(first_row: np.ndarray) -> np.ndarray:
    """Return the circulant matrix ``M[i, j] = first_row[(j - i) % n]``."""

    row = np.asarray(first_row, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError("first_row must be one-dimensional")
    return np.stack([np.roll(row, index) for index in range(len(row))])


def is_circulant(matrix: np.ndarray, *, atol: float = 1e-12) -> bool:
    """Return whether ``matrix`` is circulant to within ``atol``."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return False
    return bool(np.allclose(values, circulant(values[0]), atol=atol))


def require_circulant(matrix: np.ndarray, *, name: str) -> np.ndarray:
    """Validate the Z_3 symmetry the factored decomposition relies on."""

    values = np.asarray(matrix, dtype=np.float64)
    if not is_circulant(values):
        raise ValueError(
            f"{name} must be circulant for the shift factorization to hold"
        )
    return values


def action_shift_distribution(
    action: int,
    strength: float,
    *,
    n_actions: int = N_TOKENS,
) -> np.ndarray:
    """Return the Z_3 shift law applied by one guess."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("feedback strength must lie in [0, 1]")
    index = int(action)
    if not 0 <= index < n_actions:
        raise ValueError(f"action {index} is outside the guess alphabet")
    shifts = np.zeros(n_actions, dtype=np.float64)
    shifts[0] = 1.0 - strength
    shifts[index % n_actions] += strength
    return shifts


def feedback_shift_operator(action: int, strength: float) -> np.ndarray:
    """Return ``R(a) = (1 - kappa) I + kappa C^a`` on the state space."""

    return circulant(action_shift_distribution(action, strength))


def feedback_transition(
    action: int,
    strength: float,
    *,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> np.ndarray:
    """Return the row-stochastic kernel executed by one guess."""

    matrix = require_circulant(base, name="base transition matrix")
    return matrix @ feedback_shift_operator(action, strength)


def feedback_transitions(
    strength: float,
    *,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
    n_actions: int = N_TOKENS,
) -> np.ndarray:
    """Return every guess-conditioned kernel stacked on the leading axis."""

    return np.stack(
        [
            feedback_transition(action, strength, base=base)
            for action in range(n_actions)
        ]
    )


def joint_transition(
    action: int,
    strength: float,
    *,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> np.ndarray:
    """Return the factored kernel on ``(m, Phi)`` indexed as ``m * 3 + Phi``."""

    matrix = require_circulant(base, name="base transition matrix")
    return np.kron(matrix, feedback_shift_operator(action, strength))


def joint_transitions(
    strength: float,
    *,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
    n_actions: int = N_TOKENS,
) -> np.ndarray:
    """Return every guess-conditioned factored kernel."""

    return np.stack(
        [
            joint_transition(action, strength, base=base)
            for action in range(n_actions)
        ]
    )


def joint_emission(emission: np.ndarray) -> np.ndarray:
    """Return ``P(x | m, Phi) = E[m, (x - Phi) % 3]`` on the factored space."""

    likelihood = require_circulant(emission, name="emission matrix")
    n_states, n_tokens = likelihood.shape
    tokens = np.arange(n_tokens)
    by_phase = np.stack(
        [likelihood[:, (tokens - phase) % n_tokens] for phase in range(n_states)]
    )
    return by_phase.transpose(1, 0, 2).reshape(n_states * n_states, n_tokens)


def lumping_matrix(n_states: int = N_STATES) -> np.ndarray:
    """Return the ``(m, Phi) -> (m + Phi) % 3`` state aggregation."""

    factors = np.arange(n_states)
    sums = (factors[:, None] + factors[None, :]) % n_states
    lump = np.zeros((n_states * n_states, n_states), dtype=np.float64)
    lump[np.arange(n_states * n_states), sums.reshape(-1)] = 1.0
    return lump


def joint_initial_distribution(initial: np.ndarray) -> np.ndarray:
    """Start the register at ``Phi = 0`` and the chain at the MESS3 prior."""

    prior = np.asarray(initial, dtype=np.float64)
    register = np.zeros_like(prior)
    register[0] = 1.0
    return np.kron(prior, register)


def factor_marginals(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a joint ``(m, Phi)`` belief into its two factor beliefs."""

    values = np.asarray(joint, dtype=np.float64)
    size = int(round(np.sqrt(values.shape[-1])))
    if size * size != values.shape[-1]:
        raise ValueError("joint beliefs must live on a square factor product")
    grid = values.reshape(*values.shape[:-1], size, size)
    return grid.sum(axis=-1), grid.sum(axis=-2)


def product_state(chain: np.ndarray, register: np.ndarray) -> np.ndarray:
    """Return the factored (product) approximation of a joint belief."""

    left = np.asarray(chain, dtype=np.float64)
    right = np.asarray(register, dtype=np.float64)
    return (left[..., :, None] * right[..., None, :]).reshape(
        *left.shape[:-1],
        left.shape[-1] * right.shape[-1],
    )


def cyclic_convolution(chain: np.ndarray, register: np.ndarray) -> np.ndarray:
    """Return the executed-state belief implied by a product state."""

    return np.asarray(product_state(chain, register)) @ lumping_matrix(
        np.asarray(chain).shape[-1]
    )


def executed_state_belief(joint: np.ndarray) -> np.ndarray:
    """Aggregate a joint ``(m, Phi)`` belief down to the executed state."""

    values = np.asarray(joint, dtype=np.float64)
    size = int(round(np.sqrt(values.shape[-1])))
    return values @ lumping_matrix(size)


def base_model_matrices(
    *,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the validated circulant transition and emission pair."""

    return (
        require_circulant(base, name="base transition matrix"),
        require_circulant(emission_matrix(alpha), name="emission matrix"),
    )
