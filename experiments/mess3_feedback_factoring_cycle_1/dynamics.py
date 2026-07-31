"""The composed two-factor generator and its guess-driven dynamics.

The hidden state is a pair ``(m, phi)`` indexed as ``m * 3 + phi``:

``m``
    an untouched passive MESS3 chain, evolving under the circulant kernel ``T``;
``phi``
    a ``Z_3`` register that only the agent's own guesses move.

Guess ``a`` executes ``kron(T, R(a))`` with

    R(a) = (1 - kappa) I + kappa C^a,

where ``C`` is the forward cyclic shift. The joint kernel is therefore always an
exact tensor product: guess ``0`` leaves the register alone, guess ``1`` rotates
it one step and guess ``2`` two steps, each firing with probability ``kappa``.

Each factor emits its own sub-token, and the observed token is the pair
``(x, rho)`` drawn from a nine-symbol alphabet indexed ``x * 3 + rho``:

``x = u + phi (mod 3)``
    the composite token the agent is scored on, with ``u`` the passive MESS3
    sub-token;
``rho``
    the register's sub-token, equal to ``phi`` with probability ``1 - epsilon``
    and uniform noise otherwise.

So

    P((x, rho) | m, phi) = E[m, (x - phi) % 3] * ((1 - eps) 1{rho = phi} + eps/3).

``epsilon`` is the knob of Shai et al. (arXiv:2602.02385). At ``eps = 0`` the
token-labelled operator splits as ``A(m) (x) B(phi)``, which is conditional
independence in the sense of their Definition 2.1, so a factored representation
is lossless. At ``eps = 1`` the register sub-token is pure noise, only the sum
``m + phi`` is ever observable, and both factor marginals collapse onto their
uniform priors: a factored representation then carries no predictive
information at all. Intermediate ``epsilon`` mixes a factoring and a
non-factoring operator, exactly as their ``eps T_int + (1 - eps) (x) T_n``.

``kappa`` is a separate axis with no counterpart in that paper: it sets how hard
a guess shoves the process, not how far the belief sits off the product
manifold.
"""

from __future__ import annotations

import numpy as np

from envs.mess3.model import (
    N_STATES,
    N_TOKENS,
    PASSIVE_TRANSITION_MATRIX,
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
    """Return ``R(a) = (1 - kappa) I + kappa C^a`` on the register."""

    return circulant(action_shift_distribution(action, strength))


def composite_transition(
    action: int,
    strength: float,
    *,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> np.ndarray:
    """Return the kernel executed on the composite state ``s = m + phi``."""

    matrix = require_circulant(base, name="base transition matrix")
    return matrix @ feedback_shift_operator(action, strength)


def composite_transitions(
    strength: float,
    *,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
    n_actions: int = N_TOKENS,
) -> np.ndarray:
    """Return every guess-conditioned composite kernel, stacked by guess."""

    return np.stack(
        [
            composite_transition(action, strength, base=base)
            for action in range(n_actions)
        ]
    )


def joint_transition(
    action: int,
    strength: float,
    *,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> np.ndarray:
    """Return the factored kernel on ``(m, phi)``, indexed ``m * 3 + phi``."""

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


def chain_factor(joint: np.ndarray, *, size: int = N_STATES) -> np.ndarray:
    """Recover ``T`` from ``kron(T, I)``, validating the tensor structure."""

    matrix = np.asarray(joint, dtype=np.float64)
    if matrix.shape != (size * size, size * size):
        raise ValueError("a factored kernel must act on the state product")
    corners = np.arange(size) * size
    extracted = matrix[np.ix_(corners, corners)]
    if not np.allclose(matrix, np.kron(extracted, np.eye(size)), atol=1e-12):
        raise ValueError("the model kernel must be the register-inert product")
    return extracted


def register_channel(noise: float, *, size: int = N_STATES) -> np.ndarray:
    """Return ``P(rho | phi)``: the register report, corrupted with ``eps``."""

    if not 0.0 <= float(noise) <= 1.0:
        raise ValueError("register noise must lie in [0, 1]")
    return (1.0 - noise) * np.eye(size) + noise / size


def joint_emission(
    emission: np.ndarray,
    *,
    register_noise: float,
) -> np.ndarray:
    """Return ``P((x, rho) | m, phi)`` on the nine-symbol paired alphabet."""

    likelihood = require_circulant(emission, name="emission matrix")
    n_states, n_tokens = likelihood.shape
    tokens = np.arange(n_tokens)
    report = register_channel(register_noise, size=n_states)
    # composite[m, phi, x] = P(x | m, phi); report[phi, rho] = P(rho | phi).
    composite = np.stack(
        [likelihood[:, (tokens - phase) % n_tokens] for phase in range(n_states)]
    ).transpose(1, 0, 2)
    paired = composite[:, :, :, None] * report[None, :, None, :]
    return paired.reshape(n_states * n_states, n_tokens * n_states)


def composite_likelihood(
    emission: np.ndarray,
    *,
    register_noise: float = 1.0,
) -> np.ndarray:
    """Return ``P(x | m, phi)`` after marginalizing the register sub-token."""

    n_states = np.asarray(emission).shape[0]
    paired = joint_emission(emission, register_noise=register_noise)
    return paired.reshape(n_states * n_states, -1, n_states).sum(axis=2)


def composite_token(token, *, size: int = N_STATES):
    """Return the scored sub-token ``x`` from a paired-alphabet index."""

    return token // size


def register_token(token, *, size: int = N_STATES):
    """Return the register sub-token ``rho`` from a paired-alphabet index."""

    return token % size


def lumping_matrix(n_states: int = N_STATES) -> np.ndarray:
    """Return the ``(m, phi) -> (m + phi) % 3`` state aggregation."""

    factors = np.arange(n_states)
    sums = (factors[:, None] + factors[None, :]) % n_states
    lump = np.zeros((n_states * n_states, n_states), dtype=np.float64)
    lump[np.arange(n_states * n_states), sums.reshape(-1)] = 1.0
    return lump


def joint_initial_distribution(initial: np.ndarray) -> np.ndarray:
    """Start the register at ``phi = 0`` and the chain at the MESS3 prior."""

    prior = np.asarray(initial, dtype=np.float64)
    register = np.zeros_like(prior)
    register[0] = 1.0
    return np.kron(prior, register)


def factor_marginals(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a joint ``(m, phi)`` belief into its two factor beliefs."""

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


def composite_state_belief(joint: np.ndarray) -> np.ndarray:
    """Aggregate a joint ``(m, phi)`` belief down to ``s = m + phi``."""

    values = np.asarray(joint, dtype=np.float64)
    size = int(round(np.sqrt(values.shape[-1])))
    return values @ lumping_matrix(size)
