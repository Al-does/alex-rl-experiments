"""The composed ``(m, phi)`` generator as a plain finite HMM."""

from __future__ import annotations

import numpy as np

from envs.hmm import HMMModel, stationary_distribution
from envs.mess3.model import (
    N_STATES,
    PASSIVE_TRANSITION_MATRIX,
    emission_matrix,
)
from experiments.mess3_feedback_factoring_cycle_1.dynamics import (
    joint_emission,
    joint_initial_distribution,
    joint_transition,
)


STATE_LABELS = tuple(
    f"m{chain}_phi{register}"
    for chain in range(N_STATES)
    for register in range(N_STATES)
)
TOKEN_LABELS = tuple(
    f"x{composite}_rho{report}"
    for composite in range(N_STATES)
    for report in range(N_STATES)
)


def composed_model(
    *,
    alpha: float = 0.85,
    feedback_strength: float = 0.0,
    register_noise: float = 1.0,
) -> HMMModel:
    """Build the nine-state passive-chain-times-register generator.

    ``transition_matrix`` is the register-inert kernel ``kron(T, I)`` executed
    by guess zero. The task supplies the guess-conditioned kernels at each
    step, so this is the model's reference dynamics rather than its only one.
    """

    chain = np.asarray(PASSIVE_TRANSITION_MATRIX, dtype=np.float64)
    return HMMModel(
        initial_distribution=joint_initial_distribution(
            stationary_distribution(chain)
        ),
        transition_matrix=joint_transition(0, feedback_strength, base=chain),
        emission_matrix=joint_emission(
            emission_matrix(alpha),
            register_noise=register_noise,
        ),
        state_labels=STATE_LABELS,
        token_labels=TOKEN_LABELS,
    )
