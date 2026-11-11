"""Persistent-nuisance v2 factor design (zero-based states)."""
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base

BASELINE_TRANSITION_MATRIX = np.array([
    [.995505, .002495, .002], [.002495, .995505, .002], [.005, .005, .99],
])
EMISSION_MATRIX = np.array([[.1125, .0375, .85], [.0375, .1125, .85], [0., 0., 1.]])
EFFECT_SIZE = 2.5
REWARD_STATE = 2
INITIAL_STATE_DISTRIBUTION = np.full(3, 1 / 3)
DIRECTIONS = base.DIRECTIONS


def emission_matrix():
    return EMISSION_MATRIX.copy()


def action_transitions(variant):
    if variant not in DIRECTIONS:
        raise ValueError("variant must be 1, 2, or 3")
    return np.stack([BASELINE_TRANSITION_MATRIX] + [
        base.tilt_reward_state(BASELINE_TRANSITION_MATRIX, DIRECTIONS[variant][:, action],
                               effect_size=EFFECT_SIZE)
        for action in (1, 2)
    ])


def ladder_process(variant):
    reward = (np.arange(3) == REWARD_STATE).astype(float)
    return make_process(INITIAL_STATE_DISTRIBUTION, action_transitions(variant),
                        emission_matrix(), reward, reward[None])


def environment_metadata():
    return {
        **base.environment_metadata(),
        "environment_revision": "persistent_nuisance_v2",
        "baseline_transition_matrix": BASELINE_TRANSITION_MATRIX.tolist(),
        "emission_matrix": EMISSION_MATRIX.tolist(),
        "effect_size": EFFECT_SIZE,
        "initial_state_distribution": INITIAL_STATE_DISTRIBUTION.tolist(),
        "reward_state": REWARD_STATE,
    }
