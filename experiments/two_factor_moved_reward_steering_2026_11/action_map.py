"""Myopic best-action maps: maximize expected reward at the next time step.

Reward at step t is occupancy of the pre-transition state S_t, so the action
chosen at t only affects r_{t+1} = 1[S_{t+1} = reward state].  Given the
filtered belief b_t over S_t, the expected next-step factor reward is
``b_t @ T_a[:, reward_state]``.  Factors are independent and rewards additive,
so the joint map is the pair of per-factor maps.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from experiments.mess3_reward_state_action_symmetry_cycle_5.task import (
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import (
    ACTION_PAIRS,
    factor_transition,
)


FACTOR_ACTIONS = (NOOP_ACTION, POSITIVE_ACTION, NEGATIVE_ACTION)
FACTOR_ACTION_NAMES = {
    NOOP_ACTION: "Noop", POSITIVE_ACTION: "Tilt X", NEGATIVE_ACTION: "Tilt Y",
}
PAIR_TO_ACTION = {pair: index for index, pair in enumerate(ACTION_PAIRS)}
# Full-State Quotient map of the source task (reward state 2): the trained
# Joint Reward policy takes Tilt X in state 0, Tilt Y in state 1, Noop in 2.
SOURCE_ACTION_STATE = {POSITIVE_ACTION: 0, NEGATIVE_ACTION: 1, NOOP_ACTION: 2}


def next_reward_table(reward_state: int) -> np.ndarray:
    """``table[i, a] = P(S_{t+1} = reward_state | S_t = i, action a)``."""
    return np.column_stack(
        [factor_transition(action)[:, reward_state] for action in FACTOR_ACTIONS]
    )


def expected_next_reward(belief: np.ndarray, reward_state: int) -> np.ndarray:
    """Expected next-step reward for each factor action (columns follow FACTOR_ACTIONS)."""
    belief = np.asarray(belief, dtype=np.float64)
    return belief @ next_reward_table(reward_state)


def best_factor_action(belief: np.ndarray, reward_state: int) -> np.ndarray:
    """Myopic best factor action; exact ties resolve to the lower action index."""
    values = expected_next_reward(belief, reward_state)
    return np.asarray(FACTOR_ACTIONS)[np.argmax(values, axis=-1)]


def best_joint_action(
    factor_beliefs: np.ndarray, reward_states: Sequence[int]
) -> np.ndarray:
    """Joint action index from ``factor_beliefs[..., factor, state]``."""
    first = best_factor_action(factor_beliefs[..., 0, :], reward_states[0])
    second = best_factor_action(factor_beliefs[..., 1, :], reward_states[1])
    return np.vectorize(lambda a, b: PAIR_TO_ACTION[(int(a), int(b))])(first, second)
