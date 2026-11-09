"""Two independent controlled MESS3 factors as one nine-state JAX process.

Joint state ``3 * s1 + s2`` and joint token ``3 * x1 + x2`` follow
``np.kron`` ordering. Each joint action applies an independent variant-3
tilt to both factors.
"""

from __future__ import annotations

import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.shared import (
    action_transitions,
    emission_matrix,
)

CONDITIONS = ("reward_both", "reward_factor_1")
NOOP, POSITIVE, NEGATIVE = 0, 1, 2
ACTION_PAIRS = (
    (NOOP, NOOP), (POSITIVE, POSITIVE), (POSITIVE, NEGATIVE),
    (NOOP, POSITIVE), (NOOP, NEGATIVE), (NEGATIVE, POSITIVE),
    (NEGATIVE, NEGATIVE), (POSITIVE, NOOP), (NEGATIVE, NOOP),
)


def factor_process(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    factor = action_transitions(3)
    transitions = np.stack([np.kron(factor[a], factor[b]) for a, b in ACTION_PAIRS])
    emission = np.kron(emission_matrix(), emission_matrix())
    first, second = np.divmod(np.arange(9), 3)
    components = np.stack([(first == 2), (second == 2)]).astype(float)
    reward = components[0] + (components[1] if condition == "reward_both" else 0.0)
    return make_process(np.full(9, 1 / 9), transitions, emission, reward, components)


def factor_targets(belief):
    joint = belief.reshape(-1, 3, 3)
    return {"joint": belief, "factor_1": joint.sum(2), "factor_2": joint.sum(1)}
