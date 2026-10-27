"""Pure-JAX Pusher-B b10 guess-or-pass task."""

from __future__ import annotations

import math
from numbers import Real

import jax
import jax.numpy as jnp
from envs.hmm.jax_env import JaxHMMEnv

from experiments.pusher_b_jax_2026_10.process import pusher_b_model

NUM_ACTIONS = 3
EPISODE_LENGTH = 127


def guess_or_pass_reward(cost: float):
    if (
        isinstance(cost, bool)
        or not isinstance(cost, Real)
        or not math.isfinite(cost)
        or cost <= 0
    ):
        raise ValueError("cost must be a finite positive number")
    cost = float(cost)

    def reward(action: jax.Array, token: jax.Array) -> jax.Array:
        return jnp.where(
            action == 2,
            0.0,
            jnp.where(action == token, 1.0, -cost),
        )

    return reward


def make_env(cost: float) -> JaxHMMEnv:
    return JaxHMMEnv.from_model(
        pusher_b_model("b10"),
        episode_length=EPISODE_LENGTH,
        reward_fn=guess_or_pass_reward(cost),
        delay=1,
    )
