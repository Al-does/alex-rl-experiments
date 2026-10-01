"""Pure-JAX RockSample with the same dynamics and observation as ``envs.rocksample``.

Only the fixed-layout training setting (``randomize_train_layout=False``) is
implemented. All functions are jittable and vmappable over environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from envs.rocksample.model import (
    BAD_ROCK_REWARD,
    EXIT_REWARD,
    GOOD_ROCK_REWARD,
    Action,
    Observation,
    instance_definition,
)

NUM_BASE_ACTIONS = len(Action)
NUM_SYMBOLS = len(Observation)


@dataclass(frozen=True)
class RockSampleParams:
    """Static instance definition; hashable so it can close over jitted code."""

    n: int
    k: int
    start: tuple[int, int]
    rocks: tuple[tuple[int, int], ...]
    half_efficiency_distance: float
    episode_length: int = 100

    @classmethod
    def from_instance(
        cls,
        n: int = 5,
        k: int = 7,
        *,
        episode_length: int = 100,
        layout_seed: int = 0,
    ) -> RockSampleParams:
        instance = instance_definition(n, k, layout_seed=layout_seed)
        return cls(
            n=n,
            k=k,
            start=instance.start,
            rocks=instance.rocks,
            half_efficiency_distance=instance.sensor_half_efficiency_distance,
            episode_length=episode_length,
        )

    @property
    def num_actions(self) -> int:
        return NUM_BASE_ACTIONS + self.k

    @property
    def obs_dim(self) -> int:
        return 2 + 2 * self.k + NUM_SYMBOLS + self.num_actions


class EnvState(NamedTuple):
    rover: jax.Array  # (2,) int32
    qualities: jax.Array  # (k,) bool
    symbol: jax.Array  # () int32
    previous_action: jax.Array  # () int32, -1 after reset
    step: jax.Array  # () int32


class StepOutput(NamedTuple):
    obs: jax.Array
    state: EnvState
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    illegal: jax.Array


def observation(params: RockSampleParams, state: EnvState) -> jax.Array:
    scale = float(params.n - 1)
    rocks = jnp.asarray(params.rocks, dtype=jnp.float32).reshape(-1) / scale
    action = jax.nn.one_hot(state.previous_action, params.num_actions)
    return jnp.concatenate(
        [
            state.rover.astype(jnp.float32) / scale,
            rocks,
            jax.nn.one_hot(state.symbol, NUM_SYMBOLS),
            action * (state.previous_action >= 0),
        ]
    ).astype(jnp.float32)


def reset(params: RockSampleParams, key: jax.Array) -> tuple[jax.Array, EnvState]:
    state = EnvState(
        rover=jnp.asarray(params.start, dtype=jnp.int32),
        qualities=jax.random.bernoulli(key, 0.5, (params.k,)),
        symbol=jnp.int32(Observation.NONE),
        previous_action=jnp.int32(-1),
        step=jnp.int32(0),
    )
    return observation(params, state), state


def step(
    params: RockSampleParams,
    key: jax.Array,
    state: EnvState,
    action: jax.Array,
) -> StepOutput:
    n = params.n
    rocks = jnp.asarray(params.rocks, dtype=jnp.int32)
    x, y = state.rover[0], state.rover[1]
    action = action.astype(jnp.int32)

    north = action == Action.NORTH
    south = action == Action.SOUTH
    east = action == Action.EAST
    west = action == Action.WEST
    sample = action == Action.SAMPLE
    check = action >= NUM_BASE_ACTIONS

    at_rock = jnp.all(rocks == state.rover[None, :], axis=1)
    on_rock = jnp.any(at_rock)
    rock_here = jnp.argmax(at_rock)
    good_here = state.qualities[rock_here]

    exits = east & (x == n - 1)
    illegal = (
        (north & (y == n - 1))
        | (south & (y == 0))
        | (west & (x == 0))
        | (sample & ~on_rock)
    )
    dx = jnp.where(east & ~exits, 1, 0) - jnp.where(west & (x > 0), 1, 0)
    dy = jnp.where(north & (y < n - 1), 1, 0) - jnp.where(south & (y > 0), 1, 0)
    rover = jnp.stack([x + dx, y + dy])

    sampled = sample & on_rock
    reward = jnp.where(exits, EXIT_REWARD, 0.0) + jnp.where(
        sampled,
        jnp.where(good_here, GOOD_ROCK_REWARD, BAD_ROCK_REWARD),
        0.0,
    )
    qualities = jnp.where(
        sampled & (jnp.arange(params.k) == rock_here),
        False,
        state.qualities,
    )

    checked = jnp.clip(action - NUM_BASE_ACTIONS, 0, params.k - 1)
    offset = (state.rover - rocks[checked]).astype(jnp.float32)
    distance = jnp.sqrt(jnp.sum(offset**2))
    efficiency = 2.0 ** (-distance / params.half_efficiency_distance)
    correct = jax.random.uniform(key) < (1.0 + efficiency) / 2.0
    reports_good = state.qualities[checked] == correct
    symbol = jnp.where(
        check,
        jnp.where(reports_good, Observation.GOOD, Observation.BAD),
        Observation.NONE,
    ).astype(jnp.int32)

    count = state.step + 1
    terminated = exits
    truncated = (count >= params.episode_length) & ~terminated
    next_state = EnvState(
        rover=rover.astype(jnp.int32),
        qualities=qualities,
        symbol=symbol,
        previous_action=action,
        step=count,
    )
    return StepOutput(
        observation(params, next_state),
        next_state,
        reward.astype(jnp.float32),
        terminated,
        truncated,
        illegal,
    )


class AutoResetOutput(NamedTuple):
    obs: jax.Array  # next policy input (reset obs when the episode ended)
    final_obs: jax.Array  # pre-reset observation, for truncation bootstrapping
    state: EnvState
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    illegal: jax.Array


def step_autoreset(
    params: RockSampleParams,
    key: jax.Array,
    state: EnvState,
    action: jax.Array,
) -> AutoResetOutput:
    step_key, reset_key = jax.random.split(key)
    out = step(params, step_key, state, action)
    reset_obs, reset_state = reset(params, reset_key)
    done = out.terminated | out.truncated
    next_state = jax.tree.map(
        lambda fresh, cont: jnp.where(done, fresh, cont), reset_state, out.state
    )
    return AutoResetOutput(
        obs=jnp.where(done, reset_obs, out.obs),
        final_obs=out.obs,
        state=next_state,
        reward=out.reward,
        terminated=out.terminated,
        truncated=out.truncated,
        illegal=out.illegal,
    )
