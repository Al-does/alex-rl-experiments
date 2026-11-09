"""Pure-JAX action-controlled HMM with zero-delay token emission.

Semantics match ``envs.hmm.HMMEnv`` with ``delay=0``: reset samples a state and
its token; a step scores the pre-transition state, samples the next state from
the executed action's transition law, then emits that state's token. The
observation is ``[one_hot(current token), one_hot(previous action)]`` with an
all-zero action block at reset.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ControlledHMM(NamedTuple):
    initial: jax.Array  # (S,)
    transitions: jax.Array  # (A, S, S)
    emission: jax.Array  # (S, K)
    reward: jax.Array  # (S,)
    reward_components: jax.Array  # (C, S)

    @property
    def n_tokens(self) -> int:
        return int(self.emission.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def obs_dim(self) -> int:
        return self.n_tokens + self.n_actions


class EnvState(NamedTuple):
    state: jax.Array
    token: jax.Array
    prev_action: jax.Array


def make_process(initial, transitions, emission, reward, reward_components):
    arrays = [np.asarray(value, dtype=np.float64) for value in (
        initial, transitions, emission, reward, reward_components
    )]
    initial, transitions, emission, reward, components = arrays
    if not np.allclose(transitions.sum(-1), 1.0) or not np.allclose(emission.sum(-1), 1.0):
        raise ValueError("transition and emission rows must be distributions")
    if not np.isclose(initial.sum(), 1.0):
        raise ValueError("initial distribution must sum to one")
    return ControlledHMM(*(jnp.asarray(value, dtype=jnp.float32) for value in arrays))


def _categorical(key, probabilities):
    return jax.random.categorical(key, jnp.log(probabilities), axis=-1)


def reset(process: ControlledHMM, key: jax.Array) -> EnvState:
    state_key, token_key = jax.random.split(key)
    state = _categorical(state_key, process.initial)
    token = _categorical(token_key, process.emission[state])
    return EnvState(state, token, jnp.asarray(-1, dtype=jnp.int32))


def observe(process: ControlledHMM, env: EnvState) -> jax.Array:
    return jnp.concatenate([
        jax.nn.one_hot(env.token, process.n_tokens),
        jax.nn.one_hot(env.prev_action, process.n_actions),
    ])


def step(process: ControlledHMM, key: jax.Array, env: EnvState, action: jax.Array):
    """Return ``(next_env, reward, reward_components)`` for one decision."""

    state_key, token_key = jax.random.split(key)
    reward = process.reward[env.state]
    components = process.reward_components[:, env.state]
    state = _categorical(state_key, process.transitions[action, env.state])
    token = _categorical(token_key, process.emission[state])
    return EnvState(state, token, action.astype(jnp.int32)), reward, components


def belief_reset(process: ControlledHMM, token: jax.Array) -> jax.Array:
    belief = process.initial * process.emission[:, token]
    return belief / belief.sum()


def belief_update(process, belief, action, token):
    """Exact filter over the current state given tokens and executed actions."""

    belief = (belief @ process.transitions[action]) * process.emission[:, token]
    return belief / belief.sum()
