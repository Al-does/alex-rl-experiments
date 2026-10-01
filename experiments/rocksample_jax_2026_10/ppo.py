"""End-to-end jitted PPO for RockSample (PureJaxRL style, no RLlib/Ray).

Mirrors the ``rocksample_ppo_2026_09`` RLlib recipe: clipped surrogate,
unclipped value loss, entropy bonus, RLlib's adaptive KL penalty, GAE with
batch-standardised advantages, truncation bootstrapped from the final
observation's value, global-norm gradient clipping and Adam.

``init`` and ``train_chunk`` are pure functions of a PRNG key / runner state,
so a population of independent runs is ``jax.vmap`` over keys.

The policy objective is pluggable: an ``Objective`` maps
``(meta, ratio, advantages)`` to per-sample surrogates and an entropy
coefficient. ``meta`` lives in ``RunnerState``, so runs with different
objective parameters can also be vmapped together.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 64
    num_steps: int = 128
    num_minibatches: int = 8
    num_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    vf_coeff: float = 0.5
    vf_clip: float = 1_000_000.0
    entropy_coeff: float = 0.05
    max_grad_norm: float = 0.5
    kl_coeff: float = 0.2
    kl_target: float = 0.01
    use_kl_loss: bool = True

    def __post_init__(self) -> None:
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps


class Rollout(NamedTuple):
    env_state: rs.EnvState
    obs: jax.Array  # (N, obs_dim)
    cache: tm.KVCache
    episode_id: jax.Array  # (N,)
    episode_return: jax.Array  # (N,)
    history_obs: jax.Array  # (N, lookback, obs_dim)
    history_episode: jax.Array  # (N, lookback), -1 = padding


class RunnerState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    kl_coeff: jax.Array
    rollout: Rollout
    key: jax.Array
    env_steps: jax.Array
    meta: jax.Array


Objective = Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


def clipped_objective(config: PPOConfig) -> Objective:
    """PPO's clipped surrogate with ``config.entropy_coeff``; ignores ``meta``."""

    def objective(meta, ratio, advantages):
        eps = config.clip_param
        surrogate = jnp.minimum(
            ratio * advantages, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
        )
        return surrogate, jnp.float32(config.entropy_coeff)

    return objective


class Transition(NamedTuple):
    obs: jax.Array
    episode_id: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    bootstrap_value: jax.Array
    episode_return: jax.Array  # return of episodes ending at this step
    illegal: jax.Array


def make_spec(params: rs.RockSampleParams, d_model: int = 64) -> tm.TransformerSpec:
    return tm.TransformerSpec(
        obs_dim=params.obs_dim,
        num_actions=params.num_actions,
        d_model=d_model,
        n_layers=3,
        n_heads=4,
        context_len=32,
    )


def _optimizer(config: PPOConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr, eps=1e-8),
    )


def init(
    config: PPOConfig,
    env_params: rs.RockSampleParams,
    spec: tm.TransformerSpec,
    key: jax.Array,
    meta: jax.Array | None = None,
) -> RunnerState:
    key, param_key, env_key = jax.random.split(key, 3)
    params = tm.init_params(spec, param_key)
    obs, env_state = jax.vmap(partial(rs.reset, env_params))(
        jax.random.split(env_key, config.num_envs)
    )
    n = config.num_envs
    cache = jax.vmap(lambda _: tm.empty_cache(spec))(jnp.arange(n))
    rollout = Rollout(
        env_state=env_state,
        obs=obs,
        cache=cache,
        episode_id=jnp.zeros(n, jnp.int32),
        episode_return=jnp.zeros(n),
        history_obs=jnp.zeros((n, spec.lookback, spec.obs_dim)),
        history_episode=jnp.full((n, spec.lookback), -1, jnp.int32),
    )
    return RunnerState(
        params=params,
        opt_state=_optimizer(config).init(params),
        kl_coeff=jnp.float32(config.kl_coeff),
        rollout=rollout,
        key=key,
        env_steps=jnp.int32(0),
        meta=jnp.zeros(0) if meta is None else jnp.asarray(meta, jnp.float32),
    )


def _policy_step(spec, params, cache, obs):
    embedding, cache = jax.vmap(partial(tm.encode_step, spec, params))(cache, obs)
    logits, value = tm.heads(params, embedding)
    return logits, value, cache


def collect(
    config: PPOConfig,
    env_params: rs.RockSampleParams,
    spec: tm.TransformerSpec,
    params: dict,
    rollout: Rollout,
    key: jax.Array,
) -> tuple[Rollout, Transition, jax.Array]:
    """Run ``num_steps`` vectorised env steps; returns the bootstrap value too."""

    empty = jax.vmap(lambda _: tm.empty_cache(spec))(jnp.arange(config.num_envs))

    def body(carry: Rollout, step_key):
        action_key, env_key = jax.random.split(step_key)
        logits, value, cache = _policy_step(spec, params, carry.cache, carry.obs)
        action = jax.random.categorical(action_key, logits)
        log_prob = jnp.take_along_axis(
            jax.nn.log_softmax(logits), action[:, None], axis=1
        )[:, 0]
        out = jax.vmap(partial(rs.step_autoreset, env_params))(
            jax.random.split(env_key, config.num_envs), carry.env_state, action
        )
        bootstrap = jax.lax.cond(
            jnp.any(out.truncated),
            lambda: _policy_step(spec, params, cache, out.final_obs)[1],
            lambda: jnp.zeros(config.num_envs),
        )
        done = out.terminated | out.truncated
        total = carry.episode_return + out.reward
        transition = Transition(
            obs=carry.obs,
            episode_id=carry.episode_id,
            action=action,
            log_prob=log_prob,
            value=value,
            reward=out.reward,
            terminated=out.terminated,
            truncated=out.truncated,
            bootstrap_value=bootstrap,
            episode_return=jnp.where(done, total, jnp.nan),
            illegal=out.illegal,
        )
        cache = jax.tree.map(
            lambda fresh, cont: jnp.where(
                done.reshape((-1,) + (1,) * (cont.ndim - 1)), fresh, cont
            ),
            empty,
            cache,
        )
        carry = carry._replace(
            env_state=out.state,
            obs=out.obs,
            cache=cache,
            episode_id=carry.episode_id + done.astype(jnp.int32),
            episode_return=jnp.where(done, 0.0, total),
        )
        return carry, transition

    final, traj = jax.lax.scan(body, rollout, jax.random.split(key, config.num_steps))
    _, last_value, _ = _policy_step(spec, params, final.cache, final.obs)
    traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)  # (N, T, ...)
    return final, traj, last_value


def gae(
    config: PPOConfig, traj: Transition, last_value: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Advantages and value targets, time-major over (N, T)."""

    next_value = jnp.concatenate([traj.value[:, 1:], last_value[:, None]], axis=1)
    next_value = jnp.where(traj.truncated, traj.bootstrap_value, next_value)
    next_value = jnp.where(traj.terminated, 0.0, next_value)
    delta = traj.reward + config.gamma * next_value - traj.value
    keep = 1.0 - (traj.terminated | traj.truncated)

    def body(running, inputs):
        d, k = inputs
        running = d + config.gamma * config.gae_lambda * k * running
        return running, running

    _, advantages = jax.lax.scan(
        body,
        jnp.zeros(delta.shape[0]),
        (delta.T, keep.T),
        reverse=True,
    )
    advantages = advantages.T
    return advantages, advantages + traj.value


def _loss(config, spec, objective, params, kl_coeff, meta, batch):
    seq_obs, seq_episode, action, old_log_prob, old_logits, advantages, targets = batch
    lookback = spec.lookback
    embedding = jax.vmap(partial(tm.encode_window, spec, params))(seq_obs, seq_episode)
    logits, values = tm.heads(params, embedding[:, lookback:])
    log_probs = jax.nn.log_softmax(logits)
    log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]
    ratio = jnp.exp(log_prob - old_log_prob)
    surrogate, entropy_coeff = objective(meta, ratio, advantages)
    policy_loss = -surrogate.mean()
    vf_loss = jnp.minimum((values - targets) ** 2, config.vf_clip).mean()
    probs = jnp.exp(log_probs)
    entropy = -(probs * log_probs).sum(-1).mean()
    old_log_probs = jax.nn.log_softmax(old_logits)
    kl = (jnp.exp(old_log_probs) * (old_log_probs - log_probs)).sum(-1).mean()
    total = policy_loss + config.vf_coeff * vf_loss - entropy_coeff * entropy
    if config.use_kl_loss:
        total = total + kl_coeff * kl
    return total, {
        "policy_loss": policy_loss,
        "vf_loss": vf_loss,
        "entropy": entropy,
        "kl": kl,
        "entropy_coeff": entropy_coeff,
    }


def make_update(
    config: PPOConfig,
    env_params: rs.RockSampleParams,
    spec: tm.TransformerSpec,
    objective: Objective | None = None,
):
    optimizer = _optimizer(config)
    objective = objective or clipped_objective(config)
    loss_and_grad = jax.value_and_grad(
        partial(_loss, config, spec, objective), has_aux=True
    )

    def update(state: RunnerState, _=None) -> tuple[RunnerState, dict]:
        key, collect_key, shuffle_key = jax.random.split(state.key, 3)
        prev = state.rollout
        rollout, traj, last_value = collect(
            config, env_params, spec, state.params, prev, collect_key
        )
        advantages, targets = gae(config, traj, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        seq_obs = jnp.concatenate([prev.history_obs, traj.obs], axis=1)
        seq_episode = jnp.concatenate([prev.history_episode, traj.episode_id], axis=1)
        old_logits, _ = tm.heads(
            state.params,
            jax.vmap(partial(tm.encode_window, spec, state.params))(
                seq_obs, seq_episode
            )[:, spec.lookback :],
        )
        batch = (
            seq_obs,
            seq_episode,
            traj.action,
            traj.log_prob,
            old_logits,
            advantages,
            targets,
        )

        def epoch(carry, epoch_key):
            params, opt_state = carry
            order = jax.random.permutation(epoch_key, config.num_envs)
            minibatches = jax.tree.map(
                lambda x: x[order].reshape((config.num_minibatches, -1) + x.shape[1:]),
                batch,
            )

            def minibatch(carry, mb):
                params, opt_state = carry
                (_, aux), grads = loss_and_grad(params, state.kl_coeff, state.meta, mb)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            return jax.lax.scan(minibatch, (params, opt_state), minibatches)

        (params, opt_state), aux = jax.lax.scan(
            epoch,
            (state.params, state.opt_state),
            jax.random.split(shuffle_key, config.num_epochs),
        )
        aux = jax.tree.map(jnp.mean, aux)
        kl_coeff = jnp.where(
            aux["kl"] > 2.0 * config.kl_target,
            state.kl_coeff * 1.5,
            jnp.where(
                aux["kl"] < 0.5 * config.kl_target,
                state.kl_coeff * 0.5,
                state.kl_coeff,
            ),
        )
        lookback = spec.lookback
        rollout = rollout._replace(
            history_obs=seq_obs[:, -lookback:],
            history_episode=seq_episode[:, -lookback:],
        )
        finished = ~jnp.isnan(traj.episode_return)
        episodes = finished.sum()
        env_steps = state.env_steps + config.batch_size
        metrics = {
            **aux,
            "env_steps": env_steps,
            "episodes": episodes,
            "episode_return_mean": jnp.where(
                episodes > 0,
                jnp.nansum(traj.episode_return) / jnp.maximum(episodes, 1),
                jnp.nan,
            ),
            "illegal_action_rate": traj.illegal.mean(),
            "kl_coeff": kl_coeff,
        }
        return (
            RunnerState(
                params, opt_state, kl_coeff, rollout, key, env_steps, state.meta
            ),
            metrics,
        )

    return update


def make_train_chunk(
    config: PPOConfig,
    env_params: rs.RockSampleParams,
    spec: tm.TransformerSpec,
    num_updates: int,
    objective: Objective | None = None,
):
    """Jittable ``state -> (state, metrics[num_updates])``."""

    update = make_update(config, env_params, spec, objective)

    def chunk(state: RunnerState):
        return jax.lax.scan(update, state, None, length=num_updates)

    return chunk
