"""Baseline-free Monte Carlo REINFORCE in the RLlib PPO shell, fully in JAX.

This reproduces the history studies' PPO configuration (``use_critic=False``,
``lambda_=1``, one epoch, one full-batch minibatch): advantages are discounted
episode reward-to-go, standardized over the whole batch with RLlib's
``max(1e-4, std)`` floor, and the clipped PPO surrogate reduces to REINFORCE
because the importance ratio is exactly one. Every iteration samples
``num_envs`` synchronized complete episodes, then takes one Adam step. Rollout,
returns, loss and update run inside a single jitted function.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.belief_quotient_history_jax_2026_11 import env as hmm
from experiments.belief_quotient_history_jax_2026_11 import models


@dataclass(frozen=True)
class TrainConfig:
    total_env_steps: int
    num_envs: int = 352
    episode_length: int = 1024
    lr: float = 4.2e-4
    gamma: float = 0.99
    clip_param: float = 0.2
    num_epochs: int = 1
    num_minibatches: int = 1
    entropy_schedule: tuple[tuple[int, float], ...] = ((0, 0.0),)

    @property
    def batch_steps(self) -> int:
        return self.num_envs * self.episode_length

    @property
    def iterations(self) -> int:
        return -(-self.total_env_steps // self.batch_steps)


def entropy_coefficient(schedule, steps):
    """RLlib piecewise-linear schedule, held constant after the last point."""

    points = np.asarray(schedule, dtype=np.float64)
    return float(np.interp(steps, points[:, 0], points[:, 1]))


def rollout(spec, process, params, key, num_envs, episode_length, *, greedy=False,
            track_belief=False):
    """Sample ``num_envs`` synchronized episodes with ``lax.scan``."""

    reset_key, key = jax.random.split(key)
    env_state = jax.vmap(hmm.reset, (None, 0))(process, jax.random.split(reset_key, num_envs))
    belief = jax.vmap(hmm.belief_reset, (None, 0))(process, env_state.token)
    carry = models.init_carry(spec, num_envs)
    chunk = spec.chunk_len if spec.kind == "gru" else episode_length
    if episode_length % chunk:
        raise ValueError("episode length must be a multiple of the GRU chunk length")

    def decision(state, t):
        env_state, carry, belief, key = state
        key, action_key, step_key = jax.random.split(key, 3)
        obs = jax.vmap(hmm.observe, (None, 0))(process, env_state)
        logits, residual, embedding, carry = models.step(spec, params, carry, obs, t)
        if greedy:
            action = jnp.argmax(logits, -1)
        else:
            action = jax.random.categorical(action_key, logits, -1)
        log_probs = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(log_probs, action[:, None], -1)[:, 0]
        next_env, reward, components = jax.vmap(hmm.step, (None, 0, 0, 0))(
            process, jax.random.split(step_key, num_envs), env_state, action
        )
        record = {"obs": obs, "action": action, "logp": logp, "reward": reward,
                  "components": components, "entropy": -(jnp.exp(log_probs) * log_probs).sum(-1)}
        if track_belief:
            record.update(belief=belief, residual=residual, embedding=embedding,
                          state=env_state.state)
            belief = jax.vmap(hmm.belief_update, (None, 0, 0, 0))(
                process, belief, action, next_env.token
            )
        return (next_env, carry, belief, key), record

    def chunk_body(state, chunk_index):
        start_carry = state[1]
        times = chunk_index * chunk + jnp.arange(chunk)
        state, records = jax.lax.scan(decision, state, times)
        return state, (records, start_carry if spec.kind == "gru" else ())

    _, (records, starts) = jax.lax.scan(
        chunk_body, (env_state, carry, belief, key), jnp.arange(episode_length // chunk)
    )
    # records: (n_chunks, chunk, B, ...) -> (B, T, ...)
    records = jax.tree_util.tree_map(
        lambda a: jnp.moveaxis(a.reshape(episode_length, *a.shape[2:]), 0, 1), records
    )
    if spec.kind == "gru":
        records["chunk_starts"] = jnp.transpose(starts, (2, 1, 0, 3))  # (B, L, C, H)
    return records


def discounted_returns(rewards, gamma):
    def body(future, reward):
        value = reward + gamma * future
        return value, value

    _, returns = jax.lax.scan(body, jnp.zeros(rewards.shape[0]), rewards.T, reverse=True)
    return returns.T


def standardized_advantages(rewards, gamma):
    returns = discounted_returns(rewards, gamma)
    return (returns - returns.mean()) / jnp.maximum(1e-4, returns.std())


def surrogate_loss(spec, config, params, batch, entropy_coeff):
    logits, _, _ = models.sequence(spec, params, batch["obs"], batch.get("chunk_starts"))
    log_probs = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(log_probs, batch["action"][..., None], -1)[..., 0]
    ratio = jnp.exp(logp - batch["logp"])
    advantages = batch["advantages"]
    clipped = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param)
    surrogate = jnp.minimum(advantages * ratio, advantages * clipped)
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1)
    loss = jnp.mean(-surrogate - entropy_coeff * entropy)
    return loss, {"policy_loss": -surrogate.mean(), "entropy": entropy.mean()}


def make_iteration(spec, process, config: TrainConfig, optimizer):
    def update(params, opt_state, batch, entropy_coeff, key):
        n = config.num_envs
        size = n // config.num_minibatches

        def epoch(carry, epoch_key):
            def minibatch(carry, indices):
                params, opt_state = carry
                sub = jax.tree_util.tree_map(lambda a: a[indices], batch)
                (_, info), grads = jax.value_and_grad(
                    partial(surrogate_loss, spec, config), has_aux=True
                )(params, sub, entropy_coeff)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                info["grad_norm"] = optax.tree.norm(grads)
                return (optax.apply_updates(params, updates), opt_state), info

            order = jnp.arange(n) if config.num_minibatches == 1 else jax.random.permutation(epoch_key, n)
            return jax.lax.scan(minibatch, carry, order[: size * config.num_minibatches].reshape(-1, size))

        (params, opt_state), info = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(key, config.num_epochs)
        )
        return params, opt_state, jax.tree_util.tree_map(lambda a: a[-1, -1], info)

    @jax.jit
    def iteration(params, opt_state, key, entropy_coeff):
        rollout_key, update_key = jax.random.split(key)
        records = rollout(spec, process, params, rollout_key, config.num_envs, config.episode_length)
        batch = {name: records[name] for name in ("obs", "action", "logp")}
        if spec.kind == "gru":
            batch["chunk_starts"] = records["chunk_starts"]
        batch["advantages"] = standardized_advantages(records["reward"], config.gamma)
        params, opt_state, info = update(params, opt_state, batch, entropy_coeff, update_key)
        actions = jax.nn.one_hot(records["action"], spec.n_actions).mean((0, 1))
        metrics = {
            **info,
            "episode_return_mean": records["reward"].sum(1).mean(),
            "reward_per_step": records["reward"].mean(),
            "reward_components_per_step": records["components"].mean((0, 1)),
            "sampled_entropy": records["entropy"].mean(),
            "action_fractions": actions,
        }
        return params, opt_state, metrics

    return iteration


def train(spec, process, config: TrainConfig, *, seed: int, checkpoint_steps=(),
          on_checkpoint: Callable | None = None, log: Callable | None = None):
    """Run the recipe; ``on_checkpoint(steps, params)`` fires at step 0, each
    requested threshold crossing, and the end."""

    root = jax.random.PRNGKey(seed)
    params = models.init_params(spec, jax.random.fold_in(root, 0))
    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(params)
    iteration = make_iteration(spec, process, config, optimizer)
    if on_checkpoint:
        on_checkpoint(0, params)
    pending = sorted(set(int(s) for s in checkpoint_steps if 0 < s))
    curves, steps, compile_seconds, train_seconds = [], 0, None, 0.0
    for index in range(config.iterations):
        coeff = jnp.float32(entropy_coefficient(config.entropy_schedule, steps))
        key = jax.random.fold_in(jax.random.fold_in(root, 1), index)
        start = time.perf_counter()
        if index == 0:
            compiled = iteration.lower(params, opt_state, key, coeff).compile()
            compile_seconds = time.perf_counter() - start
            start = time.perf_counter()
        params, opt_state, metrics = compiled(params, opt_state, key, coeff)
        metrics = jax.device_get(metrics)
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        steps += config.batch_steps
        record = {
            "iteration": index + 1, "env_steps": steps, "iteration_seconds": elapsed,
            "entropy_coeff": float(coeff),
            **{name: np.asarray(value).tolist() for name, value in metrics.items()},
        }
        curves.append(record)
        if log:
            log(record)
        while pending and steps >= pending[0]:
            pending.pop(0)
            if on_checkpoint and index + 1 < config.iterations:
                on_checkpoint(steps, params)
    if on_checkpoint:
        on_checkpoint(steps, params)
    summary = {
        "env_steps": steps,
        "iterations": config.iterations,
        "batch_steps": config.batch_steps,
        "compile_seconds": compile_seconds,
        "train_seconds": train_seconds,
        "env_steps_per_second": steps / train_seconds if train_seconds else None,
        "steady_env_steps_per_second": (
            (steps - config.batch_steps) / (train_seconds - curves[0]["iteration_seconds"])
            if len(curves) > 1 else None
        ),
        "parameter_count": models.parameter_count(params),
        "device": str(jax.devices()[0]),
        "final_reward_per_step": curves[-1]["reward_per_step"],
    }
    return params, curves, summary
