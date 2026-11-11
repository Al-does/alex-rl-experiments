"""Standard clipped PPO with a learned critic for the history studies, in JAX.

Same environment, rollout sampling and policy encoders as the critic-free
trainer in ``belief_quotient_history_jax_2026_11.reinforce``; the changes are
the PPO components:

- a linear value head on the shared post-final-LayerNorm embedding (the
  RL-Harness actor-critic layout), PyTorch default initialization;
- GAE(gamma, lambda) advantages, standardized over the batch, with the
  1,024-step episode end treated as a truncation and bootstrapped from the
  critic's value of the next observation;
- several epochs of shuffled whole-episode minibatches, clipped surrogate,
  unclipped squared-error value loss, entropy bonus, global-norm gradient
  clipping and Adam;
- optional RLlib-style adaptive KL penalty ``kl_coeff * KL(old || new)``,
  off by default (``use_kl_loss=False``). When on, the coefficient is x1.5
  if the epoch-averaged KL exceeds 2 x ``kl_target`` and x0.5 below half of it,
  updated once per rollout.

The critic regresses (1 - gamma)-scaled returns (per-step reward units), so
value targets stay O(1) on the shared torso; standardized advantages are
invariant to that scale. Rollout, GAE and updates run in one jitted function.
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
class PPOConfig:
    total_env_steps: int
    num_envs: int = 19_456
    episode_length: int = 1024
    num_minibatches: int = 64
    num_epochs: int = 4
    lr: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    vf_coeff: float = 0.5
    entropy_coeff: float = 0.015
    max_grad_norm: float = 0.5
    temperature: float = 1.0  # softmax temperature for sampling and the PPO likelihoods
    use_kl_loss: bool = False
    kl_coeff: float = 0.2  # initial coefficient (RLlib default); unused unless use_kl_loss
    kl_target: float = 0.01

    def __post_init__(self):
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @property
    def batch_steps(self) -> int:
        return self.num_envs * self.episode_length

    @property
    def iterations(self) -> int:
        return -(-self.total_env_steps // self.batch_steps)

    @property
    def value_scale(self) -> float:
        return 1.0 - self.gamma


def init_params(spec, key):
    """Actor parameters exactly as the critic-free trainer draws them for this
    key, plus an independently keyed linear value head."""

    params = models.init_params(spec, key)
    params["value"] = models._linear(jax.random.fold_in(key, 1), spec.width, 1)
    return params


def value(params, embedding):
    return models._dense(params["value"], embedding)[..., 0]


def rollout(spec, process, params, key, num_envs, episode_length, *, temperature=1.0,
            record_log_probs=False):
    """Sample synchronized episodes; same RNG stream and actions as
    ``reinforce.rollout`` for the same key. Also returns per-step critic values
    and the bootstrap value of the observation after the last step;
    ``record_log_probs`` keeps the full behaviour distribution for the KL term."""

    reset_key, key = jax.random.split(key)
    env_state = jax.vmap(hmm.reset, (None, 0))(process, jax.random.split(reset_key, num_envs))
    carry = models.init_carry(spec, num_envs)
    chunk = spec.chunk_len if spec.kind == "gru" else episode_length
    if episode_length % chunk:
        raise ValueError("episode length must be a multiple of the GRU chunk length")

    def decision(state, t):
        env_state, carry, key = state
        key, action_key, step_key = jax.random.split(key, 3)
        obs = jax.vmap(hmm.observe, (None, 0))(process, env_state)
        logits, _, embedding, carry = models.step(spec, params, carry, obs, t)
        logits = logits / temperature
        action = jax.random.categorical(action_key, logits, -1)
        log_probs = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(log_probs, action[:, None], -1)[:, 0]
        next_env, reward, components = jax.vmap(hmm.step, (None, 0, 0, 0))(
            process, jax.random.split(step_key, num_envs), env_state, action
        )
        record = {"obs": obs, "action": action, "logp": logp, "reward": reward,
                  "components": components, "value": value(params, embedding),
                  "entropy": -(jnp.exp(log_probs) * log_probs).sum(-1)}
        if record_log_probs:
            record["log_probs"] = log_probs
        return (next_env, carry, key), record

    def chunk_body(state, chunk_index):
        start_carry = state[1]
        times = chunk_index * chunk + jnp.arange(chunk)
        state, records = jax.lax.scan(decision, state, times)
        return state, (records, start_carry if spec.kind == "gru" else ())

    (env_state, carry, _), (records, starts) = jax.lax.scan(
        chunk_body, (env_state, carry, key), jnp.arange(episode_length // chunk)
    )
    records = jax.tree_util.tree_map(
        lambda a: jnp.moveaxis(a.reshape(episode_length, *a.shape[2:]), 0, 1), records
    )
    if spec.kind == "gru":
        records["chunk_starts"] = jnp.transpose(starts, (2, 1, 0, 3))  # (B, L, C, H)
    final_obs = jax.vmap(hmm.observe, (None, 0))(process, env_state)
    _, _, embedding, _ = models.step(spec, params, carry, final_obs, jnp.int32(episode_length))
    records["bootstrap_value"] = value(params, embedding)
    return records


def gae(rewards, values, bootstrap, gamma, gae_lambda):
    """GAE over ``(B, T)`` truncated episodes; returns (advantages, value targets)."""

    next_values = jnp.concatenate([values[:, 1:], bootstrap[:, None]], 1)
    deltas = rewards + gamma * next_values - values

    def body(running, delta):
        running = delta + gamma * gae_lambda * running
        return running, running

    _, advantages = jax.lax.scan(body, jnp.zeros(rewards.shape[0]), deltas.T, reverse=True)
    advantages = advantages.T
    return advantages, advantages + values


def ppo_loss(spec, config, params, batch, kl_coeff):
    logits, _, embedding = models.sequence(spec, params, batch["obs"], batch.get("chunk_starts"))
    log_probs = jax.nn.log_softmax(logits / config.temperature)
    logp = jnp.take_along_axis(log_probs, batch["action"][..., None], -1)[..., 0]
    log_ratio = logp - batch["logp"]
    ratio = jnp.exp(log_ratio)
    advantages = batch["advantages"]
    clipped = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param)
    surrogate = jnp.minimum(advantages * ratio, advantages * clipped)
    vf_loss = jnp.mean((value(params, embedding) - batch["value_targets"]) ** 2)
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1)
    policy_loss = -surrogate.mean()
    loss = policy_loss + config.vf_coeff * vf_loss - config.entropy_coeff * entropy.mean()
    info = {
        "policy_loss": policy_loss, "vf_loss": vf_loss, "entropy": entropy.mean(),
        "approx_kl": jnp.mean(ratio - 1 - log_ratio),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1) > config.clip_param),
    }
    if config.use_kl_loss:
        old = batch["log_probs"]
        info["kl"] = (jnp.exp(old) * (old - log_probs)).sum(-1).mean()
        loss = loss + kl_coeff * info["kl"]
    return loss, info


def adapt_kl_coeff(config, kl_coeff, kl):
    return jnp.where(kl > 2 * config.kl_target, kl_coeff * 1.5,
                     jnp.where(kl < .5 * config.kl_target, kl_coeff * .5, kl_coeff))


def make_optimizer(config: PPOConfig):
    return optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                       optax.adam(config.lr, eps=config.adam_eps))


def make_iteration(spec, process, config: PPOConfig, optimizer):
    n, size = config.num_envs, config.num_envs // config.num_minibatches

    def update(params, opt_state, batch, kl_coeff, key):
        def epoch(carry, epoch_key):
            def minibatch(carry, indices):
                params, opt_state = carry
                sub = jax.tree_util.tree_map(lambda a: a[indices], batch)
                (_, info), grads = jax.value_and_grad(
                    partial(ppo_loss, spec, config), has_aux=True)(params, sub, kl_coeff)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                info["grad_norm"] = optax.tree.norm(grads)
                return (optax.apply_updates(params, updates), opt_state), info

            order = jax.random.permutation(epoch_key, n).reshape(config.num_minibatches, size)
            return jax.lax.scan(minibatch, carry, order)

        (params, opt_state), info = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(key, config.num_epochs))
        # Last minibatch of the last epoch, plus epoch-averaged KL/clip diagnostics.
        last = jax.tree_util.tree_map(lambda a: a[-1, -1], info)
        last["approx_kl_mean"] = info["approx_kl"].mean()
        last["clip_fraction_mean"] = info["clip_fraction"].mean()
        if config.use_kl_loss:
            last["kl_mean"] = info["kl"].mean()
        return params, opt_state, last

    @jax.jit
    def iteration(params, opt_state, kl_coeff, key):
        rollout_key, update_key = jax.random.split(key)
        records = rollout(spec, process, params, rollout_key, n, config.episode_length,
                          temperature=config.temperature, record_log_probs=config.use_kl_loss)
        advantages, targets = gae(records["reward"] * config.value_scale, records["value"],
                                  records["bootstrap_value"], config.gamma, config.gae_lambda)
        batch = {name: records[name] for name in ("obs", "action", "logp")}
        if spec.kind == "gru":
            batch["chunk_starts"] = records["chunk_starts"]
        if config.use_kl_loss:
            batch["log_probs"] = records["log_probs"]
        batch["advantages"] = (advantages - advantages.mean()) / jnp.maximum(1e-8, advantages.std())
        batch["value_targets"] = targets
        params, opt_state, info = update(params, opt_state, batch, kl_coeff, update_key)
        residual = targets - records["value"]
        metrics = {
            **info,
            "episode_return_mean": records["reward"].sum(1).mean(),
            "reward_per_step": records["reward"].mean(),
            "reward_components_per_step": records["components"].mean((0, 1)),
            "sampled_entropy": records["entropy"].mean(),
            "action_fractions": jax.nn.one_hot(records["action"], spec.n_actions).mean((0, 1)),
            "value_mean": records["value"].mean(),
            "explained_variance": 1 - residual.var() / jnp.maximum(1e-12, targets.var()),
            "kl_coeff": kl_coeff,
        }
        if config.use_kl_loss:
            kl_coeff = adapt_kl_coeff(config, kl_coeff, info["kl_mean"])
        return params, opt_state, kl_coeff, metrics

    return iteration


def train(spec, process, config: PPOConfig, *, seed: int, checkpoint_steps=(),
          on_checkpoint: Callable | None = None, log: Callable | None = None):
    """Fresh run; ``on_checkpoint(steps, params)`` fires at the start, each
    requested threshold crossing, and the end. Keys follow the critic-free
    trainer, so step-0 actor parameters match it for the same seed."""

    root = jax.random.PRNGKey(seed)
    params = init_params(spec, jax.random.fold_in(root, 0))
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(params)
    kl_coeff = jnp.float32(config.kl_coeff if config.use_kl_loss else 0.0)
    iteration = make_iteration(spec, process, config, optimizer)
    if on_checkpoint:
        on_checkpoint(0, params)
    pending = sorted(set(int(s) for s in checkpoint_steps if s > 0))
    curves, steps, compile_seconds, train_seconds = [], 0, None, 0.0
    for index in range(config.iterations):
        key = jax.random.fold_in(jax.random.fold_in(root, 1), index)
        start = time.perf_counter()
        if index == 0:
            compiled = iteration.lower(params, opt_state, kl_coeff, key).compile()
            compile_seconds = time.perf_counter() - start
            start = time.perf_counter()
        params, opt_state, kl_coeff, metrics = compiled(params, opt_state, kl_coeff, key)
        metrics = jax.device_get(metrics)
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        steps += config.batch_steps
        record = {"iteration": index + 1, "env_steps": steps, "iteration_seconds": elapsed,
                  "entropy_coeff": config.entropy_coeff,
                  **{name: np.asarray(value).tolist() for name, value in metrics.items()}}
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
        "adam_steps": config.iterations * config.num_epochs * config.num_minibatches,
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
