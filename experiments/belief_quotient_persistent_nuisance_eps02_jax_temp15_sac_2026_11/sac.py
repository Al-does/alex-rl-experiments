"""Discrete-action SAC with independent transformer critics and episode replay."""
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.belief_quotient_history_jax_2026_11 import models, reinforce
from experiments.belief_quotient_history_jax_2026_11.training import _save


@dataclass(frozen=True)
class TrainConfig(reinforce.TrainConfig):
    algorithm: str = "SAC"
    alpha: float = 0.0
    auto_entropy_tuning: bool = False
    tau: float = 0.005
    replay_rollouts: int = 4
    gradient_steps: int = 64
    critic_lr: float = 4.2e-4
    q_scaling: str = "1-gamma"

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.replay_rollouts < 1 or self.gradient_steps < 1:
            raise ValueError("replay_rollouts and gradient_steps must be positive")
        if self.alpha < 0 or self.auto_entropy_tuning:
            raise ValueError("alpha must be nonnegative and fixed")


def configure(config):
    return TrainConfig(**asdict(config))


def for_context(context):
    def train_in_context(*args, **kwargs):
        return train(*args, **kwargs,
                     critic_checkpoint_dir=context.artifacts_dir / "checkpoints")
    return SimpleNamespace(configure=configure, train=train_in_context)


def init_buffer(config):
    shape = (config.replay_rollouts * config.num_envs, config.episode_length)
    return {
        "token": jnp.zeros(shape, jnp.uint8),
        "action": jnp.zeros(shape, jnp.uint8),
        "reward": jnp.zeros(shape, jnp.float32),
        "position": jnp.int32(0),
        "size": jnp.int32(0),
    }


def buffer_bytes(config):
    return config.replay_rollouts * config.num_envs * config.episode_length * 6 + 8


def insert_buffer(buffer, records, n_tokens):
    count = records["action"].shape[0]
    capacity = buffer["token"].shape[0]
    indices = (buffer["position"] + jnp.arange(count)) % capacity
    packed = {
        "token": jnp.argmax(records["obs"][..., :n_tokens], -1).astype(jnp.uint8),
        "action": records["action"].astype(jnp.uint8),
        "reward": records["reward"].astype(jnp.float32),
    }
    return {
        **{name: buffer[name].at[indices].set(value) for name, value in packed.items()},
        "position": (buffer["position"] + count) % capacity,
        "size": jnp.minimum(buffer["size"] + count, capacity),
    }


def sample_buffer(buffer, key, batch_size, n_tokens, n_actions):
    indices = jax.random.randint(key, (batch_size,), 0, buffer["size"])
    tokens = buffer["token"][indices].astype(jnp.int32)
    actions = buffer["action"][indices].astype(jnp.int32)
    previous = jnp.concatenate([jnp.full_like(actions[:, :1], -1), actions[:, :-1]], 1)
    obs = jnp.concatenate([jax.nn.one_hot(tokens, n_tokens),
                           jax.nn.one_hot(previous, n_actions)], -1)
    return {"obs": obs, "action": actions, "reward": buffer["reward"][indices]}


def critic_targets(rewards, next_log_probs, target_q1, target_q2, gamma, alpha):
    soft_values = (jnp.exp(next_log_probs) *
                   (jnp.minimum(target_q1, target_q2) - alpha * next_log_probs)).sum(-1)
    shifted = jnp.concatenate([soft_values[:, 1:], jnp.zeros_like(soft_values[:, :1])], 1)
    return jax.lax.stop_gradient(rewards + gamma * shifted)


def actor_objective(log_probs, q1, q2, alpha):
    q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
    return (jnp.exp(log_probs) * (alpha * log_probs - q)).sum(-1).mean()


def polyak_update(target, online, tau):
    return jax.tree.map(lambda old, new: (1 - tau) * old + tau * new, target, online)


def q_values(spec, params, obs, gamma):
    logits, _, _ = models.sequence(spec, params, obs)
    return logits / (1 - gamma)


def critic_loss(spec, config, critics, batch, targets):
    def one(params):
        predictions, _, _ = models.sequence(spec, params, batch["obs"])
        selected = jnp.take_along_axis(predictions, batch["action"][..., None], -1)[..., 0]
        return jnp.mean((selected - (1 - config.gamma) * targets) ** 2)
    return one(critics[0]) + one(critics[1])


def actor_loss(spec, config, actor, obs, q1, q2):
    logits, _, _ = models.sequence(spec, actor, obs)
    log_probs = jax.nn.log_softmax(logits / config.temperature)
    loss = actor_objective(log_probs, q1, q2, config.alpha)
    return loss, {
        "actor_loss": loss,
        "mean_q": jnp.minimum(q1, q2).mean(),
        "policy_entropy": -(jnp.exp(log_probs) * log_probs).sum(-1).mean(),
    }


def make_iteration(spec, process, config, actor_optimizer, critic_optimizer):
    batch_size = config.num_envs // config.num_minibatches

    def gradient_step(state, key):
        actor, critics, targets, actor_opt, critic_opt, buffer = state
        batch = sample_buffer(buffer, key, batch_size, process.n_tokens, spec.n_actions)
        logits, _, _ = models.sequence(spec, actor, batch["obs"])
        log_probs = jax.nn.log_softmax(logits / config.temperature)
        target_q = [q_values(spec, q, batch["obs"], config.gamma) for q in targets]
        y = critic_targets(batch["reward"], log_probs, *target_q, config.gamma, config.alpha)
        loss, grads = jax.value_and_grad(partial(critic_loss, spec, config))(
            critics, batch, y)
        updates, critic_opt = critic_optimizer.update(grads, critic_opt, critics)
        critics = optax.apply_updates(critics, updates)
        q1, q2 = [jax.lax.stop_gradient(q_values(spec, q, batch["obs"], config.gamma))
                  for q in critics]
        (_, info), grads = jax.value_and_grad(partial(actor_loss, spec, config), has_aux=True)(
            actor, batch["obs"], q1, q2)
        updates, actor_opt = actor_optimizer.update(grads, actor_opt, actor)
        actor = optax.apply_updates(actor, updates)
        targets = polyak_update(targets, critics, config.tau)
        info["critic_loss"] = loss
        return (actor, critics, targets, actor_opt, critic_opt, buffer), info

    @partial(jax.jit, donate_argnums=(0,))
    def iteration(state, key):
        actor, critics, targets, actor_opt, critic_opt, buffer = state
        rollout_key, update_key = jax.random.split(key)
        records = reinforce.rollout(
            spec, process, actor, rollout_key, config.num_envs, config.episode_length,
            temperature=config.temperature, epsilon=config.exploration_epsilon)
        buffer = insert_buffer(buffer, records, process.n_tokens)
        state, infos = jax.lax.scan(
            gradient_step, (actor, critics, targets, actor_opt, critic_opt, buffer),
            jax.random.split(update_key, config.gradient_steps))
        metrics = {
            **jax.tree.map(lambda a: a.mean(), infos),
            "episode_return_mean": records["reward"].sum(1).mean(),
            "reward_per_step": records["reward"].mean(),
            "reward_components_per_step": records["components"].mean((0, 1)),
            "sampled_entropy": records["entropy"].mean(),
            "action_fractions": jax.nn.one_hot(records["action"], spec.n_actions).mean((0, 1)),
        }
        return state, metrics

    return iteration


def train(spec, process, config, *, seed, checkpoint_steps=(), on_checkpoint=None,
          log=None, initial_params=None, step_offset=0, critic_checkpoint_dir=None):
    if spec.kind != "transformer":
        raise ValueError("SAC supports transformer episodes only")
    if step_offset or initial_params is not None:
        raise ValueError("SAC studies require fresh training")
    if config.exploration_epsilon != 0:
        raise ValueError("SAC requires exploration_epsilon=0")
    root = jax.random.PRNGKey(seed)
    actor = models.init_params(spec, jax.random.fold_in(root, 0))
    critics = tuple(models.init_params(spec, jax.random.fold_in(root, i)) for i in (3, 4))
    targets = jax.tree.map(lambda x: x.copy(), critics)
    actor_optimizer, critic_optimizer = optax.adam(config.lr), optax.adam(config.critic_lr)
    state = (actor, critics, targets, actor_optimizer.init(actor),
             critic_optimizer.init(critics), init_buffer(config))
    iteration = make_iteration(spec, process, config, actor_optimizer, critic_optimizer)
    if on_checkpoint:
        on_checkpoint(0, actor)
    pending = sorted(set(int(s) for s in checkpoint_steps if s > 0))
    curves, steps, compile_seconds, train_seconds = [], 0, None, 0.
    for index in range(config.iterations):
        key = jax.random.fold_in(jax.random.fold_in(root, 1), index)
        start = time.perf_counter()
        if index == 0:
            compiled = iteration.lower(state, key).compile()
            compile_seconds = time.perf_counter() - start
            start = time.perf_counter()
        state, metrics = compiled(state, key)
        metrics = jax.device_get(metrics)
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        steps += config.batch_steps
        record = {
            "iteration": index + 1, "env_steps": steps, "iteration_seconds": elapsed,
            "entropy_coeff": config.alpha,
            **{k: np.asarray(v).tolist() for k, v in metrics.items()},
        }
        curves.append(record)
        if log:
            log(record)
        crossed = False
        while pending and steps >= pending[0]:
            pending.pop(0)
            crossed = True
        if crossed and on_checkpoint and index + 1 < config.iterations:
            on_checkpoint(steps, state[0])
    actor, critics, targets = state[:3]
    if on_checkpoint:
        on_checkpoint(steps, actor)
    if critic_checkpoint_dir is not None:
        directory = Path(critic_checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _save(directory / f"critic_step_{steps:010d}.npz",
              {"critics": critics, "targets": targets})
    summary = {
        "env_steps": steps, "step_offset": 0, "iterations": config.iterations,
        "batch_steps": config.batch_steps, "compile_seconds": compile_seconds,
        "train_seconds": train_seconds, "env_steps_per_second": steps / train_seconds,
        "steady_env_steps_per_second": (
            (steps - config.batch_steps) / (train_seconds - curves[0]["iteration_seconds"])
            if len(curves) > 1 else None),
        "parameter_count": models.parameter_count(actor),
        "critic_parameter_count": models.parameter_count(critics),
        "replay_buffer_bytes": buffer_bytes(config),
        "device": str(jax.devices()[0]),
        "final_reward_per_step": curves[-1]["reward_per_step"],
    }
    return actor, curves, summary
