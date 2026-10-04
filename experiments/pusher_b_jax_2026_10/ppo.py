"""Pusher-B token-guess PPO in pure JAX (no RLlib/Ray).

Recipe of ``experiments/pusher_b/rl.py``: complete 127-step episodes from a
fresh reset, the 4-layer d128 RoPE/RMSNorm/gated-GELU actor-critic over the
full episode, gamma = lambda = 0 (advantage = reward - value), advantages
standardised over the batch (RLlib GAE connector), clipped surrogate 0.2,
value loss 0.25 x min(mse, 10), Adam (eps 1e-8, no grad clip), 6 epochs,
lr 5e-5 -> 1e-5 and entropy 0.01 -> 0 piecewise-linear in lifetime env
steps. RLlib's batch was ~262k steps in 64-episode (8k-step) minibatches;
here a batch is ``num_envs`` synchronised episodes (2,048 x 127 = 260,096)
split into 64-episode minibatches. Optional next-token auxiliary CE
(``pusher_b/learning.py``): an extra head predicts each step's pending token,
coefficient 1.

Rollouts use the KV-cached step path, training the full causal forward.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from envs.hmm.jax_env import JaxHMMEnv
from harness.artifacts import RunArtifacts

from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.process import PRESETS, make_env, pusher_b_model

Schedule = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PPOConfig:
    total_env_steps: int = 15_000_000
    num_envs: int = 2_048
    minibatch_episodes: int = 64
    num_epochs: int = 6
    lr_schedule: Schedule = ((0, 5e-5), (5_000_000, 5e-5), (10_000_000, 1e-5))
    entropy_schedule: Schedule = ((0, 0.01), (6_000_000, 0.01), (8_000_000, 0.0))
    gamma: float = 0.0
    gae_lambda: float = 0.0
    clip_param: float = 0.2
    vf_coeff: float = 0.25
    vf_clip: float = 10.0
    next_token_aux: bool = False
    aux_coeff: float = 1.0
    rollout_chunks: int = 1  # split the KV-cached rollout to bound memory

    def __post_init__(self) -> None:
        if self.num_envs % self.minibatch_episodes:
            raise ValueError("num_envs must be divisible by minibatch_episodes")
        if self.num_envs % self.rollout_chunks:
            raise ValueError("num_envs must be divisible by rollout_chunks")

    @classmethod
    def smoke(cls, **overrides) -> PPOConfig:
        return cls(
            **{
                "total_env_steps": 2 * 8 * 127,
                "num_envs": 8,
                "minibatch_episodes": 4,
                "num_epochs": 2,
                **overrides,
            }
        )


class RunnerState(NamedTuple):
    params: dict
    opt_state: Any
    key: jax.Array
    env_steps: jax.Array


class Batch(NamedTuple):
    tokens: jax.Array  # (N, T) policy input tokens (BOS first)
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    raw_token: jax.Array  # pending token each action is scored against


def schedule_value(schedule: Schedule, env_steps: jax.Array) -> jax.Array:
    xs = jnp.array([float(x) for x, _ in schedule])
    ys = jnp.array([float(y) for _, y in schedule])
    return jnp.interp(env_steps.astype(jnp.float32), xs, ys)


def _optimizer(config: PPOConfig):
    return optax.inject_hyperparams(optax.adam)(
        learning_rate=config.lr_schedule[0][1], eps=1e-8
    )


def init(config: PPOConfig, spec: tm.ModelSpec, key: jax.Array) -> RunnerState:
    key, param_key = jax.random.split(key)
    params = tm.init_params(
        spec, param_key, actor_critic=True, aux_head=config.next_token_aux
    )
    return RunnerState(params, _optimizer(config).init(params), key, jnp.int32(0))


def collect(
    env: JaxHMMEnv, spec: tm.ModelSpec, num_envs: int, params: dict, key: jax.Array
) -> Batch:
    """One synchronised episode per env with the KV-cached policy."""

    reset_key, key = jax.random.split(key)
    obs, env_state = jax.vmap(env.reset)(jax.random.split(reset_key, num_envs))
    cache = tm.empty_cache(spec, num_envs)

    def body(carry, inputs):
        env_state, obs, cache = carry
        position, step_key = inputs
        tokens = tm.observation_tokens(obs)
        embedding, cache = tm.encode_step(spec, params, cache, tokens, position)
        logits, value = tm.policy_value(params, embedding)
        action_key, env_key = jax.random.split(step_key)
        action = jax.random.categorical(action_key, logits)
        log_prob = jnp.take_along_axis(
            jax.nn.log_softmax(logits), action[:, None], axis=-1
        )[:, 0]
        out = jax.vmap(env.step)(
            jax.random.split(env_key, num_envs), env_state, action
        )
        return (out.state, out.obs, cache), Batch(
            tokens, action, log_prob, value, out.reward, out.raw_token_before
        )

    length = env.episode_length
    _, traj = jax.lax.scan(
        body,
        (env_state, obs, cache),
        (jnp.arange(length), jax.random.split(key, length)),
    )
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)


def gae(config: PPOConfig, reward: jax.Array, value: jax.Array):
    """Complete episodes: no bootstrap after the last step. (N, T) arrays."""

    next_value = jnp.concatenate([value[:, 1:], jnp.zeros_like(value[:, :1])], axis=1)
    delta = reward + config.gamma * next_value - value

    def body(running, d):
        running = d + config.gamma * config.gae_lambda * running
        return running, running

    _, advantages = jax.lax.scan(
        body, jnp.zeros(delta.shape[0]), delta.T, reverse=True
    )
    advantages = advantages.T
    return advantages, advantages + value


def _loss(config, spec, params, entropy_coeff, mb):
    tokens, action, old_log_prob, advantages, targets, raw_token = mb
    embedding = tm.encode(spec, params, tokens)
    logits, values = tm.policy_value(params, embedding)
    log_probs = jax.nn.log_softmax(logits)
    log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]
    ratio = jnp.exp(log_prob - old_log_prob)
    eps = config.clip_param
    surrogate = jnp.minimum(
        ratio * advantages, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
    )
    policy_loss = -surrogate.mean()
    vf_loss = jnp.minimum((values - targets) ** 2, config.vf_clip).mean()
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
    total = policy_loss + config.vf_coeff * vf_loss - entropy_coeff * entropy
    aux = {
        "policy_loss": policy_loss,
        "vf_loss": vf_loss,
        "entropy": entropy,
        "mean_kl": (old_log_prob - log_prob).mean(),
    }
    if config.next_token_aux:
        # Row t+1 of the delayed observation reveals the token scored at t;
        # the last step has no next row inside the episode.
        aux_logits = tm._dense(params["aux"], embedding[:, :-1])
        aux_ce = optax.softmax_cross_entropy_with_integer_labels(
            aux_logits, raw_token[:, :-1]
        ).mean()
        total = total + config.aux_coeff * aux_ce
        aux["next_token_aux_ce"] = aux_ce
    return total, aux


def make_update(config: PPOConfig, env: JaxHMMEnv, spec: tm.ModelSpec):
    optimizer = _optimizer(config)
    loss_and_grad = jax.value_and_grad(partial(_loss, config, spec), has_aux=True)
    num_minibatches = config.num_envs // config.minibatch_episodes
    batch_size = config.num_envs * env.episode_length

    def update(state: RunnerState) -> tuple[RunnerState, dict]:
        key, collect_key, shuffle_key = jax.random.split(state.key, 3)
        chunk = config.num_envs // config.rollout_chunks
        traj = jax.lax.map(
            lambda k: collect(env, spec, chunk, state.params, k),
            jax.random.split(collect_key, config.rollout_chunks),
        )
        traj = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), traj)
        env_steps = state.env_steps + batch_size
        lr = schedule_value(config.lr_schedule, env_steps)
        entropy_coeff = schedule_value(config.entropy_schedule, env_steps)
        advantages, targets = gae(config, traj.reward, traj.value)
        advantages = (advantages - advantages.mean()) / jnp.maximum(
            advantages.std(), 1e-4
        )
        batch = (
            traj.tokens,
            traj.action,
            traj.log_prob,
            advantages,
            targets,
            traj.raw_token,
        )
        opt_state = state.opt_state
        opt_state.hyperparams["learning_rate"] = lr

        def epoch(carry, epoch_key):
            order = jax.random.permutation(epoch_key, config.num_envs)
            minibatches = jax.tree.map(
                lambda x: x[order].reshape((num_minibatches, -1) + x.shape[1:]),
                batch,
            )

            def minibatch(carry, mb):
                params, opt_state = carry
                (_, aux), grads = loss_and_grad(params, entropy_coeff, mb)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            return jax.lax.scan(minibatch, carry, minibatches)

        (params, opt_state), aux = jax.lax.scan(
            epoch,
            (state.params, opt_state),
            jax.random.split(shuffle_key, config.num_epochs),
        )
        returns = traj.reward.sum(1)
        predictive = env.predictive_distributions(traj.raw_token)
        metrics = {
            **jax.tree.map(jnp.mean, aux),
            "steps": env_steps,
            "return_mean": returns.mean(),
            "return_min": returns.min(),
            "return_max": returns.max(),
            "bayes_greedy_return_mean": predictive.max(-1).sum(-1).mean(),
            "lr": lr,
            "entropy_coeff": entropy_coeff,
        }
        return RunnerState(params, opt_state, key, env_steps), metrics

    return update


def train(
    preset: str,
    *,
    seed: int,
    config: PPOConfig | None = None,
    spec: tm.ModelSpec | None = None,
    log=print,
) -> tuple[RunnerState, list[dict], dict]:
    config = config or PPOConfig()
    spec = spec or tm.ModelSpec()
    env = make_env(preset)
    batch_size = config.num_envs * env.episode_length
    num_updates = math.ceil(config.total_env_steps / batch_size)
    state = init(config, spec, jax.random.key(seed))
    update = jax.jit(make_update(config, env, spec), donate_argnums=0)
    t0 = time.perf_counter()
    compiled = update.lower(state).compile()
    compile_s = time.perf_counter() - t0
    started = time.perf_counter()
    curves = []
    for iteration in range(1, num_updates + 1):
        t0 = time.perf_counter()
        state, metrics = compiled(state)
        metrics = {k: float(v) for k, v in metrics.items()}
        dt = time.perf_counter() - t0
        record = {
            "iteration": iteration,
            **metrics,
            "time_iter_s": dt,
            "time_total_s": time.perf_counter() - started,
            "env_steps_per_s": batch_size / dt,
        }
        curves.append(record)
        log(record)
    active = curves[-1]["time_total_s"]
    summary = {
        "completed_env_steps": int(state.env_steps),
        "target_env_steps": config.total_env_steps,
        "num_updates": num_updates,
        "episode_length_mean": float(env.episode_length),
        "episode_return_mean": curves[-1]["return_mean"],
        "compile_seconds": compile_s,
        "active_training_wall_seconds": active,
        "env_steps_per_second_active": int(state.env_steps) / max(active, 1e-9),
        "device": str(jax.devices()[0]),
        "parameter_count": tm.parameter_count(state.params),
    }
    return state, curves, summary


def run_ppo(context, *, preset: str, config: PPOConfig | None = None) -> dict:
    config = config or PPOConfig()
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    if context.smoke:
        config = PPOConfig.smoke(next_token_aux=config.next_token_aux)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    spec = tm.ModelSpec()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "pusher_b_jax",
            "condition": "ppo_token_guess",
            "preset": preset,
            "parameters": PRESETS[preset],
            "seed": context.seed,
            "smoke": context.smoke,
            "legacy_recipe": "experiments/pusher_b/rl.py",
            "edge_transition_matrices": pusher_b_model(preset).edge_transition_matrices.tolist(),
            "reward": "1 when the sampled action equals the pending token, else 0",
            "model": asdict(spec),
            "ppo": asdict(config),
        },
    )
    state, curves, summary = train(
        preset, seed=context.seed, config=config, spec=spec, log=outputs.append_result
    )
    with open(context.results_dir / "training_curves.jsonl", "w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in curves)
    np.savez(
        context.artifacts_dir / "final_params.npz",
        **{
            jax.tree_util.keystr(k): np.asarray(v)
            for k, v in jax.tree_util.tree_leaves_with_path(state.params)
        },
    )
    summary = {
        "study": "pusher_b_jax",
        "condition": "ppo_token_guess",
        "preset": preset,
        "parameters": PRESETS[preset],
        "seed": context.seed,
        "smoke": context.smoke,
        "next_token_aux": config.next_token_aux,
        **summary,
    }
    outputs.write_json("summary.json", summary)
    return summary
