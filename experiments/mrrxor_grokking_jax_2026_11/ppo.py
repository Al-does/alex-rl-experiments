"""Finite-pool, action-independent PPO using the Pusher transformer/loss."""

import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from harness.artifacts import RunArtifacts

from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.ppo import PPOConfig as BaseConfig, _loss, gae, schedule_value
from experiments.pusher_b_jax_2026_10.heldout import AsyncCheckpointWriter, _upload_policy
from experiments.mrrxor_grokking_jax_2026_11 import process
from experiments.mrrxor_grokking_jax_2026_11.analysis import evaluate


@dataclass(frozen=True)
class PPOConfig(BaseConfig):
    total_env_steps: int = 100_000_000
    num_envs: int = 128
    minibatch_episodes: int = 32
    num_epochs: int = 4
    lr_schedule: tuple = ((0, 3e-4),)
    entropy_schedule: tuple = ((0, 0.01),)
    weight_decay: float = 1.0
    train_sequences: int = 1024
    heldout_sequences: int = 4096
    sequence_length: int = 32
    checkpoint_every: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if min(self.total_env_steps, self.num_envs, self.num_epochs, self.train_sequences,
               self.heldout_sequences, self.sequence_length, self.checkpoint_every) <= 0:
            raise ValueError("budgets and sizes must be positive")
        if self.gamma != 0 or self.gae_lambda != 0 or self.next_token_aux:
            raise ValueError("this study requires gamma=lambda=0 and no supervised auxiliary loss")

    @classmethod
    def smoke(cls, **overrides):
        return cls(**{"total_env_steps": 2048, "num_envs": 32,
                      "minibatch_episodes": 16, "num_epochs": 1,
                      "train_sequences": 32, "heldout_sequences": 64, **overrides})


def checkpoint_updates(total, every):
    points = {0, total, *range(every, total + 1, every)}
    power = 1
    while power <= total:
        points.update(x * power for x in (1, 2, 5) if x * power <= total)
        power *= 10
    return sorted(points)


@lru_cache(maxsize=8)
def make_update(config, spec):
    optimizer = optax.scale_by_adam()
    loss_grad = jax.value_and_grad(partial(_loss, config, spec), has_aux=True)

    @jax.jit
    def update(params, opt_state, key, env_steps, pool, lr_schedule, weight_decay):
        lr = jnp.interp(env_steps, lr_schedule[:, 0], lr_schedule[:, 1])
        key, data_key, act_key, shuffle_key = jax.random.split(key, 4)
        raw = pool[jax.random.randint(data_key, (config.num_envs,), 0, len(pool))]
        tokens = jnp.concatenate([jnp.full((len(raw), 1), 2, dtype=jnp.int32), raw[:, :-1]], -1)
        logits, values = tm.policy_value(params, tm.encode(spec, params, tokens))
        actions = jax.random.categorical(act_key, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], -1)[..., 0]
        rewards = process.reward(actions, raw)
        adv, targets = gae(config, rewards, values)
        adv = (adv - adv.mean()) / jnp.maximum(adv.std(), 1e-4)
        batch = jax.tree.map(jax.lax.stop_gradient, (tokens, actions, logp, adv, targets, raw))
        entropy = schedule_value(config.entropy_schedule, env_steps)

        def epoch(carry, key):
            order = jax.random.permutation(key, config.num_envs)
            minibatches = jax.tree.map(lambda x: x[order].reshape(
                (config.num_envs // config.minibatch_episodes, config.minibatch_episodes) + x.shape[1:]), batch)

            def minibatch(carry, mb):
                p, state = carry
                (_, metrics), grads = loss_grad(p, entropy, mb)
                updates, state = optimizer.update(grads, state, p)
                updates = jax.tree.map(lambda u, w: -lr * (u + weight_decay * w), updates, p)
                return (optax.apply_updates(p, updates), state), metrics
            return jax.lax.scan(minibatch, carry, minibatches)

        (params, opt_state), metrics = jax.lax.scan(epoch, (params, opt_state),
                                                  jax.random.split(shuffle_key, config.num_epochs))
        return params, opt_state, key, jax.tree.map(jnp.mean, metrics)
    return optimizer, update


def train(context, config=None, spec=None):
    if context.resume_from:
        raise ValueError("optimizer resume is not supported; checkpoints are analysis snapshots")
    config = config or (PPOConfig.smoke() if context.smoke else PPOConfig())
    spec = spec or tm.ModelSpec(vocab=3, context_length=config.sequence_length)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    root = Path(context.artifacts_dir)
    pools = process.fixed_pools(context.seed, config.train_sequences, config.heldout_sequences, config.sequence_length)
    np.savez(root / "pools.npz", **pools)
    recipe = {"seed": context.seed, "model": asdict(spec), "training": asdict(config),
              "process": "MRRXOR eps=0.05, jump before emit, stationary reset",
              "jax_version": jax.__version__, "parameter_snapshot_only": True}
    (root / "recipe.json").write_text(json.dumps(recipe, indent=2))
    outputs.write_json("resolved_recipe.json", recipe)
    params = tm.init_params(spec, jax.random.PRNGKey(context.seed), actor_critic=True)
    compiled_config = replace(config, train_sequences=1024, heldout_sequences=4096,
                              lr_schedule=((0, 1e-4),), weight_decay=0.1)
    optimizer, update = make_update(compiled_config, spec)
    pool = jnp.asarray(pools["train"])
    lr_schedule = jnp.asarray(config.lr_schedule, dtype=jnp.float32)
    state, key = optimizer.init(params), jax.random.PRNGKey(context.seed + 1)
    total = math.ceil(config.total_env_steps / (config.num_envs * config.sequence_length))
    checkpoints = set(checkpoint_updates(total, config.checkpoint_every))
    writer = AsyncCheckpointWriter(context, root / "param_checkpoints", upload=_upload_policy(context))
    index, curve = [], []
    try:
        for step in range(total + 1):
            env_steps = step * config.num_envs * config.sequence_length
            if step in checkpoints:
                writer.submit(step, jax.device_get(params))
                entry = {"update": step, "env_steps": env_steps,
                         "path": f"param_checkpoints/update_{step:07d}/params.npz"}
                index.append(entry)
                metrics, _, _ = evaluate(params, pools, spec)
                record = {**entry, "metrics": metrics}
                curve.append(record)
                outputs.append_result(record)
                print(json.dumps(record), flush=True)
            if step < total:
                params, state, key, _ = update(params, state, key, jnp.int32(env_steps),
                                              pool, lr_schedule, jnp.float32(config.weight_decay))
    finally:
        uploads = writer.close()
        (root / "checkpoint_index.json").write_text(json.dumps(index, indent=2))
    outputs.write_json("curve.json", {"checkpoints": curve})
    summary = {"env_steps": total * config.num_envs * config.sequence_length,
               "checkpoint_count": len(index), "parameter_count": tm.parameter_count(params),
               "artifact_run_dir": str(root), "uploads": uploads}
    outputs.write_json("summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
