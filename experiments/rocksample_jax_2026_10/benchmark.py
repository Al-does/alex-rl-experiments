"""Throughput benchmark: RLlib/Torch RockSample vs the pure-JAX port.

    python -m experiments.rocksample_jax_2026_10.benchmark env-old
    python -m experiments.rocksample_jax_2026_10.benchmark env-jax --num-envs 4096
    python -m experiments.rocksample_jax_2026_10.benchmark train-rllib
    python -m experiments.rocksample_jax_2026_10.benchmark train-jax --num-envs 64

Every mode prints one JSON line. JAX timings exclude compilation (reported
separately as ``compile_s``); RLlib timings exclude the first two iterations.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import numpy as np


def env_old(args) -> dict:
    from envs.rocksample import RockSampleEnv

    env = RockSampleEnv({})
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    actions = rng.integers(0, env.action_space.n, args.steps)
    start = time.perf_counter()
    for action in actions:
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    return {"steps": args.steps, "seconds": elapsed, "steps_per_s": args.steps / elapsed}


def env_jax(args) -> dict:
    import jax

    from experiments.rocksample_jax_2026_10 import env as rs

    params = rs.RockSampleParams.from_instance(5, 7)
    n = args.num_envs
    length = max(1, args.steps // n)

    @jax.jit
    def run(key):
        reset_key, key = jax.random.split(key)
        _, state = jax.vmap(partial(rs.reset, params))(jax.random.split(reset_key, n))

        def body(carry, step_key):
            state, total = carry
            action_key, env_key = jax.random.split(step_key)
            action = jax.random.randint(action_key, (n,), 0, params.num_actions)
            out = jax.vmap(partial(rs.step_autoreset, params))(
                jax.random.split(env_key, n), state, action
            )
            return (out.state, total + out.reward.sum() + out.obs.sum()), None

        (_, total), _ = jax.lax.scan(body, (state, 0.0), jax.random.split(key, length))
        return total

    start = time.perf_counter()
    run(jax.random.key(0)).block_until_ready()
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    run(jax.random.key(1)).block_until_ready()
    elapsed = time.perf_counter() - start
    steps = n * length
    return {
        "device": str(jax.devices()[0]),
        "num_envs": n,
        "steps": steps,
        "compile_s": compile_s,
        "seconds": elapsed,
        "steps_per_s": steps / elapsed,
    }


def train_jax(args) -> dict:
    import jax

    from experiments.rocksample_jax_2026_10 import env as rs
    from experiments.rocksample_jax_2026_10 import ppo

    params = rs.RockSampleParams.from_instance(5, 7)
    config = ppo.PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
    )
    spec = ppo.make_spec(params, d_model=args.d_model)
    chunk = ppo.make_train_chunk(config, params, spec, num_updates=args.updates_per_chunk)
    seeds = args.seeds
    keys = jax.random.split(jax.random.key(0), seeds)
    states = jax.jit(jax.vmap(partial(ppo.init, config, params, spec)))(keys)
    step = jax.jit(jax.vmap(chunk))

    start = time.perf_counter()
    states, metrics = step(states)
    jax.block_until_ready(metrics)
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(args.chunks):
        states, metrics = step(states)
    jax.block_until_ready(metrics)
    elapsed = time.perf_counter() - start
    steps = seeds * args.chunks * args.updates_per_chunk * config.batch_size
    returns = np.asarray(metrics["episode_return_mean"])[:, -1]
    return {
        "device": str(jax.devices()[0]),
        "seeds": seeds,
        "num_envs": config.num_envs,
        "batch_size": config.batch_size,
        "minibatch_size": config.batch_size // config.num_minibatches,
        "num_epochs": config.num_epochs,
        "d_model": args.d_model,
        "steps": steps,
        "compile_s": compile_s,
        "seconds": elapsed,
        "steps_per_s": steps / elapsed,
        "steps_per_s_per_seed": steps / elapsed / seeds,
        "env_steps_per_seed_at_end": int(np.asarray(metrics["env_steps"])[0, -1]),
        "last_return_mean": returns.tolist(),
    }


def train_rllib(args) -> dict:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from envs.rocksample import RockSampleEnv
    from harness.hardware import PROFILES, configure_hardware, resolve_env_runners
    from learners.models.transformer import TransformerModel, TransformerModelConfig

    profile = PROFILES[args.profile]
    configure_hardware(profile)
    # Same settings as experiments/rocksample_ppo_2026_09/shared.py (full run).
    config = (
        PPOConfig()
        .environment(RockSampleEnv, env_config={"n": 5, "k": 7})
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            vf_clip_param=1_000_000.0,
            entropy_coeff=0.05,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            train_batch_size_per_learner=8_192,
            minibatch_size=1_024,
            num_epochs=4,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=TransformerModelConfig(
                    d_model=args.d_model, n_layers=3, n_heads=4, context_len=32
                ).to_dict(),
            )
        )
        .debugging(seed=0)
        .env_runners(
            batch_mode="truncate_episodes",
            num_env_runners=resolve_env_runners(profile, default=4),
            num_envs_per_env_runner=profile.num_envs_per_env_runner,
            num_gpus_per_env_runner=profile.num_gpus_per_env_runner,
            sample_timeout_s=600.0,
        )
        .learners(num_gpus_per_learner=1 if profile.learner_device == "cuda" else 0)
    )
    algo = config.build_algo()
    key = "env_runners/num_env_steps_sampled_lifetime"

    def lifetime(result) -> float:
        node = result
        for part in key.split("/"):
            node = node[part]
        return float(node)

    for _ in range(2):
        result = algo.train()
    begin = lifetime(result)
    start = time.perf_counter()
    for _ in range(args.iterations):
        result = algo.train()
    elapsed = time.perf_counter() - start
    steps = lifetime(result) - begin
    algo.stop()
    return {
        "profile": profile.name,
        "num_env_runners": config.num_env_runners,
        "num_envs_per_env_runner": config.num_envs_per_env_runner,
        "batch_size": 8_192,
        "d_model": args.d_model,
        "iterations": args.iterations,
        "steps": steps,
        "seconds": elapsed,
        "steps_per_s": steps / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("env-old")
    p.add_argument("--steps", type=int, default=200_000)
    p = sub.add_parser("env-jax")
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--num-envs", type=int, default=4096)
    p = sub.add_parser("train-jax")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--updates-per-chunk", type=int, default=10)
    p.add_argument("--chunks", type=int, default=3)
    p = sub.add_parser("train-rllib")
    p.add_argument("--profile", default="cuda4090")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    fn = {"env-old": env_old, "env-jax": env_jax, "train-jax": train_jax, "train-rllib": train_rllib}
    print(json.dumps({"mode": args.mode, **fn[args.mode](args)}), flush=True)


if __name__ == "__main__":
    main()
