"""Plain PPO on RockSample[5,7], pure JAX (no RLlib), many seeds vmapped.

Same hyperparameters as ``rocksample_ppo_2026_09`` (clip 0.2, entropy 0.05,
adaptive KL, batch 8,192, minibatch 1,024, 4 epochs, d64 transformer), with
``NUM_SEEDS`` independent runs trained as one ``jax.vmap``. Requires the
``jax-cuda`` (GPU) or ``jax`` (CPU) dependency group.

Outputs: ``training_curves.jsonl`` (one row per update, per-seed lists),
``summary.json`` (final-tail return per seed) and final parameters in
``artifacts/final_params.npz``.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from functools import partial

import jax
import numpy as np
from harness.artifacts import RunArtifacts

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import ppo
from experiments.rocksample_jax_2026_10.meta import fitness_from_history

N, K, D_MODEL = 5, 7, 64
NUM_SEEDS = 16
SMOKE_NUM_SEEDS = 2
UPDATES_PER_CHUNK = 51
NUM_CHUNKS = 12  # 612 updates x 8,192 = 5,013,504 env steps per seed
SMOKE_UPDATES_PER_CHUNK = 1
SMOKE_NUM_CHUNKS = 2  # 2 updates x 1,024 = 2,048 env steps per seed
TAIL_FRACTION = 0.2
CURVE_METRICS = (
    "episode_return_mean",
    "episodes",
    "entropy",
    "kl",
    "kl_coeff",
    "illegal_action_rate",
    "policy_loss",
    "vf_loss",
)


def ppo_config(context) -> ppo.PPOConfig:
    if context.smoke:
        return ppo.PPOConfig(num_envs=8, num_steps=128, num_minibatches=2)
    return ppo.PPOConfig()


def run(context):
    if context.seed is None:
        raise ValueError("RockSample JAX PPO requires a resolved seed")
    config = ppo_config(context)
    num_seeds = SMOKE_NUM_SEEDS if context.smoke else NUM_SEEDS
    per_chunk = SMOKE_UPDATES_PER_CHUNK if context.smoke else UPDATES_PER_CHUNK
    num_chunks = SMOKE_NUM_CHUNKS if context.smoke else NUM_CHUNKS
    env_params = rs.RockSampleParams.from_instance(N, K)
    spec = ppo.make_spec(env_params, d_model=D_MODEL)

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "ppo": asdict(config),
            "instance": [N, K],
            "d_model": D_MODEL,
            "num_seeds": num_seeds,
            "num_updates": per_chunk * num_chunks,
            "env_steps_per_seed": per_chunk * num_chunks * config.batch_size,
            "seed": context.seed,
            "jax_device": str(jax.devices()[0]),
        },
    )

    keys = jax.random.split(jax.random.key(context.seed), num_seeds)
    states = jax.jit(jax.vmap(partial(ppo.init, config, env_params, spec)))(keys)
    train = jax.jit(jax.vmap(ppo.make_train_chunk(config, env_params, spec, per_chunk)))
    returns: list[np.ndarray] = []
    start = time.perf_counter()
    for chunk in range(num_chunks):
        states, metrics = train(states)
        metrics = jax.tree.map(np.asarray, metrics)
        for update in range(per_chunk):
            row = {
                "update": chunk * per_chunk + update,
                "env_steps": int(metrics["env_steps"][0, update]),
                "seconds": time.perf_counter() - start,
            }
            row.update(
                {name: metrics[name][:, update].tolist() for name in CURVE_METRICS}
            )
            outputs.append_jsonl("training_curves.jsonl", row)
        returns.append(metrics["episode_return_mean"])
        print(
            f"chunk {chunk + 1}/{num_chunks} env_steps/seed="
            f"{int(metrics['env_steps'][0, -1])} "
            f"return median={np.nanmedian(metrics['episode_return_mean'][:, -1]):.2f}",
            flush=True,
        )
    seconds = time.perf_counter() - start

    curves = np.concatenate(returns, axis=1)
    tail = [fitness_from_history(curve, TAIL_FRACTION) for curve in curves]
    total_steps = num_seeds * curves.shape[1] * config.batch_size
    outputs.write_json(
        "summary.json",
        {
            "tail_return_per_seed": tail,
            "tail_return_median": float(np.nanmedian(tail)),
            "tail_return_mean": float(np.nanmean(tail)),
            "tail_fraction": TAIL_FRACTION,
            "seconds_including_compile": seconds,
            "total_env_steps": total_steps,
            "env_steps_per_s_total": total_steps / seconds,
        },
    )
    flat = {
        jax.tree_util.keystr(path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(states.params)
    }
    np.savez(context.artifacts_dir / "final_params.npz", **flat)
    return tail
