"""Plain PPO on RockSample[5,7], pure JAX (no RLlib), many seeds vmapped.

Shared runner for the ``ppo_*`` leaves. Outputs: ``training_curves.jsonl``
(one row per update, per-seed lists), ``summary.json`` (final-tail return per
seed) and final parameters in ``artifacts/final_params.npz``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import partial

import jax
import numpy as np
from harness.artifacts import RunArtifacts

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import ppo
from experiments.rocksample_jax_2026_10.meta import fitness_from_history

N, K, D_MODEL = 5, 7, 64
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
SMOKE_CONFIG = ppo.PPOConfig(num_envs=8, num_steps=128, num_minibatches=2)


@dataclass(frozen=True)
class BaselineRecipe:
    ppo: ppo.PPOConfig
    num_seeds: int
    updates_per_chunk: int
    num_chunks: int

    @property
    def env_steps_per_seed(self) -> int:
        return self.updates_per_chunk * self.num_chunks * self.ppo.batch_size


def smoke_recipe() -> BaselineRecipe:
    """2 seeds x 2 updates x 1,024 = 2,048 env steps per seed."""
    return BaselineRecipe(SMOKE_CONFIG, num_seeds=2, updates_per_chunk=1, num_chunks=2)


def run(context, recipe: BaselineRecipe):
    if context.seed is None:
        raise ValueError("RockSample JAX PPO requires a resolved seed")
    config = recipe.ppo
    num_seeds = recipe.num_seeds
    per_chunk = recipe.updates_per_chunk
    num_chunks = recipe.num_chunks
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
            "minibatch_size": config.batch_size // config.num_minibatches,
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
