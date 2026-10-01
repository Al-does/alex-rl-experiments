"""PPO arms at the largest batch the GPU fits, one seed at a time (no vmap).

Shared runner for the ``ppo_bmax_*`` leaves. Each arm is a full ``PPOConfig``
plus a transformer width; every arm gets the same env-step budget, so the
number of updates is ``ceil(budget / batch_size)``. Seeds run sequentially
inside one process (one compile per arm) because a single seed already fills
the GPU at these batch sizes. Outputs: ``training_curves.jsonl`` (one row per
arm, seed and update), ``summary.json`` (tail/final return per arm and seed).
An arm that fails (typically out of memory) is recorded and skipped.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from functools import partial

import jax
import numpy as np
from harness.artifacts import RunArtifacts

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import ppo
from experiments.rocksample_jax_2026_10.baseline import CURVE_METRICS, N, K, TAIL_FRACTION
from experiments.rocksample_jax_2026_10.meta import fitness_from_history


@dataclass(frozen=True)
class Arm:
    name: str
    ppo: ppo.PPOConfig
    d_model: int = 64
    note: str = ""

    @property
    def minibatch_size(self) -> int:
        return self.ppo.batch_size // self.ppo.num_minibatches

    def num_updates(self, env_steps: int) -> int:
        return math.ceil(env_steps / self.ppo.batch_size)

    def describe(self, env_steps: int) -> dict:
        updates = self.num_updates(env_steps)
        return {
            "name": self.name,
            "note": self.note,
            "ppo": asdict(self.ppo),
            "d_model": self.d_model,
            "batch_size": self.ppo.batch_size,
            "minibatch_size": self.minibatch_size,
            "num_updates": updates,
            "gradient_steps_per_update": self.ppo.num_epochs * self.ppo.num_minibatches,
            "env_steps_per_seed": updates * self.ppo.batch_size,
        }


@dataclass(frozen=True)
class SweepRecipe:
    arms: tuple[Arm, ...]
    env_steps_per_seed: int
    num_seeds: int

    def __post_init__(self) -> None:
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names: {names}")


def smoke_recipe() -> SweepRecipe:
    """2 arms x 1 seed x 2 updates x 1,024 = 2,048 env steps each."""
    small = ppo.PPOConfig(num_envs=8, num_steps=128, num_minibatches=2)
    return SweepRecipe(
        arms=(
            Arm("smoke_d16", small, d_model=16),
            Arm("smoke_d32_lr1e-3", ppo.PPOConfig(
                num_envs=8, num_steps=128, num_minibatches=2, lr=1e-3,
                vf_coeff=1.0, vf_clip=10.0, use_kl_loss=False,
            ), d_model=32),
        ),
        env_steps_per_seed=2_048,
        num_seeds=1,
    )


def _arm_summary(arm: Arm, env_steps: int, seeds: list[dict], error: str | None):
    tails = [s["tail_return"] for s in seeds]
    return {
        **arm.describe(env_steps),
        "seeds": seeds,
        "tail_return_per_seed": tails,
        "tail_return_median": float(np.nanmedian(tails)) if tails else float("nan"),
        "tail_return_mean": float(np.nanmean(tails)) if tails else float("nan"),
        "final_return_per_seed": [s["final_return"] for s in seeds],
        "error": error,
    }


def run(context, recipe: SweepRecipe):
    if context.seed is None:
        raise ValueError("RockSample JAX PPO requires a resolved seed")
    budget = recipe.env_steps_per_seed
    env_params = rs.RockSampleParams.from_instance(N, K)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "arms": [arm.describe(budget) for arm in recipe.arms],
            "instance": [N, K],
            "env_steps_per_seed": budget,
            "num_seeds": recipe.num_seeds,
            "tail_fraction": TAIL_FRACTION,
            "seed": context.seed,
            "jax_device": str(jax.devices()[0]),
        },
    )
    # Same seed keys for every arm (common random numbers across arms).
    keys = jax.random.split(jax.random.key(context.seed), recipe.num_seeds)

    summary: dict = {"env_steps_per_seed": budget, "arms": {}}
    for arm in recipe.arms:
        updates = arm.num_updates(budget)
        spec = ppo.make_spec(env_params, d_model=arm.d_model)
        init = jax.jit(partial(ppo.init, arm.ppo, env_params, spec))
        step = jax.jit(ppo.make_train_chunk(arm.ppo, env_params, spec, 1))
        seeds: list[dict] = []
        error = None
        print(
            f"arm {arm.name}: batch={arm.ppo.batch_size} mb={arm.minibatch_size} "
            f"d={arm.d_model} updates={updates}",
            flush=True,
        )
        for seed_index in range(recipe.num_seeds):
            returns: list[float] = []
            start = time.perf_counter()
            try:
                state = init(keys[seed_index])
                for update in range(updates):
                    state, metrics = step(state)
                    metrics = {k: float(np.asarray(v)[0]) for k, v in metrics.items()}
                    row = {
                        "arm": arm.name,
                        "seed_index": seed_index,
                        "update": update,
                        "env_steps": int(metrics["env_steps"]),
                        "seconds": time.perf_counter() - start,
                    }
                    row.update({name: metrics[name] for name in CURVE_METRICS})
                    outputs.append_jsonl("training_curves.jsonl", row)
                    returns.append(metrics["episode_return_mean"])
                    print(
                        f"  seed {seed_index} update {update + 1}/{updates} "
                        f"steps={row['env_steps']} return={metrics['episode_return_mean']:.2f} "
                        f"kl={metrics['kl']:.4f} ent={metrics['entropy']:.3f} "
                        f"t={row['seconds']:.0f}s",
                        flush=True,
                    )
                jax.block_until_ready(state)
                del state
            except jax.errors.JaxRuntimeError as exc:
                error = str(exc).splitlines()[0][:300]
                print(f"  arm {arm.name} failed: {error}", flush=True)
                break
            seeds.append(
                {
                    "seed_index": seed_index,
                    "tail_return": fitness_from_history(returns, TAIL_FRACTION),
                    "final_return": returns[-1],
                    "returns": returns,
                    "seconds_including_compile": time.perf_counter() - start,
                }
            )
        summary["arms"][arm.name] = _arm_summary(arm, budget, seeds, error)
        outputs.write_json("summary.json", summary)
        jax.clear_caches()
    return summary
