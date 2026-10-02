"""PPO arms at the largest batch the GPU fits, one seed at a time (no vmap).

Shared runner for the ``ppo_bmax_*`` leaves. Each arm is a full ``PPOConfig``
plus a transformer width; every arm gets the same env-step budget, so the
number of updates is ``ceil(budget / batch_size)``. Seeds run sequentially
inside one process (one compile per arm) because a single seed already fills
the GPU at these batch sizes. Outputs: ``training_curves.jsonl`` (one row per
arm, seed and update), ``summary.json`` (tail/final return per arm and seed),
and the final trainer state of every seed in ignored
``artifacts/<arm>/seed<i>.pkl`` so a later leaf can continue the run
(``Arm.resume_from``) instead of retraining. ``SweepRecipe.num_checkpoints``
adds log-spaced intermediate checkpoints (params, optimizer, KL coeff, key;
no rollout, so a few MB each) under ``artifacts/<arm>/seed<i>/update<u>.pkl``.
An arm that fails (typically out of memory) is recorded and skipped.
"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import numpy as np
from harness.artifacts import RunArtifacts

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import ppo
from experiments.rocksample_jax_2026_10.baseline import (
    CURVE_METRICS,
    TAIL_FRACTION,
    K,
    N,
)
from experiments.rocksample_jax_2026_10.meta import fitness_from_history


@dataclass(frozen=True)
class Arm:
    name: str
    ppo: ppo.PPOConfig
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    context_len: int = 32
    note: str = ""
    resume_from: str | None = None
    """Directory holding ``seed<i>.pkl`` trainer states to continue from."""

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
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "context_len": self.context_len,
            "batch_size": self.ppo.batch_size,
            "minibatch_size": self.minibatch_size,
            "num_updates": updates,
            "gradient_steps_per_update": self.ppo.num_epochs * self.ppo.num_minibatches,
            "env_steps_per_seed": updates * self.ppo.batch_size,
            "resume_from": self.resume_from,
        }


@dataclass(frozen=True)
class SweepRecipe:
    arms: tuple[Arm, ...]
    env_steps_per_seed: int
    num_seeds: int
    num_keys: int | None = None
    """Split the root key this many ways (default ``num_seeds``) so a recipe
    running fewer seeds reuses the same per-seed keys as a wider one."""
    first_seed: int = 0
    """Seed index to start at, so a follow-up can run the remaining seeds."""
    num_checkpoints: int = 1
    """Trainer states saved per seed at log-spaced updates (final always
    included); 1 keeps only the final state."""

    def __post_init__(self) -> None:
        if self.first_seed < 0:
            raise ValueError("first_seed must be >= 0")
        if self.num_checkpoints < 1:
            raise ValueError("num_checkpoints must be >= 1")
        if (self.num_keys or self.num_seeds) < self.first_seed + self.num_seeds:
            raise ValueError("num_keys must be >= first_seed + num_seeds")
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names: {names}")


def checkpoint_updates(num_updates: int, num_checkpoints: int) -> tuple[int, ...]:
    """1-indexed update counts at which to checkpoint: ``min(num_checkpoints,
    num_updates)`` distinct values, geometrically spaced from 1 to
    ``num_updates`` (ties bumped to the next free update), last = final."""
    n = min(num_checkpoints, num_updates)
    chosen: list[int] = []
    for i in range(n):
        target = round(num_updates ** (i / max(n - 1, 1))) if n > 1 else num_updates
        chosen.append(max(target, chosen[-1] + 1 if chosen else 1))
    chosen[-1] = num_updates
    return tuple(chosen)


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


def save_state(path: Path, state: ppo.RunnerState, *, with_rollout: bool = True) -> None:
    """Pickle ``state`` on the host. ``with_rollout=False`` drops the env
    state / KV cache / history (the bulk of the ~0.9 GB at batch 1M), leaving
    a few-MB checkpoint that can be evaluated or probed but not resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = state._replace(key=jax.random.key_data(state.key))
    if not with_rollout:
        state = state._replace(rollout=None)
    host = jax.tree.map(np.asarray, state)
    with path.open("wb") as f:
        pickle.dump(host, f)


def load_state(path: Path) -> ppo.RunnerState:
    with path.open("rb") as f:
        state = jax.tree.map(jax.numpy.asarray, pickle.load(f))
    if state.rollout is None:
        raise ValueError(f"{path} is a rollout-free checkpoint and cannot be resumed")
    return state._replace(key=jax.random.wrap_key_data(state.key))


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
            "first_seed": recipe.first_seed,
            "num_checkpoints": recipe.num_checkpoints,
            "checkpoint_updates": {
                arm.name: checkpoint_updates(arm.num_updates(budget), recipe.num_checkpoints)
                for arm in recipe.arms
            },
            "tail_fraction": TAIL_FRACTION,
            "seed": context.seed,
            "jax_device": str(jax.devices()[0]),
        },
    )
    # Same seed keys for every arm (common random numbers across arms).
    keys = jax.random.split(jax.random.key(context.seed), recipe.num_keys or recipe.num_seeds)

    summary: dict = {"env_steps_per_seed": budget, "arms": {}}
    for arm in recipe.arms:
        updates = arm.num_updates(budget)
        ckpt_updates = set(checkpoint_updates(updates, recipe.num_checkpoints))
        spec = ppo.make_spec(
            env_params,
            d_model=arm.d_model,
            n_layers=arm.n_layers,
            n_heads=arm.n_heads,
            context_len=arm.context_len,
        )
        init = jax.jit(partial(ppo.init, arm.ppo, env_params, spec))
        step = jax.jit(ppo.make_train_chunk(arm.ppo, env_params, spec, 1))
        seeds: list[dict] = []
        error = None
        print(
            f"arm {arm.name}: batch={arm.ppo.batch_size} mb={arm.minibatch_size} "
            f"d={arm.d_model} updates={updates}",
            flush=True,
        )
        for seed_index in range(recipe.first_seed, recipe.first_seed + recipe.num_seeds):
            returns: list[float] = []
            checkpoints: list[dict] = []
            start = time.perf_counter()
            try:
                if arm.resume_from is None:
                    state = init(keys[seed_index])
                    first_update = 0
                else:
                    state = load_state(Path(arm.resume_from) / f"seed{seed_index}.pkl")
                    first_update = int(state.env_steps) // arm.ppo.batch_size
                for update in range(first_update, updates):
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
                    if update + 1 in ckpt_updates and update + 1 < updates:
                        path = outputs.artifacts_dir / arm.name / f"seed{seed_index}" / f"update{update + 1:03d}.pkl"
                        save_state(path, state, with_rollout=False)
                        checkpoints.append({
                            "update": update + 1,
                            "env_steps": row["env_steps"],
                            "path": str(path.relative_to(outputs.artifacts_dir)),
                        })
                jax.block_until_ready(state)
                path = outputs.artifacts_dir / arm.name / f"seed{seed_index}.pkl"
                save_state(path, state)
                checkpoints.append({
                    "update": updates,
                    "env_steps": int(state.env_steps),
                    "path": str(path.relative_to(outputs.artifacts_dir)),
                })
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
                    "checkpoints": checkpoints,
                    "seconds_including_compile": time.perf_counter() - start,
                }
            )
        summary["arms"][arm.name] = _arm_summary(arm, budget, seeds, error)
        outputs.write_json("summary.json", summary)
        jax.clear_caches()
    return summary
