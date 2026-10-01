"""ES outer loop over LPO drift parameters with vmapped pure-JAX inner PPO.

Every generation trains the whole ES population (antithetic pairs plus the
centre) as one ``jax.vmap`` over fresh inner PPO runs; antithetic pair members
share an inner PRNG key (common random numbers).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from harness.artifacts import RunArtifacts
from harness.context import RunContext

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import lpo, ppo
from experiments.rocksample_jax_2026_10.es import ESState, OpenES

META_STATE_FILENAME = "meta_state.json"
META_PROGRESS_FILENAME = "meta_progress.jsonl"


@dataclass(frozen=True)
class InnerRecipe:
    """One inner training run: fixed-layout RockSample + transformer + LPO PPO."""

    ppo: ppo.PPOConfig
    total_env_steps: int
    n: int = 5
    k: int = 7
    d_model: int = 64

    @property
    def num_updates(self) -> int:
        return math.ceil(self.total_env_steps / self.ppo.batch_size)


@dataclass(frozen=True)
class MetaRecipe:
    drift: lpo.DriftSpec
    es: OpenES
    generations: int
    fitness_tail_fraction: float
    evaluate_center: bool = True


def inner_seed(meta_seed: int, generation: int, slot: int) -> int:
    """Common random numbers: both members of an antithetic pair share a seed."""
    return int(
        np.random.default_rng([meta_seed, generation, slot, 7]).integers(0, 2**31 - 1)
    )


def fitness_from_history(returns: Sequence[float], tail_fraction: float) -> float:
    """Mean undiscounted return over the final ``tail_fraction`` of updates."""
    finite = [float(v) for v in returns if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    count = max(1, math.ceil(len(finite) * tail_fraction))
    return float(np.mean(finite[-count:]))


class PopulationTrainer:
    """Jitted ``(seeds, metas) -> per-update return curves`` for a population."""

    def __init__(self, inner: InnerRecipe, drift: lpo.DriftSpec):
        env_params = rs.RockSampleParams.from_instance(inner.n, inner.k)
        spec = ppo.make_spec(env_params, d_model=inner.d_model)
        init = partial(ppo.init, inner.ppo, env_params, spec)
        chunk = ppo.make_train_chunk(
            inner.ppo, env_params, spec, inner.num_updates, lpo.objective(drift)
        )

        def train(keys, metas):
            states = jax.vmap(init)(keys, metas)
            _, metrics = jax.vmap(chunk)(states)
            return {
                name: metrics[name]
                for name in ("episode_return_mean", "env_steps", "entropy", "kl")
            }

        self._train = jax.jit(train)

    def __call__(
        self, seeds: Sequence[int], metas: np.ndarray
    ) -> dict[str, np.ndarray]:
        keys = jax.vmap(jax.random.key)(jnp.asarray(seeds, jnp.uint32))
        out = self._train(keys, jnp.asarray(metas, jnp.float32))
        return {name: np.asarray(value) for name, value in out.items()}


def load_state(context: RunContext, meta: MetaRecipe) -> ESState:
    """Resume from ``--resume-from <meta_state.json>`` or start at PPO."""
    if context.resume_from is not None:
        payload = json.loads(Path(context.resume_from).read_text())
        state = ESState.from_dict(payload["es_state"])
        if state.mean.size != meta.drift.num_params:
            raise ValueError("resumed meta-state does not match the drift shape")
        return state
    return ESState(mean=meta.drift.initial_params(context.seed))


def run_generation(
    context: RunContext,
    meta: MetaRecipe,
    trainer: PopulationTrainer,
    state: ESState,
) -> tuple[ESState, dict[str, Any]]:
    noise = meta.es.noise(state, context.seed)
    population = meta.es.candidates(state, noise)
    seeds = [
        inner_seed(context.seed, state.generation, index // 2)
        for index in range(len(population))
    ]
    metas = population
    if meta.evaluate_center:
        seeds.append(inner_seed(context.seed, state.generation, meta.es.num_pairs))
        metas = np.concatenate([population, state.mean[None]], axis=0)

    start = time.perf_counter()
    curves = trainer(seeds, metas)
    seconds = time.perf_counter() - start
    scores = np.array(
        [
            fitness_from_history(curve, meta.fitness_tail_fraction)
            for curve in curves["episode_return_mean"]
        ]
    )
    fitness = scores[: len(population)]
    if not np.all(np.isfinite(fitness)):
        raise RuntimeError("an inner run reported no finite episode return")
    next_state = meta.es.step(state, noise, fitness)
    row = {
        "generation": state.generation,
        "fitness": fitness.tolist(),
        "fitness_mean": float(fitness.mean()),
        "fitness_max": float(fitness.max()),
        "center_return": float(scores[-1]) if meta.evaluate_center else None,
        "inner_env_steps": int(curves["env_steps"][0, -1]),
        "inner_runs": len(seeds),
        "seconds": seconds,
        "update_norm": float(np.linalg.norm(next_state.mean - state.mean)),
        "w_ppo": float(next_state.mean[-2]),
        "entropy_coeff": float(math.exp(next_state.mean[-1])),
    }
    return next_state, row


def run_meta_training(
    context: RunContext, inner: InnerRecipe, meta: MetaRecipe
) -> ESState:
    """ES meta-training loop; writes compact progress after every generation."""
    if context.seed is None:
        raise ValueError("meta-RL study requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "inner": {
                "ppo": asdict(inner.ppo),
                "total_env_steps": inner.total_env_steps,
                "num_updates": inner.num_updates,
                "instance": [inner.n, inner.k],
                "d_model": inner.d_model,
            },
            "drift": {**asdict(meta.drift), "num_params": meta.drift.num_params},
            "es": asdict(meta.es),
            "generations": meta.generations,
            "fitness_tail_fraction": meta.fitness_tail_fraction,
            "evaluate_center": meta.evaluate_center,
            "seed": context.seed,
            "resume_from": str(context.resume_from) if context.resume_from else None,
            "jax_device": str(jax.devices()[0]),
        },
    )
    state = load_state(context, meta)
    trainer = PopulationTrainer(inner, meta.drift)
    while state.generation < meta.generations:
        state, row = run_generation(context, meta, trainer, state)
        outputs.append_jsonl(META_PROGRESS_FILENAME, row)
        outputs.write_json(
            META_STATE_FILENAME,
            {"es_state": state.to_dict(), "drift_hidden": meta.drift.hidden},
        )
    return state
