"""Inner PPO-family recipe and ES outer loop for the RockSample meta-RL study."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.rocksample import RockSampleEnv
from experiments.rocksample_meta_rl_2026_09.es import ESState, OpenES
from experiments.rocksample_meta_rl_2026_09.learned_objective import (
    CANDIDATE_KEY,
    DriftSpec,
    LearnedDriftPPOLearner,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import (
    PROFILES,
    configure_hardware,
    resolve_env_runners,
    shutdown_ray_if_owned,
)
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig

RETURN_METRIC = "env_runners/episode_return_mean"
STEPS_METRIC = "env_runners/num_env_steps_sampled_lifetime"
META_STATE_FILENAME = "meta_state.json"
META_PROGRESS_FILENAME = "meta_progress.jsonl"


@dataclass(frozen=True, slots=True)
class InnerRecipe:
    """One inner training run: RockSample + transformer + learned-drift PPO."""

    env_config: Mapping[str, Any]
    d_model: int
    total_env_steps: int
    train_batch_size: int
    minibatch_size: int
    num_env_runners: int
    gpus_per_trial: float
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_: float = 0.95
    num_epochs: int = 4
    vf_loss_coeff: float = 0.5
    vf_clip_param: float = 1_000_000.0
    grad_clip: float = 0.5
    n_layers: int = 3
    n_heads: int = 4
    context_len: int = 32


@dataclass(frozen=True, slots=True)
class MetaRecipe:
    """Outer-loop budget: ES over the learned drift's meta-parameters."""

    drift: DriftSpec
    es: OpenES
    generations: int
    fitness_tail_fraction: float
    evaluate_center: bool = True


def build_inner_config(
    context: RunContext,
    inner: InnerRecipe,
    candidates: Sequence[Mapping[str, Any]],
    drift: DriftSpec,
) -> PPOConfig:
    """One PPOConfig whose Tune grid spans every candidate objective."""
    profile = context.hardware or PROFILES["cpu"]
    num_runners = (
        0
        if context.smoke
        else min(
            inner.num_env_runners,
            resolve_env_runners(profile, default=inner.num_env_runners),
        )
    )
    grid = [
        dict(candidate, hidden=drift.hidden, clip_param=drift.clip_param)
        for candidate in candidates
    ]
    return (
        PPOConfig()
        .environment(RockSampleEnv, env_config=dict(inner.env_config))
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
        .training(
            lr=inner.lr,
            gamma=inner.gamma,
            lambda_=inner.lambda_,
            clip_param=drift.clip_param,
            vf_loss_coeff=inner.vf_loss_coeff,
            vf_clip_param=inner.vf_clip_param,
            entropy_coeff=0.0,
            grad_clip=inner.grad_clip,
            grad_clip_by="global_norm",
            train_batch_size_per_learner=inner.train_batch_size,
            minibatch_size=inner.minibatch_size,
            num_epochs=inner.num_epochs,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=TransformerModelConfig(
                    d_model=inner.d_model,
                    n_layers=inner.n_layers,
                    n_heads=inner.n_heads,
                    context_len=inner.context_len,
                ).to_dict(),
            )
        )
        .debugging(seed=tune.sample_from(candidate_seed))
        .env_runners(
            batch_mode="truncate_episodes",
            num_env_runners=num_runners,
            num_envs_per_env_runner=1
            if context.smoke
            else profile.num_envs_per_env_runner,
            num_gpus_per_env_runner=0
            if context.smoke
            else profile.num_gpus_per_env_runner,
            sample_timeout_s=600.0,
        )
        .learners(
            learner_class=LearnedDriftPPOLearner,
            learner_config_dict={CANDIDATE_KEY: tune.grid_search(grid)},
            num_gpus_per_learner=(
                inner.gpus_per_trial if profile.learner_device == "cuda" else 0
            ),
        )
    )


def candidate_seed(spec) -> int:
    """Tune dependency: each trial's RLlib seed comes from its candidate."""
    return int(spec.config["learner_config_dict"][CANDIDATE_KEY]["seed"])


def inner_seed(meta_seed: int, generation: int, slot: int) -> int:
    """Common random numbers: both members of an antithetic pair share a seed."""
    return int(
        np.random.default_rng([meta_seed, generation, slot, 7]).integers(0, 2**31 - 1)
    )


def fitness_from_history(returns: Sequence[float], tail_fraction: float) -> float:
    """Mean undiscounted return over the final ``tail_fraction`` of iterations."""
    finite = [
        float(value)
        for value in returns
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return float("nan")
    count = max(1, math.ceil(len(finite) * tail_fraction))
    return float(np.mean(finite[-count:]))


def _trial_returns(result) -> list[float]:
    frame = result.metrics_dataframe
    if frame is not None and RETURN_METRIC in frame:
        return frame[RETURN_METRIC].tolist()
    metrics = result.metrics or {}
    value = metrics.get("env_runners", {}).get("episode_return_mean")
    return [] if value is None else [value]


def _trial_steps(result) -> float | None:
    metrics = result.metrics or {}
    return metrics.get("env_runners", {}).get("num_env_steps_sampled_lifetime")


def _generation_context(context: RunContext, generation: int) -> RunContext:
    label = f"gen_{generation:04d}"
    return replace(
        context,
        results_dir=context.results_dir / "generations" / label,
        artifacts_dir=context.artifacts_dir / "generations" / label,
        resume_from=None,
    )


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
    inner: InnerRecipe,
    meta: MetaRecipe,
    state: ESState,
) -> tuple[ESState, dict[str, Any]]:
    """Train every candidate objective once and take one ES step."""
    noise = meta.es.noise(state, context.seed)
    population = meta.es.candidates(state, noise)
    candidates = [
        {
            "index": index,
            "role": "perturbed",
            "seed": inner_seed(context.seed, state.generation, index // 2),
            "params": row.tolist(),
        }
        for index, row in enumerate(population)
    ]
    if meta.evaluate_center:
        candidates.append(
            {
                "index": len(population),
                "role": "center",
                "seed": inner_seed(context.seed, state.generation, meta.es.num_pairs),
                "params": state.mean.tolist(),
            }
        )

    generation_context = _generation_context(context, state.generation)
    RunArtifacts.from_context(generation_context).prepare()
    result_grid = run_tune(
        build_inner_config(generation_context, inner, candidates, meta.drift),
        generation_context,
        stop={STEPS_METRIC: inner.total_env_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=None, checkpoint_frequency=0, checkpoint_at_end=False
            ),
        },
    )

    by_index: dict[int, tuple[float, float | None]] = {}
    for result in result_grid:
        if result.error is not None:
            raise RuntimeError(f"inner trial failed: {result.error}")
        candidate = result.config["learner_config_dict"][CANDIDATE_KEY]
        by_index[int(candidate["index"])] = (
            fitness_from_history(_trial_returns(result), meta.fitness_tail_fraction),
            _trial_steps(result),
        )
    if len(by_index) != len(candidates):
        raise RuntimeError(
            f"expected {len(candidates)} inner trials, got {len(by_index)}"
        )

    fitness = np.array([by_index[index][0] for index in range(len(population))])
    if not np.all(np.isfinite(fitness)):
        raise RuntimeError("an inner run reported no finite episode return")
    next_state = meta.es.step(state, noise, fitness)
    center = by_index.get(len(population))
    row = {
        "generation": state.generation,
        "fitness": fitness.tolist(),
        "fitness_mean": float(fitness.mean()),
        "fitness_max": float(fitness.max()),
        "center_return": center[0] if center else None,
        "inner_env_steps": [by_index[index][1] for index in sorted(by_index)],
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
            "inner": asdict(inner),
            "drift": {
                "hidden": meta.drift.hidden,
                "clip_param": meta.drift.clip_param,
                "initial_entropy_coeff": meta.drift.initial_entropy_coeff,
                "num_params": meta.drift.num_params,
            },
            "es": asdict(meta.es),
            "generations": meta.generations,
            "fitness_tail_fraction": meta.fitness_tail_fraction,
            "evaluate_center": meta.evaluate_center,
            "seed": context.seed,
            "resume_from": str(context.resume_from) if context.resume_from else None,
        },
    )
    state = load_state(context, meta)
    started_ray = (
        configure_hardware(context.hardware) if context.hardware is not None else False
    )
    try:
        while state.generation < meta.generations:
            state, row = run_generation(context, inner, meta, state)
            outputs.append_jsonl(META_PROGRESS_FILENAME, row)
            outputs.write_json(
                META_STATE_FILENAME,
                {"es_state": state.to_dict(), "drift_hidden": meta.drift.hidden},
            )
    finally:
        shutdown_ray_if_owned(started_ray)
    return state
