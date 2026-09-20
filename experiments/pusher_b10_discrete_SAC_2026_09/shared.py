"""Long-horizon Pusher-B b=0.1 discrete-SAC recipes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import partial
from numbers import Real
from pathlib import Path

from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
)
from experiments.pusher_b.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    PRESETS,
    TOKEN_COUNT,
    environment_config,
)
from experiments.pusher_b10_discrete_SAC_2026_09.model import (
    PusherSACCatalog,
    PusherSharedTrunkSAC,
    PusherSplitSAC,
    SharedTrunkSACTorchLearner,
)
from experiments.storage.training_curves import write_training_curves
from experiments.two_factor_reward_state_SAC_cycle_1.shared import (
    EightMinibatchSAC,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


ARCHITECTURES = ("shared_trunk", "split_transformers")
TOTAL_ENV_STEPS = 300_000_000
SMOKE_ENV_STEPS = 128
TRAIN_BATCH_SIZE = 8_192
LEARNER_MINIBATCH_COUNT = 8
LEARNER_MINIBATCH_SIZE = TRAIN_BATCH_SIZE // LEARNER_MINIBATCH_COUNT
SMOKE_BATCH_SIZE = 64
LEARNING_STARTS = 50_000
SMOKE_LEARNING_STARTS = 32
REPLAY_CAPACITY = 2_000_000
SMOKE_REPLAY_CAPACITY = 1_024
TRAINING_INTENSITY = 1.0
ACTOR_LEARNING_RATE = 3e-5
CRITIC_LEARNING_RATE = 3e-4
SHARED_ENCODER_LEARNING_RATE = 1e-4
ALPHA_LEARNING_RATE = 3e-4
TARGET_ENTROPY_FRACTION = 0.6
TARGET_ENTROPY = TARGET_ENTROPY_FRACTION * math.log(TOKEN_COUNT)
LOG_CHECKPOINT_END_ENV_STEPS = 30_000_000
CHECKPOINT_INTERVAL_ENV_STEPS = 25_000_000
NUM_ENVS_PER_ENV_RUNNER = 8
MODEL_CONFIG = {
    **FactoredReproductionModelConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_mlp=512,
        context_length=CONTEXT_LENGTH,
        max_seq_len=CONTEXT_LENGTH,
        activation="gated_gelu",
        normalization="rms_norm",
        positional_embedding="rope",
        attention_implementation="sdpa",
        training_sequence_mode="sliding_window",
    ).to_dict(),
    "head_fcnet_hiddens": [],
}


def sac_environment_config() -> dict[str, object]:
    config = environment_config("b10")
    config["observation"] = {
        "token": {"depth": CONTEXT_LENGTH},
        "action": None,
    }
    return config


def _metric(metrics: Mapping[str, object], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, Real):
        return float(direct)
    value: object = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if not isinstance(value, Real):
        return None
    return float(value)


def checkpoint_decision(
    *,
    training_iteration: int,
    env_steps: int,
    records: list[dict[str, object]],
) -> tuple[str, int] | None:
    if training_iteration <= 0 or env_steps <= 0:
        return None
    if env_steps <= LOG_CHECKPOINT_END_ENV_STEPS:
        if training_iteration.bit_count() != 1:
            return None
        if any(
            int(record["training_iteration"]) == training_iteration
            for record in records
        ):
            return None
        return "log", env_steps

    target = LOG_CHECKPOINT_END_ENV_STEPS + (
        (env_steps - LOG_CHECKPOINT_END_ENV_STEPS)
        // CHECKPOINT_INTERVAL_ENV_STEPS
    ) * CHECKPOINT_INTERVAL_ENV_STEPS
    if any(
        record.get("target_env_steps") is not None
        and int(record["target_env_steps"]) >= target
        for record in records
    ):
        return None
    return "interval", target


def _save_scheduled_checkpoint(
    *,
    algorithm: Algorithm,
    result: Mapping[str, object],
    checkpoint_root: str,
    **_: object,
) -> None:
    iteration_value = _metric(result, "training_iteration")
    steps_value = _metric(
        result,
        "env_runners/num_env_steps_sampled_lifetime",
    )
    if iteration_value is None or steps_value is None:
        return
    iteration = int(iteration_value)
    steps = int(steps_value)
    root = Path(checkpoint_root)
    index_path = root / "index.json"
    records: list[dict[str, object]] = []
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("checkpoint index must contain a JSON object")
        stored_records = payload.get("checkpoints", [])
        if not isinstance(stored_records, list) or not all(
            isinstance(record, dict) for record in stored_records
        ):
            raise ValueError("checkpoint index records must be JSON objects")
        records = [dict(record) for record in stored_records]

    decision = checkpoint_decision(
        training_iteration=iteration,
        env_steps=steps,
        records=records,
    )
    if decision is None:
        return
    phase, target = decision
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (
        f"{phase}_iteration_{iteration:06d}_steps_{steps:09d}"
    )
    saved = Path(algorithm.save_to_path(str(destination)))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": iteration,
            "env_steps": steps,
            "phase": phase,
            "target_env_steps": target,
        }
    )
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"checkpoints": records}, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(index_path)


def build_config(
    context: RunContext,
    *,
    architecture: str,
) -> SACConfig:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    profile = context.hardware or PROFILES["cpu"]
    module_class = (
        PusherSharedTrunkSAC
        if architecture == "shared_trunk"
        else PusherSplitSAC
    )
    config = (
        SACConfig(algo_class=EightMinibatchSAC)
        .environment(HMMEnv, env_config=sac_environment_config())
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            gamma=0.0,
            n_step=1,
            twin_q=True,
            tau=0.005,
            actor_lr=ACTOR_LEARNING_RATE,
            critic_lr=CRITIC_LEARNING_RATE,
            alpha_lr=ALPHA_LEARNING_RATE,
            target_entropy=TARGET_ENTROPY,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            training_intensity=TRAINING_INTENSITY,
            num_steps_sampled_before_learning_starts=(
                SMOKE_LEARNING_STARTS
                if context.smoke
                else LEARNING_STARTS
            ),
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": (
                    SMOKE_REPLAY_CAPACITY
                    if context.smoke
                    else REPLAY_CAPACITY
                ),
                "alpha": 0.6,
                "beta": 0.4,
            },
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=module_class,
                catalog_class=PusherSACCatalog,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _save_initial_checkpoint,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
                ),
            ),
            on_train_result=partial(
                _save_scheduled_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "scheduled_checkpoints"
                ),
            ),
        )
        .debugging(seed=context.seed)
        .env_runners(
            num_env_runners=(
                0
                if context.smoke
                else resolve_env_runners(profile, default=16)
            ),
            num_envs_per_env_runner=(
                1 if context.smoke else NUM_ENVS_PER_ENV_RUNNER
            ),
            num_gpus_per_env_runner=0,
            rollout_fragment_length=(
                1 if context.smoke else EPISODE_LENGTH
            ),
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            ),
        )
        .reporting(
            min_sample_timesteps_per_iteration=(
                64 if context.smoke else TRAIN_BATCH_SIZE
            ),
            min_time_s_per_iteration=0 if context.smoke else 1,
        )
    )
    if architecture == "shared_trunk":
        config = config.learners(
            learner_class=SharedTrunkSACTorchLearner,
            learner_config_dict={
                "shared_encoder_learning_rate": (
                    SHARED_ENCODER_LEARNING_RATE
                )
            },
        )
    return config


def _fixed_checkpoint_targets() -> list[int]:
    return list(
        range(
            LOG_CHECKPOINT_END_ENV_STEPS + CHECKPOINT_INTERVAL_ENV_STEPS,
            TOTAL_ENV_STEPS,
            CHECKPOINT_INTERVAL_ENV_STEPS,
        )
    )


def resolved_recipe(
    context: RunContext,
    *,
    architecture: str,
) -> dict[str, object]:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    shared = architecture == "shared_trunk"
    return {
        "study": "pusher_b10_discrete_SAC_2026_09",
        "condition": architecture,
        "preset": "b10",
        "parameters": PRESETS["b10"],
        "environment": sac_environment_config(),
        "algorithm": "discrete SAC",
        "architecture": (
            "one shared transformer trunk with actor, critic, and twin-critic heads"
            if shared
            else (
                "independent transformer and head for actor, critic, "
                "and twin critic"
            )
        ),
        "gamma": 0.0,
        "n_step": 1,
        "twin_q": True,
        "tau": 0.005,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "shared_encoder_learning_rate": (
            SHARED_ENCODER_LEARNING_RATE if shared else None
        ),
        "alpha_learning_rate": ALPHA_LEARNING_RATE,
        "target_entropy_fraction_of_categorical_maximum": (
            TARGET_ENTROPY_FRACTION
        ),
        "target_entropy": TARGET_ENTROPY,
        "train_batch_size_per_learner": (
            SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
        ),
        "learner_minibatch_count": LEARNER_MINIBATCH_COUNT,
        "learner_minibatch_size": (
            SMOKE_BATCH_SIZE
            if context.smoke
            else LEARNER_MINIBATCH_SIZE
        ),
        "learner_num_epochs": 1,
        "training_intensity": TRAINING_INTENSITY,
        "learning_starts": (
            SMOKE_LEARNING_STARTS
            if context.smoke
            else LEARNING_STARTS
        ),
        "replay_capacity": (
            SMOKE_REPLAY_CAPACITY
            if context.smoke
            else REPLAY_CAPACITY
        ),
        "model": MODEL_CONFIG,
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "checkpoint_schedule": {
            "initial_env_steps": 0,
            "log_spaced": (
                "power-of-two training iterations through 30,000,000 "
                "environment steps"
            ),
            "phase_boundary_env_steps": LOG_CHECKPOINT_END_ENV_STEPS,
            "fixed_interval_env_steps": CHECKPOINT_INTERVAL_ENV_STEPS,
            "fixed_targets_env_steps": _fixed_checkpoint_targets(),
            "final_env_steps": TOTAL_ENV_STEPS,
        },
        "torch_compile_learner": False,
        "intended_hardware": (
            "CPU smoke; one NVIDIA H100-class GPU for a full run"
        ),
    }


def run_condition(
    context: RunContext,
    *,
    architecture: str,
) -> dict[str, object]:
    if context.seed is None:
        raise ValueError("Pusher discrete SAC requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for these new recipes")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        resolved_recipe(context, architecture=architecture),
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    result_grid = run_tune(
        build_config(context, architecture=architecture),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": target_steps,
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(
            f"{architecture} expected one Tune trial, found {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{architecture} SAC training failed") from result.error
    write_training_curves(context)
    metrics = result.metrics
    summary = {
        "study": "pusher_b10_discrete_SAC_2026_09",
        "condition": architecture,
        "preset": "b10",
        "parameters": PRESETS["b10"],
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "completed_env_steps": _metric(
            metrics,
            "env_runners/num_env_steps_sampled_lifetime",
        ),
        "episode_return_mean": _metric(
            metrics,
            "env_runners/episode_return_mean",
        ),
        "episode_length_mean": _metric(
            metrics,
            "env_runners/episode_len_mean",
        ),
    }
    outputs.write_json("summary.json", summary)
    return summary
