from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from experiments.pusher_b_reward_state_action_symmetry_cycle_1.shared import (
    MIN_EPISODES_PER_TRAIN_BATCH,
    SMOKE_ENV_STEPS,
    build_config as build_source_config,
    resolved_recipe as source_resolved_recipe,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, available_cpus
from harness.runners import run_tune


BENCHMARK_ENV_STEPS = 1_048_576
PRESET = "b10"
VARIANT = 2
REWARD_STATE = "B"


def sampling_layout(
    context: RunContext,
    *,
    max_env_runners: int,
) -> tuple[int, int]:
    if max_env_runners <= 0:
        raise ValueError("max_env_runners must be positive")
    if context.smoke:
        return 0, 1
    num_env_runners = min(
        max_env_runners,
        max(1, int(available_cpus()) - 1),
    )
    return (
        num_env_runners,
        math.ceil(MIN_EPISODES_PER_TRAIN_BATCH / num_env_runners),
    )


def build_config(
    context: RunContext,
    *,
    max_env_runners: int,
) -> PPOConfig:
    config = build_source_config(
        context,
        preset=PRESET,
        variant=VARIANT,
        reward_state=REWARD_STATE,
    )
    num_env_runners, num_envs_per_env_runner = sampling_layout(
        context,
        max_env_runners=max_env_runners,
    )
    return config.env_runners(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        num_gpus_per_env_runner=0,
    )


def resolved_recipe(
    context: RunContext,
    *,
    label: str,
    max_env_runners: int,
) -> dict[str, object]:
    recipe = source_resolved_recipe(
        context,
        preset=PRESET,
        variant=VARIANT,
        reward_state=REWARD_STATE,
    )
    num_env_runners, num_envs_per_env_runner = sampling_layout(
        context,
        max_env_runners=max_env_runners,
    )
    recipe.update(
        {
            "study": (
                "pusher_b_reward_state_action_symmetry_cycle_1_throughput"
            ),
            "condition": label,
            "source_recipe": (
                "experiments.pusher_b_reward_state_action_symmetry_cycle_1."
                "b10_reward_b.variant_2.experiment"
            ),
            "throughput_hypothesis": (
                "distributing the fixed complete-episode batch over more "
                "independent CPU EnvRunners reduces the sampling bottleneck "
                "without materially changing initial learning"
            ),
            "primary_comparison": (
                "same-seed b10 reward-B variant-2 PPO with at most 16, 32, "
                "or 52 CPU EnvRunners on one matched GPU host"
            ),
            "total_env_steps": (
                SMOKE_ENV_STEPS if context.smoke else BENCHMARK_ENV_STEPS
            ),
            "sampling_layout": {
                "max_env_runners": max_env_runners,
                "num_env_runners": num_env_runners,
                "num_envs_per_env_runner": num_envs_per_env_runner,
                "episodes_per_sampling_round": (
                    num_env_runners * num_envs_per_env_runner
                ),
                "batch_mode": "complete_episodes",
                "env_runner": (
                    "harness.env_runners:"
                    "FreshEpisodeSingleAgentEnvRunner"
                ),
                "env_runner_device": "cpu",
            },
            "intended_hardware": (
                "one RTX 4090-equivalent learner GPU and at least 53 CPUs"
            ),
        }
    )
    return recipe


def _metric(metrics: Mapping[str, object], path: str) -> object | None:
    value: object = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def run_condition(
    context: RunContext,
    *,
    label: str,
    max_env_runners: int,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("Pusher-B throughput runs require a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    profile = context.hardware or PROFILES["cpu"]
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        resolved_recipe(
            context,
            label=label,
            max_env_runners=max_env_runners,
        ),
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else BENCHMARK_ENV_STEPS
    result_grid = run_tune(
        build_config(
            context,
            max_env_runners=max_env_runners,
        ),
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
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError(f"{label} PPO training failed")
    write_training_curves(context)
    metrics = results[0].metrics
    num_env_runners, num_envs_per_env_runner = sampling_layout(
        context,
        max_env_runners=max_env_runners,
    )
    summary = {
        "study": "pusher_b_reward_state_action_symmetry_cycle_1_throughput",
        "condition": label,
        "source_condition": "b10_reward_b_variant_2",
        "seed": context.seed,
        "smoke": context.smoke,
        "hardware_profile": profile.name,
        "target_env_steps": target_steps,
        "num_env_runners": num_env_runners,
        "num_envs_per_env_runner": num_envs_per_env_runner,
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
        "sampling_time_s": _metric(
            metrics,
            "timers/env_runner_sampling_timer",
        ),
        "learner_update_time_s": _metric(
            metrics,
            "timers/learner_update_timer",
        ),
        "training_iteration_time_s": _metric(
            metrics,
            "timers/training_iteration",
        ),
        "sampling_throughput_env_steps_s": _metric(
            metrics,
            "env_runners/num_env_steps_sampled_lifetime_throughput/"
            "throughput_since_last_reduce",
        ),
        "learner_throughput_env_steps_s": _metric(
            metrics,
            "learners/__all_modules__/"
            "num_env_steps_trained_lifetime_throughput/"
            "throughput_since_last_reduce",
        ),
        "cpu_util_percent": _metric(metrics, "perf/cpu_util_percent"),
        "ram_util_percent": _metric(metrics, "perf/ram_util_percent"),
    }
    outputs.write_json("summary.json", summary)
    return summary
