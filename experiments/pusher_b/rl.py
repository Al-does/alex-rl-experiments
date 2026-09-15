from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
from harness.hardware import HardwareProfile, PROFILES, resolve_env_runners
from harness.runners import run_tune

from .process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    PRESETS,
    environment_config,
    pusher_b_model,
)


TOTAL_ENV_STEPS = 15_000_000
SMOKE_ENV_STEPS = 1_024
TRAIN_BATCH_SIZE = 262_144
SMOKE_BATCH_SIZE = 512
MINIBATCH_SIZE = 8_192
SMOKE_MINIBATCH_SIZE = 128
NUM_ENVS_PER_ENV_RUNNER = 19
LEARNING_RATE = 5e-5
NUM_EPOCHS = 6
ENTROPY_COEFF_SCHEDULE = [
    [0, 0.01],
    [6_000_000, 0.01],
    [8_000_000, 0.0],
]
LEARNING_RATE_SCHEDULE = [
    [0, LEARNING_RATE],
    [5_000_000, LEARNING_RATE],
    [10_000_000, 1e-5],
]
MODEL_CONFIG = FactoredReproductionModelConfig(
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
    training_sequence_mode="complete_episode",
).to_dict()


@dataclass(frozen=True)
class PPOSettings:
    """Budget and optimizer knobs for a full (non-smoke) Pusher-B PPO run."""

    total_env_steps: int = TOTAL_ENV_STEPS
    train_batch_size: int = TRAIN_BATCH_SIZE
    minibatch_size: int = MINIBATCH_SIZE
    lr: float | list[list[float]] = tuple(map(tuple, LEARNING_RATE_SCHEDULE))
    entropy_coeff: float | list[list[float]] = tuple(
        map(tuple, ENTROPY_COEFF_SCHEDULE)
    )
    checkpoint_every_env_steps: int | None = None
    checkpoint_origin_env_steps: int = 0
    num_env_runners: int | None = None
    max_train_time_s: float | None = None

    def _schedule(self, value):
        if isinstance(value, (int, float)):
            return value
        return [list(row) for row in value]

    def resolved(self, smoke: bool) -> dict[str, Any]:
        return {
            "total_env_steps": SMOKE_ENV_STEPS if smoke else self.total_env_steps,
            "train_batch_size": (
                SMOKE_BATCH_SIZE if smoke else self.train_batch_size
            ),
            "minibatch_size": SMOKE_MINIBATCH_SIZE if smoke else self.minibatch_size,
            "lr": self._schedule(self.lr),
            "entropy_coeff": self._schedule(self.entropy_coeff),
            "checkpoint_every_env_steps": self.checkpoint_every_env_steps,
            "checkpoint_origin_env_steps": self.checkpoint_origin_env_steps,
            "num_env_runners": self.num_env_runners,
            "max_train_time_s": self.max_train_time_s,
        }


DEFAULT_SETTINGS = PPOSettings()


def _init_algorithm(
    *,
    algorithm,
    checkpoint_path: str,
    warm_start_path: str | None = None,
    **kwargs,
) -> None:
    if warm_start_path is not None:
        algorithm.restore_from_path(warm_start_path)
    _save_initial_checkpoint(
        algorithm=algorithm,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )


def _save_interval_checkpoint(
    *,
    algorithm,
    result: Mapping[str, Any],
    checkpoint_root: str,
    every_env_steps: int,
    origin_env_steps: int,
    **_: Any,
) -> None:
    """Save a public Algorithm checkpoint each time a step boundary is crossed."""

    iteration_value = _metric(result, "training_iteration")
    steps_value = _metric(result, "env_runners/num_env_steps_sampled_lifetime")
    if iteration_value is None or steps_value is None:
        return
    steps = int(steps_value)
    bucket = (steps - origin_env_steps) // every_env_steps
    if bucket <= 0:
        return
    root = Path(checkpoint_root)
    index_path = root / "index.json"
    records = (
        json.loads(index_path.read_text()).get("checkpoints", [])
        if index_path.is_file()
        else []
    )
    if any(int(record["bucket"]) >= bucket for record in records):
        return
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (
        f"iteration_{int(iteration_value):06d}_steps_{steps:09d}"
    )
    saved = Path(algorithm.save_to_path(str(destination)))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": int(iteration_value),
            "agent_steps": steps,
            "bucket": bucket,
            "target_agent_steps": origin_env_steps + bucket * every_env_steps,
        }
    )
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"checkpoints": records}, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(index_path)


def _sampling_layout(
    context: RunContext,
    profile: HardwareProfile,
    settings: PPOSettings = DEFAULT_SETTINGS,
) -> tuple[int, int]:
    if context.smoke:
        return 0, 1
    requested = settings.num_env_runners
    if requested is not None:
        profile = replace(profile, num_env_runners=requested)
    return resolve_env_runners(profile, default=16), NUM_ENVS_PER_ENV_RUNNER


def build_config(
    context: RunContext,
    *,
    preset: str,
    settings: PPOSettings = DEFAULT_SETTINGS,
) -> PPOConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
        settings,
    )
    knobs = settings.resolved(context.smoke)
    on_train_result = [
        partial(
            _save_log_spaced_checkpoint,
            checkpoint_root=str(
                context.artifacts_dir / "log_spaced_checkpoints"
            ),
        )
    ]
    if knobs["checkpoint_every_env_steps"]:
        on_train_result.append(
            partial(
                _save_interval_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "interval_checkpoints"
                ),
                every_env_steps=int(knobs["checkpoint_every_env_steps"]),
                origin_env_steps=int(knobs["checkpoint_origin_env_steps"]),
            )
        )
    return (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(preset))
        .framework(
            "torch",
            torch_compile_learner=(
                not context.smoke and profile.learner_device == "cuda"
            ),
            torch_compile_learner_what_to_compile="forward_train",
            torch_compile_learner_dynamo_backend="inductor",
            torch_compile_learner_dynamo_mode="reduce-overhead",
            torch_compile_worker=False,
        )
        .training(
            lr=knobs["lr"],
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.25,
            entropy_coeff=knobs["entropy_coeff"],
            train_batch_size_per_learner=knobs["train_batch_size"],
            minibatch_size=knobs["minibatch_size"],
            num_epochs=NUM_EPOCHS,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=FactoredReproductionActorCritic,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _init_algorithm,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
                ),
                warm_start_path=(
                    str(context.resume_from)
                    if context.resume_from is not None
                    else None
                ),
            ),
            on_train_result=on_train_result,
        )
        .debugging(seed=context.seed)
        .env_runners(
            env_runner_cls=FreshEpisodeSingleAgentEnvRunner,
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_gpus_per_env_runner=0,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            )
        )
    )


def resolved_recipe(
    context: RunContext,
    *,
    preset: str,
    settings: PPOSettings = DEFAULT_SETTINGS,
) -> dict[str, object]:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
        settings,
    )
    knobs = settings.resolved(context.smoke)
    model = pusher_b_model(preset)
    return {
        "study": "pusher_b",
        "condition": "ppo_token_guess",
        "preset": preset,
        "parameters": PRESETS[preset],
        "seed": context.seed,
        "smoke": context.smoke,
        "source_guide": "https://github.com/Al-does/alex-rl-experiments/pull/127",
        "hmm": {
            "states": list(model.state_labels),
            "tokens": list(model.token_labels),
            "initial_distribution": model.initial_distribution.tolist(),
            "transition_matrix": model.transition_matrix.tolist(),
            "emission_matrix": model.emission_matrix.tolist(),
            "edge_transition_matrices": (
                model.edge_transition_matrices.tolist()
            ),
        },
        "environment": environment_config(preset),
        "action_semantics": "two categorical logits, one per pending token",
        "reward": "1 when the sampled action equals the pending token, else 0",
        "previous_reward_in_observation": False,
        "previous_action_in_observation": False,
        "algorithm": "clipped PPO",
        "objective": "sampled next-token correctness only; no cross-entropy loss",
        "learning_rate": knobs["lr"],
        "gamma": 0.0,
        "lambda": 0.0,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.25,
        "entropy_coeff": knobs["entropy_coeff"],
        "train_batch_size_per_learner": knobs["train_batch_size"],
        "minibatch_size": knobs["minibatch_size"],
        "num_epochs": NUM_EPOCHS,
        "model": dict(MODEL_CONFIG),
        "sampling_layout": {
            "num_env_runners": num_env_runners,
            "num_envs_per_env_runner": num_envs_per_env_runner,
            "batch_mode": "complete_episodes",
            "env_runner": (
                "harness.env_runners:FreshEpisodeSingleAgentEnvRunner"
            ),
        },
        "episode_length": EPISODE_LENGTH,
        "total_env_steps": knobs["total_env_steps"],
        "stopping_metric": "env_runners/num_env_steps_sampled_lifetime",
        "max_train_time_s": knobs["max_train_time_s"],
        "checkpoint_schedule": (
            "initial, powers of two iterations, final"
            + (
                f", every {knobs['checkpoint_every_env_steps']} lifetime steps"
                f" from {knobs['checkpoint_origin_env_steps']}"
                if knobs["checkpoint_every_env_steps"]
                else ""
            )
        ),
        "warm_start_from": (
            str(context.resume_from) if context.resume_from is not None else None
        ),
        "intended_hardware": (
            "CPU smoke; full training sized for one NVIDIA H100-class GPU"
        ),
    }


def _metric(metrics: Mapping[str, object], path: str) -> object | None:
    value: object = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def run_ppo(
    context: RunContext,
    *,
    preset: str,
    settings: PPOSettings = DEFAULT_SETTINGS,
) -> dict[str, object]:
    if context.seed is None:
        raise ValueError("Pusher-B PPO requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        resolved_recipe(context, preset=preset, settings=settings),
    )
    tune_context = (
        replace(context, resume_from=None)
        if context.resume_from is not None
        else context
    )
    knobs = settings.resolved(context.smoke)
    target_steps = int(knobs["total_env_steps"])
    stop: dict[str, float] = {
        "env_runners/num_env_steps_sampled_lifetime": target_steps,
    }
    if knobs["max_train_time_s"] is not None:
        stop["time_total_s"] = float(knobs["max_train_time_s"])
    result_grid = run_tune(
        build_config(context, preset=preset, settings=settings),
        tune_context,
        stop=stop,
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError("Pusher-B token-guess PPO training failed")
    write_training_curves(context)
    metrics = results[0].metrics
    summary = {
        "study": "pusher_b",
        "condition": "ppo_token_guess",
        "preset": preset,
        "parameters": PRESETS[preset],
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "max_train_time_s": knobs["max_train_time_s"],
        "train_time_s": _metric(metrics, "time_total_s"),
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
