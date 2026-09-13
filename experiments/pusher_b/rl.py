from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Mapping

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


def _init_algorithm(
    *,
    algorithm,
    checkpoint_path: str,
    warm_start_path: str | None = None,
    **kwargs,
) -> None:
    _save_initial_checkpoint(
        algorithm=algorithm,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )
    if warm_start_path is not None:
        algorithm.restore_from_path(warm_start_path)


def _sampling_layout(
    context: RunContext,
    profile: HardwareProfile,
) -> tuple[int, int]:
    if context.smoke:
        return 0, 1
    return resolve_env_runners(profile, default=16), NUM_ENVS_PER_ENV_RUNNER


def build_config(context: RunContext, *, preset: str) -> PPOConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
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
            lr=LEARNING_RATE_SCHEDULE,
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.25,
            entropy_coeff=ENTROPY_COEFF_SCHEDULE,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
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
            on_train_result=partial(
                _save_log_spaced_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "log_spaced_checkpoints"
                ),
            ),
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
) -> dict[str, object]:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
    )
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
        "learning_rate": LEARNING_RATE_SCHEDULE,
        "gamma": 0.0,
        "lambda": 0.0,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.25,
        "entropy_coeff": ENTROPY_COEFF_SCHEDULE,
        "train_batch_size_per_learner": (
            SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
        ),
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
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "stopping_metric": "env_runners/num_env_steps_sampled_lifetime",
        "checkpoint_schedule": "initial, powers of two iterations, final",
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


def run_ppo(context: RunContext, *, preset: str) -> dict[str, object]:
    if context.seed is None:
        raise ValueError("Pusher-B PPO requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        resolved_recipe(context, preset=preset),
    )
    tune_context = (
        replace(context, resume_from=None)
        if context.resume_from is not None
        else context
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    result_grid = run_tune(
        build_config(context, preset=preset),
        tune_context,
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
