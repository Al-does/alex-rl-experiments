from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import math
from typing import Any

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
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.design import (
    condition_design_summary,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.process import (
    CONTEXT_LENGTH,
    EFFECT_SIZE,
    EPISODE_LENGTH,
    PRESETS,
    REWARD_STATES,
    environment_config,
    pusher_b_model,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.task import (
    N_ACTIONS,
    direction_matrix,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
from harness.hardware import HardwareProfile, PROFILES, resolve_env_runners
from harness.runners import run_tune


TOTAL_ENV_STEPS = 50_000_000
SMOKE_ENV_STEPS = 1_024
TRAIN_BATCH_SIZE = 262_144
SMOKE_BATCH_SIZE = 512
MINIBATCH_SIZE = 8_192
SMOKE_MINIBATCH_SIZE = 128
LEARNING_RATE = 4.2e-4
SMOKE_LEARNING_RATE = 3e-4
ENTROPY_COEFF = 0.01
NUM_EPOCHS = 6
MIN_EPISODES_PER_TRAIN_BATCH = math.ceil(
    TRAIN_BATCH_SIZE / EPISODE_LENGTH
)
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


def _validate_condition(
    preset: str,
    variant: int,
    reward_state: str,
) -> None:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    if reward_state not in REWARD_STATES:
        raise ValueError("reward_state must be 'A' or 'B'")


def _sampling_layout(
    context: RunContext,
    profile: HardwareProfile,
) -> tuple[int, int]:
    if context.smoke:
        return 0, 1
    num_env_runners = resolve_env_runners(profile, default=16)
    return (
        num_env_runners,
        math.ceil(MIN_EPISODES_PER_TRAIN_BATCH / num_env_runners),
    )


def build_config(
    context: RunContext,
    *,
    preset: str,
    variant: int,
    reward_state: str,
) -> PPOConfig:
    _validate_condition(preset, variant, reward_state)
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
    )
    return (
        PPOConfig()
        .environment(
            HMMEnv,
            env_config=environment_config(
                preset,
                variant,
                reward_state,
            ),
        )
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
            lr=SMOKE_LEARNING_RATE if context.smoke else LEARNING_RATE,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=ENTROPY_COEFF,
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
                _save_initial_checkpoint,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
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
    variant: int,
    reward_state: str,
) -> dict[str, object]:
    _validate_condition(preset, variant, reward_state)
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
    )
    model = pusher_b_model(preset)
    reward_state_index = model.state_labels.index(reward_state)
    return {
        "study": "pusher_b_reward_state_action_symmetry_cycle_1",
        "condition": (
            f"{preset}_reward_{reward_state.lower()}_variant_{variant}"
        ),
        "hypothesis": (
            "Delayed Pusher-B token and action history can support policies "
            "whose required hidden-state action symmetry ranges from one "
            "constant action to three distinct actions."
        ),
        "primary_comparison": (
            "within each Pusher-B preset and reward-state choice, compare "
            "variants requiring one, two, or three state-conditioned optimal "
            "actions"
        ),
        "preset": preset,
        "parameters": PRESETS[preset],
        "reward_state": reward_state,
        "reward_state_index": reward_state_index,
        "variant": variant,
        "variant_directions": direction_matrix(
            variant,
            reward_state_index,
        ).tolist(),
        "effect_size": EFFECT_SIZE,
        "effect_size_selection": (
            "odds tilt 1.5 matches the established cycle-5 symmetry recipe; "
            "all 12 full-state oracle policies have the requested action "
            "structure while every controlled destination probability stays "
            "below 0.96"
        ),
        "action_control": (
            "three actions exponentially tilt the odds of the designated "
            "reward-state destination while preserving transition support, "
            "the ratio between non-reward destinations, and "
            "P(token | source, destination)"
        ),
        "action_semantics": {
            "0": "noop baseline transition",
            "1": "positive cyclic-control action",
            "2": "negative cyclic-control action",
        },
        "reward": (
            f"one when the pre-transition hidden state is {reward_state}, "
            "else zero"
        ),
        "observation": (
            "one-step-delayed token one-hot plus executed-action one-hot; no "
            "reward, belief, or hidden state"
        ),
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
        "environment": environment_config(
            preset,
            variant,
            reward_state,
        ),
        "design_diagnostics": condition_design_summary(
            preset,
            variant,
            reward_state,
        ),
        "seed": context.seed,
        "smoke": context.smoke,
        "seed_policy": "one fixed runtime seed per experiment; default 42",
        "algorithm": "clipped PPO",
        "learning_rate": (
            SMOKE_LEARNING_RATE if context.smoke else LEARNING_RATE
        ),
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "entropy_coeff": ENTROPY_COEFF,
        "train_batch_size_per_learner": (
            SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
        ),
        "num_epochs": NUM_EPOCHS,
        "model": dict(MODEL_CONFIG),
        "context_semantics": (
            "one learned BOS position plus 127 delayed token/action decisions "
            "forms one 128-position complete learner sequence"
        ),
        "episode_length": EPISODE_LENGTH,
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "stopping_metric": "env_runners/num_env_steps_sampled_lifetime",
        "success_metrics": (
            "episode return and reward-state occupancy learning curves, with "
            "the exact full-state oracle occupancy recorded as a design "
            "diagnostic rather than a POMDP-optimality claim"
        ),
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "compact_outputs": [
            "resolved_recipe.json",
            "training_curves.jsonl",
            "summary.json",
            "run_manifest.json",
            "tune_summary.json",
        ],
        "sampling_layout": {
            "num_env_runners": num_env_runners,
            "num_envs_per_env_runner": num_envs_per_env_runner,
            "episodes_per_sampling_round": (
                num_env_runners * num_envs_per_env_runner
            ),
            "batch_mode": "complete_episodes",
            "env_runner": (
                "harness.env_runners:FreshEpisodeSingleAgentEnvRunner"
            ),
        },
        "previous_action_in_observation": True,
        "previous_reward_in_observation": False,
        "intended_hardware": (
            "CPU smoke; one NVIDIA RTX 4090-equivalent GPU per full run"
        ),
        "remote_artifact_policy": (
            "upload ignored checkpoints and Tune artifacts to B2 before "
            "self-destruct; push compact results to the launch branch"
        ),
    }


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
    preset: str,
    variant: int,
    reward_state: str,
) -> dict[str, Any]:
    _validate_condition(preset, variant, reward_state)
    if context.seed is None:
        raise ValueError("Pusher-B action symmetry requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    condition = f"{preset}_reward_{reward_state.lower()}_variant_{variant}"
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        resolved_recipe(
            context,
            preset=preset,
            variant=variant,
            reward_state=reward_state,
        ),
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    result_grid = run_tune(
        build_config(
            context,
            preset=preset,
            variant=variant,
            reward_state=reward_state,
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
        raise RuntimeError(f"{condition} PPO training failed")
    write_training_curves(context)
    metrics = results[0].metrics
    summary = {
        "study": "pusher_b_reward_state_action_symmetry_cycle_1",
        "condition": condition,
        "preset": preset,
        "parameters": PRESETS[preset],
        "variant": variant,
        "reward_state": reward_state,
        "effect_size": EFFECT_SIZE,
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
