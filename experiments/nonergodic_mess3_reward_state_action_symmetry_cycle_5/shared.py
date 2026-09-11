from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
import math
from pathlib import Path
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
    checkpoint_records,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.process import (
    COMPONENT_PARAMETERS,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    environment_config,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.task import (
    DIRECTIONS,
    EFFECT_SIZE,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
from harness.hardware import HardwareProfile, PROFILES, resolve_env_runners
from harness.runners import run_tune


ARTICLE_URL = "https://simplex.pub/nonergodic-geometry/"
TOTAL_ENV_STEPS = 700_000
SMOKE_ENV_STEPS = 1_024
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 512
MINIBATCH_SIZE = 2_048
SMOKE_MINIBATCH_SIZE = 128
LEARNING_RATE = 4.2e-4
SMOKE_LEARNING_RATE = 3e-4
DEFAULT_ENTROPY_COEFF = 0.003
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


def build_config(context: RunContext, variant: int) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
    )
    return (
        PPOConfig()
        .environment(
            HMMEnv,
            env_config=environment_config(variant),
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
            entropy_coeff=DEFAULT_ENTROPY_COEFF,
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
    variant: int,
) -> dict[str, object]:
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
    )
    return {
        "study": "nonergodic_mess3_reward_state_action_symmetry_cycle_5",
        "condition": f"variant_{variant}",
        "seed": context.seed,
        "smoke": context.smoke,
        "source": ARTICLE_URL,
        "hypothesis": (
            "The action-symmetry ladder changes which coordinates of the "
            "six-state non-ergodic belief are useful for reward control."
        ),
        "components": [dict(parameters) for parameters in COMPONENT_PARAMETERS],
        "component_prior": [0.5, 0.5],
        "component_sampling": (
            "one component is selected at reset and remains fixed for the "
            "complete 127-decision episode"
        ),
        "variant_directions": DIRECTIONS[variant].tolist(),
        "effect_size": EFFECT_SIZE,
        "action_control": (
            "exponential tilt of each component's local reward-state "
            "destination probability with the original transition support "
            "preserved; edge kernels retain P(token | source, destination)"
        ),
        "support_note": (
            "mess3_b has zero self-transition probability, so finite "
            "exponential tilts preserve that zero exactly"
        ),
        "reward": (
            "one when the pre-transition local state is state 2 in either "
            "component, else zero"
        ),
        "environment": environment_config(variant),
        "previous_action_in_observation": True,
        "previous_reward_in_observation": False,
        "environment_seed_semantics": (
            "the fixed worker seed initializes each vector environment once; "
            "later complete-episode resets advance the same RNG stream"
        ),
        "env_runner": (
            "harness.env_runners:FreshEpisodeSingleAgentEnvRunner"
        ),
        "sampling_layout": {
            "num_env_runners": num_env_runners,
            "num_envs_per_env_runner": num_envs_per_env_runner,
            "episodes_per_sampling_round": (
                num_env_runners * num_envs_per_env_runner
            ),
        },
        "algorithm": "clipped PPO",
        "learning_rate": (
            SMOKE_LEARNING_RATE if context.smoke else LEARNING_RATE
        ),
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "entropy_coeff": DEFAULT_ENTROPY_COEFF,
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
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "analysis": [
            "action-conditioned edge-transducer reconstruction of the exact "
            "six-state decision-time belief",
            "held-out layerwise affine probes of weighted belief, component "
            "posterior, reward-state belief, and antisymmetric belief",
            "pending-token-distribution and observable token/action controls",
        ],
        "intended_hardware": (
            "CPU smoke; hardware-profile learner device for full training"
        ),
    }


def run_condition(context: RunContext, variant: int) -> dict[str, Any]:
    from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.analysis import (
        analyze_checkpoint,
    )

    if context.seed is None:
        raise ValueError("non-ergodic action symmetry requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    condition = f"variant_{variant}"
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        resolved_recipe(context, variant),
    )
    result_grid = run_tune(
        build_config(context, variant),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            )
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
    records: list[Mapping[str, Any]] = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *checkpoint_records(
            results[0],
            checkpoint_root=(
                context.artifacts_dir / "log_spaced_checkpoints"
            ),
        ),
    ]
    reports = []
    for record in records:
        reports.append(
            analyze_checkpoint(
                replace(
                    context,
                    results_dir=(
                        context.results_dir
                        / "checkpoint_probes"
                        / f"steps_{record['agent_steps']:09d}"
                    ),
                    resume_from=Path(record["checkpoint_path"]),
                ),
                variant=variant,
                checkpoint=Path(record["checkpoint_path"]),
                checkpoint_label=str(record["checkpoint_name"]),
                agent_steps=int(record["agent_steps"]),
                training_iteration=int(record["training_iteration"]),
            )
        )
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "checkpoint_probes": [
            {
                "checkpoint": report["checkpoint"],
                "agent_steps": report["agent_steps"],
                "training_iteration": report["training_iteration"],
                "probe_fits": report["probe_fits"],
                "controls": report["controls"],
                "policy": report["policy"],
            }
            for report in reports
        ],
        "initial_probe": reports[0],
        "final_probe": reports[-1],
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
