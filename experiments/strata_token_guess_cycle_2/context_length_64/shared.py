from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.storage.training_curves import write_training_curves
from experiments.strata_token_guess_cycle_2.context_length_64.process import (
    CONTEXT_LENGTH,
    environment_config,
)
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 8_192
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 1_024
SMOKE_MINIBATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 6
MODEL_CONFIG = {
    **FactoredReproductionModelConfig(
        d_model=64,
        n_layers=3,
        n_heads=1,
        d_mlp=256,
        context_length=CONTEXT_LENGTH,
        max_seq_len=64,
        positional_embedding="rope",
    ).to_dict(),
    "sampling_temperature": 1.5,
}


def build_config(context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config())
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            lr=LEARNING_RATE,
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
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
                module_class=WingActorCritic,
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
            num_env_runners=(
                0 if context.smoke else resolve_env_runners(profile, default=16)
            ),
            num_envs_per_env_runner=(
                1 if context.smoke else profile.num_envs_per_env_runner
            ),
            num_gpus_per_env_runner=0,
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            )
        )
    )


def resolved_recipe(context: RunContext) -> dict[str, Any]:
    return {
        "study": "strata_token_guess_cycle_2",
        "condition": "ppo",
        "seed": context.seed,
        "smoke": context.smoke,
        "hypothesis": (
            "PPO can learn the finite-context predictor for the pending "
            "single-HMM Strata token from delayed token history."
        ),
        "primary_comparison": (
            "initial versus trained token accuracy and linear belief decodability"
        ),
        "controls": [
            "train-mean belief predictor",
            "current-visible-token belief predictor",
            "shuffled-label affine probe",
        ],
        "environment": environment_config(),
        "action_semantics": (
            "two categorical action logits, one per Strata token"
        ),
        "reward": (
            "1 when the sampled action equals event.raw_token_before, else 0"
        ),
        "previous_reward_in_observation": False,
        "previous_action_in_observation": False,
        "algorithm": "clipped PPO",
        "optimizer": "Adam (RLlib default)",
        "learning_rate": LEARNING_RATE,
        "gamma": 0.0,
        "lambda": 0.0,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "entropy_coeff": 0.0,
        "train_batch_size_per_learner": (
            SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
        ),
        "num_epochs": NUM_EPOCHS,
        "model": dict(MODEL_CONFIG),
        "context_semantics": (
            "strict last 64 delayed token frames including current"
        ),
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "stopping_metric": (
            "env_runners/num_env_steps_sampled_lifetime"
        ),
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "analysis": (
            "held-out layerwise affine probes of the exact delayed Strata belief"
        ),
        "reference_experiment": "wing_token_guess_cycle_2",
        "process_change": {
            "before": "Strata(alpha=0.97, t0=0.38, t1=0.54)",
            "after": "Strata(alpha=0.98, t0=0.30, t1=0.80)",
        },
        "intended_hardware": (
            "CPU smoke; hardware-profile learner device for full training"
        ),
    }


def run_condition(context: RunContext) -> dict[str, Any]:
    from experiments.strata_token_guess_cycle_2.analysis import analyze_checkpoint

    if context.seed is None:
        raise ValueError("Strata token-guess PPO requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", resolved_recipe(context))
    result_grid = run_tune(
        build_config(context),
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
        raise RuntimeError("Strata token-guess PPO training failed")
    write_training_curves(context)
    records = [
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
                checkpoint=Path(record["checkpoint_path"]),
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    summary = {
        "condition": "ppo",
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
