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
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
from experiments.wing_two_factor_explore_cycle_1.process import (
    ACTION_PAIRS,
    CONTEXT_LENGTH,
    environment_config,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


TOTAL_ENV_STEPS = {"reward_factor_1": 30_000_000, "reward_both": 50_000_000}
SMOKE_ENV_STEPS = 2_048
TRAIN_BATCH_SIZE = 8_192
SMOKE_BATCH_SIZE = 1_024
MINIBATCH_SIZE = 512
SMOKE_MINIBATCH_SIZE = 128
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_EPOCHS = 6
ENTROPY_COEFF = 0.003
SAMPLING_TEMPERATURE = 1.5
MODEL_CONFIG = {
    **FactoredReproductionModelConfig(
        d_model=64,
        n_layers=3,
        n_heads=1,
        d_mlp=256,
        context_length=CONTEXT_LENGTH,
        max_seq_len=32,
        positional_embedding="rope",
    ).to_dict(),
    "sampling_temperature": SAMPLING_TEMPERATURE,
}


def build_config(context: RunContext, condition: str, reward_state: int) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(condition, reward_state))
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
        .training(
            lr=LEARNING_RATE,
            gamma=GAMMA,
            lambda_=GAE_LAMBDA,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            vf_clip_param=10.0,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            entropy_coeff=ENTROPY_COEFF,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE,
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
                checkpoint_path=str(context.artifacts_dir / "initial_checkpoint"),
            ),
            on_train_result=partial(
                _save_log_spaced_checkpoint,
                checkpoint_root=str(context.artifacts_dir / "log_spaced_checkpoints"),
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
        .learners(num_gpus_per_learner=1 if profile.learner_device == "cuda" else 0)
    )


def resolved_recipe(context: RunContext, condition: str, reward_state: int) -> dict[str, Any]:
    from experiments.wing_two_factor_explore_cycle_1.design import design_summary

    return {
        "study": "wing_two_factor_explore_cycle_1",
        "condition": condition,
        "reward_state": reward_state,
        "seed": context.seed,
        "smoke": context.smoke,
        "hypothesis": "selective reward increases rewarded-factor belief decodability",
        "interpretation": (
            "Belief MSE alone is not evidence of causal use. Compare normalized MSE, "
            "NTP/log-NTP and shuffled controls, especially the NTP-invisible p0-p2 direction."
        ),
        "environment": environment_config(condition, reward_state),
        "action_pairs_by_flat_index": [list(pair) for pair in ACTION_PAIRS],
        "factor_action_labels": ["hold", "rotate_plus", "rotate_minus"],
        "reward": "mean arrival-state indicator over rewarded factors; range [0,1]",
        "previous_reward_in_observation": False,
        "analytic_design": design_summary(),
        "algorithm": "clipped PPO with shared actor-critic encoder",
        "optimizer": "Adam (RLlib default)",
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "lambda": GAE_LAMBDA,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "value_clip_param": 10.0,
        "grad_clip_global_norm": 0.5,
        "entropy_coeff": ENTROPY_COEFF,
        "train_batch_size_per_learner": SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE,
        "minibatch_size": SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "model": dict(MODEL_CONFIG),
        "context_semantics": "strict last 32 joint-token/previous-action frames including current",
        "temperature_semantics": "logits divided by 1.5 in rollout, PPO likelihoods, and stochastic probes",
        "total_env_steps": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS[condition],
        "budget_semantics": "stop after crossing threshold at a complete training iteration",
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "probe_schedule": "all saved checkpoints after training, independent train/test streams",
    }


def run_condition(context: RunContext, condition: str, reward_state: int) -> dict[str, Any]:
    from experiments.wing_two_factor_explore_cycle_1.analysis import analyze_checkpoint

    if context.seed is None:
        raise ValueError("Wing PPO requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", resolved_recipe(context, condition, reward_state))
    result_grid = run_tune(
        build_config(context, condition, reward_state),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS[condition]
            )
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(num_to_keep=1, checkpoint_at_end=True)
        },
    )
    results = list(result_grid)
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError(f"{condition}, state {reward_state}: PPO training failed")
    write_training_curves(context)
    records = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *checkpoint_records(
            results[0], checkpoint_root=context.artifacts_dir / "log_spaced_checkpoints"
        ),
    ]
    reports = []
    for record in records:
        reports.append(
            analyze_checkpoint(
                replace(
                    context,
                    results_dir=context.results_dir / "checkpoint_probes" / f"steps_{record['agent_steps']:09d}",
                    resume_from=Path(record["checkpoint_path"]),
                ),
                checkpoint=Path(record["checkpoint_path"]),
                condition=condition,
                reward_state=reward_state,
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    summary = {
        "condition": condition,
        "reward_state": reward_state,
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
