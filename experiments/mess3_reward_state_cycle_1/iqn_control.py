"""Shared IQN recipe mechanics for reward-state control conditions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.checkpoint_probe import (
    experiment as checkpoint_probe,
)
from experiments.mess3_belief_geometry_2026_07.control_costs import (
    LEARNING_RATE,
    MODEL_CONFIG as BASE_MODEL_CONFIG,
    TOTAL_ENV_STEPS,
    TRAIN_BATCH_SIZE,
    environment_config,
)
from experiments.mess3_belief_geometry_2026_07.shared import (
    SMOKE_ENV_STEPS,
    apply_runtime_resources,
)
from experiments.mess3_reward_state_cycle_1.iqn import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    NAMESPACE,
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune


IQN_CONFIG = {
    "train_quantiles": 32,
    "value_quantiles": 64,
    "n_cosines": 64,
}
IQN_LOSS_COEFFICIENT = 0.5
IQN_HUBER_KAPPA = 1.0
IQN_MINIBATCH_SIZE = 4_096
MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    NAMESPACE: IQN_CONFIG,
}
LEARNER_CONFIG = {
    LOSS_COEFFICIENT_KEY: IQN_LOSS_COEFFICIENT,
    HUBER_KAPPA_KEY: IQN_HUBER_KAPPA,
}


def build_config(
    context: RunContext,
    *,
    task_kwargs: dict[str, Any],
) -> PPOConfig:
    """Build continuous-control PPO with only the value critic changed."""

    profile = context.hardware or PROFILES["cpu"]
    compile_learner = (
        not context.smoke and profile.learner_device == "cuda"
    )
    config = (
        PPOConfig()
        .environment(
            HMMEnv,
            env_config=environment_config(task_kwargs),
        )
        .framework(
            "torch",
            torch_compile_learner=compile_learner,
            torch_compile_learner_what_to_compile="forward_train",
            torch_compile_learner_dynamo_backend="inductor",
            torch_compile_learner_dynamo_mode="reduce-overhead",
            torch_compile_worker=False,
        )
        .learners(
            learner_class=IQNPPOTorchLearner,
            learner_config_dict=LEARNER_CONFIG,
        )
        .training(
            lr=3e-4 if context.smoke else LEARNING_RATE,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.0,
            entropy_coeff=0.003,
            train_batch_size_per_learner=(
                2_048 if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                256 if context.smoke else IQN_MINIBATCH_SIZE
            ),
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=IQNTransformerModel,
                model_config=MODEL_CONFIG,
            )
        )
        .debugging(seed=context.seed)
    )
    return apply_runtime_resources(
        config,
        context,
        default_env_runners=16,
    )


def run_condition(
    context: RunContext,
    *,
    condition: str,
    task_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Train and probe one IQN reward-state condition."""

    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    condition_spec = {
        "condition": condition,
        "task_kwargs": dict(environment_config(task_kwargs)["task"]["kwargs"]),
        "critic": "implicit_quantile_network",
        "iqn_config": IQN_CONFIG,
        "iqn_loss_coefficient": IQN_LOSS_COEFFICIENT,
        "iqn_huber_kappa": IQN_HUBER_KAPPA,
        "total_env_steps": target_steps,
        "train_batch_size": (
            2_048 if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            256 if context.smoke else IQN_MINIBATCH_SIZE
        ),
        "learning_rate": 3e-4 if context.smoke else LEARNING_RATE,
        "model_config": MODEL_CONFIG,
    }
    (context.results_dir / "condition_spec.json").write_text(
        json.dumps(condition_spec, indent=2) + "\n"
    )

    result_grid = run_tune(
        build_config(context, task_kwargs=task_kwargs),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=2,
                checkpoint_frequency=1 if context.smoke else 10,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(
            f"{condition} expected one Tune trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{condition} produced no final checkpoint")

    probe_metrics = checkpoint_probe.run(
        replace(
            context,
            resume_from=Path(result.checkpoint.path),
        )
    )
    summary = {
        "condition": condition,
        "checkpoint": str(result.checkpoint.path),
        "probe": probe_metrics,
    }
    (context.results_dir / "condition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
