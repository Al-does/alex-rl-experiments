"""Shared controlled recipe for four Kelly-sizing conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess_3_kelly_cycle_1.analysis import probe_checkpoint
from experiments.mess_3_kelly_cycle_1.kelly import (
    COLLAPSE_THRESHOLD,
    MAX_WAGER,
)
from experiments.mess_3_kelly_cycle_1.learning import (
    DIRECT_LOSS_WEIGHT_KEY,
    FIXED_MODE,
    LEARNED_MODE,
    MODE_KEY,
    POLICY_MODE,
    KellyRewardPPOTorchLearner,
    WagerTransformerModel,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.transformer import (
    TransformerModel,
    TransformerModelConfig,
)


CONDITIONS = (
    "fixed_full",
    "policy_implied_kelly",
    "learned_kelly",
    "bayes_oracle",
)
TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
DIRECT_LOSS_WEIGHT = 1.0
BASE_MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()


def environment_config(condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown Kelly condition {condition!r}")
    task_name = (
        "BayesOracleKellyTask"
        if condition == "bayes_oracle"
        else "RawNextTokenTask"
    )
    return {
        "model": {
            "factory": "envs.mess3.model:passive_model",
            "kwargs": {"alpha": 0.85},
        },
        "task": {
            "class": (
                "experiments.mess_3_kelly_cycle_1.task:"
                f"{task_name}"
            ),
        },
        "observation": {"action": None},
        "delay": 1,
        "episode_length": 512,
        "randomize_first_episode_length": True,
    }


def _apply_runtime_resources(config: PPOConfig, context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return config.env_runners(
        num_env_runners=(
            0
            if context.smoke
            else resolve_env_runners(profile, default=16)
        ),
        num_envs_per_env_runner=(
            1 if context.smoke else profile.num_envs_per_env_runner
        ),
        num_gpus_per_env_runner=(
            0 if context.smoke else profile.num_gpus_per_env_runner
        ),
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=(
            1 if profile.learner_device == "cuda" else 0
        )
    )


def build_config(context: RunContext, condition: str) -> PPOConfig:
    """Build a fresh PPO config with no warm start or predictive auxiliary loss."""

    if context.seed is None:
        raise ValueError("Kelly experiments require a resolved seed")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown Kelly condition {condition!r}")
    profile = context.hardware or PROFILES["cpu"]
    module_class = (
        WagerTransformerModel
        if condition == LEARNED_MODE
        else TransformerModel
    )
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(condition))
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
            lr=3e-4,
            gamma=1.0,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=(
                2_048 if context.smoke else 32_768
            ),
            minibatch_size=256 if context.smoke else 4_096,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=module_class,
                model_config=dict(BASE_MODEL_CONFIG),
            )
        )
        .debugging(seed=context.seed)
    )
    if condition != "bayes_oracle":
        config = config.learners(
            learner_class=KellyRewardPPOTorchLearner,
            learner_config_dict={
                MODE_KEY: condition,
                DIRECT_LOSS_WEIGHT_KEY: (
                    DIRECT_LOSS_WEIGHT
                    if condition == LEARNED_MODE
                    else 0.0
                ),
            },
        )
    return _apply_runtime_resources(config, context)


def _findings(
    *,
    condition: str,
    metrics: Mapping[str, Any],
    collapsed: bool,
) -> str:
    return "\n".join(
        [
            f"# {condition.replace('_', ' ').title()}",
            "",
            f"- Held-out rank-2 belief R²: {metrics['r_squared']:.4f}",
            f"- Greedy token accuracy: {metrics['token_accuracy_greedy']:.4f}",
            f"- Mean wager: {metrics['wager_mean']:.4f}",
            (
                "- Fraction of wagers below "
                f"{COLLAPSE_THRESHOLD:g}: "
                f"{metrics['wager_collapse_fraction']:.4f}"
            ),
            (
                "- Mean expected log growth: "
                f"{metrics['expected_log_growth_mean']:.6f}"
            ),
            f"- Operational wager collapse detected: {str(collapsed).lower()}",
            "",
        ]
    )


def run_condition(context: RunContext, condition: str):
    """Train one condition, then write compact probe and learning diagnostics."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown Kelly condition {condition!r}")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": condition,
            "environment": environment_config(condition),
            "model": BASE_MODEL_CONFIG,
            "algorithm": "PPO",
            "total_env_steps": target_steps,
            "gamma": 1.0,
            "lambda": 0.95,
            "net_win_odds": 2.0,
            "max_wager": MAX_WAGER,
            "direct_kelly_loss_weight": (
                DIRECT_LOSS_WEIGHT if condition == LEARNED_MODE else 0.0
            ),
            "predictive_auxiliary_loss_weight": 0.0,
            "warm_start": False,
            "environment_reward_note": (
                "The three learner-sized arms expose correctness in the "
                "environment and replace it with Kelly log growth on-device "
                "before GAE. The Bayes-oracle task emits log growth directly."
            ),
        },
    )
    result_grid = run_tune(
        build_config(context, condition),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"{condition} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{condition} produced no final checkpoint")

    probed = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition=condition,
    )
    collapsed = bool(
        condition in {POLICY_MODE, LEARNED_MODE}
        and probed.metrics["wager_mean"] < COLLAPSE_THRESHOLD
        and probed.metrics["wager_collapse_fraction"] > 0.95
    )
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_agent_steps": target_steps,
        "warm_start": False,
        "predictive_auxiliary_loss": False,
        "wager_collapse_detected": collapsed,
        "probe": probed.metrics,
    }
    outputs.write_json("condition_summary.json", summary)
    (context.results_dir / "findings.md").write_text(
        _findings(
            condition=condition,
            metrics=probed.metrics,
            collapsed=collapsed,
        )
    )
    return summary
