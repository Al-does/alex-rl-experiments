"""Controlled gamma-zero mean/IQN recipe for Kelly credit assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    NAMESPACE as IQN_NAMESPACE,
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from experiments.mess_3_kelly_cycle_1.kelly import COLLAPSE_THRESHOLD
from experiments.mess_3_kelly_cycle_2.analysis import probe_checkpoint
from experiments.mess_3_kelly_cycle_2.learning import (
    ACTOR_MODE_KEY,
    CONDITIONAL_LAYOUT,
    CONDITIONAL_MODE,
    COUPLED_MODE,
    CORRECTNESS_MODE,
    DECOUPLED_MODE,
    DIRECT_LOSS_WEIGHT_KEY,
    NO_WAGER_LAYOUT,
    SCALAR_LAYOUT,
    WAGER_LAYOUT_KEY,
    ConditionalWagerTransformerModel,
    IQNConditionalWagerTransformerModel,
    IQNScalarWagerTransformerModel,
    KellyIQNPPOTorchLearner,
    KellyMeanPPOTorchLearner,
    ScalarWagerTransformerModel,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    actor_mode: str
    critic_mode: str
    wager_layout: str


ARMS = (
    Arm("correctness_mean", CORRECTNESS_MODE, "mean", NO_WAGER_LAYOUT),
    Arm("correctness_iqn", CORRECTNESS_MODE, "iqn", NO_WAGER_LAYOUT),
    Arm("coupled_kelly_mean", COUPLED_MODE, "mean", SCALAR_LAYOUT),
    Arm("coupled_kelly_iqn", COUPLED_MODE, "iqn", SCALAR_LAYOUT),
    Arm("decoupled_kelly_mean", DECOUPLED_MODE, "mean", SCALAR_LAYOUT),
    Arm("decoupled_kelly_iqn", DECOUPLED_MODE, "iqn", SCALAR_LAYOUT),
    Arm(
        "conditional_decoupled_kelly_mean",
        CONDITIONAL_MODE,
        "mean",
        CONDITIONAL_LAYOUT,
    ),
    Arm(
        "conditional_decoupled_kelly_iqn",
        CONDITIONAL_MODE,
        "iqn",
        CONDITIONAL_LAYOUT,
    ),
)
TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
DIRECT_LOSS_WEIGHT = 1.0
IQN_CONFIG = {
    "train_quantiles": 32,
    "value_quantiles": 64,
    "n_cosines": 64,
}
IQN_LOSS_COEFFICIENT = 0.5
IQN_HUBER_KAPPA = 1.0
BASE_MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()
ENV_CONFIG = {
    "model": {
        "factory": "envs.mess3.model:passive_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": (
            "experiments.mess_3_kelly_cycle_1.task:"
            "RawNextTokenTask"
        )
    },
    "observation": {"action": None},
    "delay": 1,
    "episode_length": 512,
    "randomize_first_episode_length": True,
}


def arm_by_name(name: str) -> Arm:
    try:
        return next(arm for arm in ARMS if arm.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown cycle-2 arm {name!r}") from error


def _module_class(arm: Arm):
    if arm.wager_layout == NO_WAGER_LAYOUT:
        return IQNTransformerModel if arm.critic_mode == "iqn" else TransformerModel
    if arm.wager_layout == SCALAR_LAYOUT:
        return (
            IQNScalarWagerTransformerModel
            if arm.critic_mode == "iqn"
            else ScalarWagerTransformerModel
        )
    return (
        IQNConditionalWagerTransformerModel
        if arm.critic_mode == "iqn"
        else ConditionalWagerTransformerModel
    )


def _learner_class(arm: Arm):
    if arm.wager_layout == NO_WAGER_LAYOUT:
        return IQNPPOTorchLearner if arm.critic_mode == "iqn" else None
    return (
        KellyIQNPPOTorchLearner
        if arm.critic_mode == "iqn"
        else KellyMeanPPOTorchLearner
    )


def _learner_config(arm: Arm) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if arm.wager_layout != NO_WAGER_LAYOUT:
        config.update(
            {
                ACTOR_MODE_KEY: arm.actor_mode,
                WAGER_LAYOUT_KEY: arm.wager_layout,
                DIRECT_LOSS_WEIGHT_KEY: DIRECT_LOSS_WEIGHT,
            }
        )
    if arm.critic_mode == "iqn":
        config.update(
            {
                LOSS_COEFFICIENT_KEY: IQN_LOSS_COEFFICIENT,
                HUBER_KAPPA_KEY: IQN_HUBER_KAPPA,
            }
        )
    return config


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


def build_config(context: RunContext, arm_name: str) -> PPOConfig:
    """Build one fresh gamma-zero controlled arm."""

    if context.seed is None:
        raise ValueError("cycle 2 requires a resolved seed")
    arm = arm_by_name(arm_name)
    profile = context.hardware or PROFILES["cpu"]
    model_config = dict(BASE_MODEL_CONFIG)
    if arm.critic_mode == "iqn":
        model_config[IQN_NAMESPACE] = dict(IQN_CONFIG)
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=ENV_CONFIG)
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
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            vf_loss_coeff=0.0 if arm.critic_mode == "iqn" else 0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=(
                2_048 if context.smoke else 32_768
            ),
            minibatch_size=256 if context.smoke else 4_096,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=_module_class(arm),
                model_config=model_config,
            )
        )
        .debugging(seed=context.seed)
    )
    learner_class = _learner_class(arm)
    if learner_class is not None:
        config = config.learners(
            learner_class=learner_class,
            learner_config_dict=_learner_config(arm),
        )
    return _apply_runtime_resources(config, context)


def run_condition(context: RunContext, arm_name: str):
    """Train and analyze one cycle-2 arm."""

    arm = arm_by_name(arm_name)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": arm.name,
            "actor_mode": arm.actor_mode,
            "critic_mode": arm.critic_mode,
            "wager_layout": arm.wager_layout,
            "environment": ENV_CONFIG,
            "model": {
                **BASE_MODEL_CONFIG,
                **(
                    {IQN_NAMESPACE: IQN_CONFIG}
                    if arm.critic_mode == "iqn"
                    else {}
                ),
            },
            "algorithm": "PPO",
            "gamma": 0.0,
            "lambda": 0.0,
            "total_env_steps": target_steps,
            "direct_kelly_loss_weight": (
                DIRECT_LOSS_WEIGHT
                if arm.wager_layout != NO_WAGER_LAYOUT
                else 0.0
            ),
            "predictive_auxiliary_loss_weight": 0.0,
            "warm_start": False,
        },
    )
    result_grid = run_tune(
        build_config(context, arm.name),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"{arm.name} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{arm.name} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{arm.name} produced no final checkpoint")

    probed = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition=arm.name,
    )
    collapse_fraction = probed.metrics["wager_collapse_fraction"]
    collapsed = bool(
        collapse_fraction is not None
        and probed.metrics["wager_mean"] < COLLAPSE_THRESHOLD
        and collapse_fraction > 0.95
    )
    summary = {
        "condition": arm.name,
        "actor_mode": arm.actor_mode,
        "critic_mode": arm.critic_mode,
        "wager_layout": arm.wager_layout,
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
        "\n".join(
            [
                f"# {arm.name.replace('_', ' ').title()}",
                "",
                f"- Held-out belief R²: {probed.metrics['r_squared']:.4f}",
                (
                    "- Greedy token accuracy: "
                    f"{probed.metrics['token_accuracy_greedy']:.4f}"
                ),
                f"- Wager collapse detected: {str(collapsed).lower()}",
                "",
            ]
        )
    )
    return summary
