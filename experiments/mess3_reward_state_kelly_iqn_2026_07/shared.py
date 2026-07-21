"""Shared mechanics for the controlled reward-state PPO/Kelly/IQN battery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.checkpoint_probe import (
    experiment as checkpoint_probe,
)
from experiments.mess3_belief_geometry_2026_07.shared import (
    SMOKE_ENV_STEPS,
    apply_runtime_resources,
    next_visible_token_targets,
)
from experiments.mess3_reward_state_cycle_1.iqn import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    NAMESPACE as IQN_NAMESPACE,
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from experiments.mess3_reward_state_kelly_iqn_2026_07.kelly import (
    CORRECTNESS_COEFFICIENT_KEY,
    DIRECT_LOSS_COEFFICIENT_KEY,
    NAMESPACE as KELLY_NAMESPACE,
    TARGET_EXTRACTOR_KEY,
    PredictiveKellyHead,
    PredictiveKellyLossMixin,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


class KellyTransformerModel(PredictiveKellyHead, TransformerModel):
    """Continuous-control transformer with predictive token/wager heads."""


class KellyIQNTransformerModel(PredictiveKellyHead, IQNTransformerModel):
    """Kelly continuous-control transformer with an IQN value critic."""


class KellyPPOTorchLearner(PredictiveKellyLossMixin, PPOTorchLearner):
    """Mean-value PPO plus predictive correctness and Kelly objectives."""


class KellyIQNPPOTorchLearner(PredictiveKellyLossMixin, IQNPPOTorchLearner):
    """IQN PPO plus predictive correctness and Kelly objectives."""


TOTAL_ENV_STEPS = 30_000_000
ACTION_LIMIT = 5.0
TRAIN_BATCH_SIZE = 65_536
MINIBATCH_SIZE = 4_096
LEARNING_RATE = 4.2e-4
TOKEN_CORRECTNESS_COEFFICIENT = 1.0
DIRECT_KELLY_LOSS_COEFFICIENT = 1.0
IQN_LOSS_COEFFICIENT = 0.5
IQN_HUBER_KAPPA = 1.0
IQN_CONFIG = {
    "train_quantiles": 32,
    "value_quantiles": 64,
    "n_cosines": 64,
}
BASE_MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()
ENV_CONFIG = {
    "model": {
        "factory": "envs.mess3.model:control_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": (
            "envs.mess3.tasks.occupancy_control:"
            "OccupancyControlTask"
        ),
        "kwargs": {"action_limit": ACTION_LIMIT},
    },
    "delay": 1,
    "episode_length": 1024,
    "randomize_first_episode_length": True,
}


def _model_class(*, use_iqn: bool, use_kelly: bool):
    if use_kelly:
        return KellyIQNTransformerModel if use_iqn else KellyTransformerModel
    return IQNTransformerModel if use_iqn else TransformerModel


def _learner_class(*, use_iqn: bool, use_kelly: bool):
    if use_kelly:
        return KellyIQNPPOTorchLearner if use_iqn else KellyPPOTorchLearner
    return IQNPPOTorchLearner if use_iqn else PPOTorchLearner


def _model_config(*, use_iqn: bool, use_kelly: bool) -> dict[str, Any]:
    config: dict[str, Any] = dict(BASE_MODEL_CONFIG)
    if use_iqn:
        config[IQN_NAMESPACE] = dict(IQN_CONFIG)
    if use_kelly:
        config[KELLY_NAMESPACE] = {"num_tokens": 3}
    return config


def _learner_config(*, use_iqn: bool, use_kelly: bool) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if use_iqn:
        config.update(
            {
                LOSS_COEFFICIENT_KEY: IQN_LOSS_COEFFICIENT,
                HUBER_KAPPA_KEY: IQN_HUBER_KAPPA,
            }
        )
    if use_kelly:
        config.update(
            {
                CORRECTNESS_COEFFICIENT_KEY: TOKEN_CORRECTNESS_COEFFICIENT,
                DIRECT_LOSS_COEFFICIENT_KEY: DIRECT_KELLY_LOSS_COEFFICIENT,
                TARGET_EXTRACTOR_KEY: next_visible_token_targets,
            }
        )
    return config


def build_config(
    context: RunContext,
    *,
    gamma: float,
    use_iqn: bool,
    use_kelly: bool,
) -> PPOConfig:
    """Build one fresh, controlled configuration for a battery condition."""

    if context.seed is None:
        raise ValueError("reward-state battery requires a resolved seed")
    if gamma not in (0.0, 0.99):
        raise ValueError("reward-state battery gamma must be 0 or 0.99")
    profile = context.hardware or PROFILES["cpu"]
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
        .learners(
            learner_class=_learner_class(
                use_iqn=use_iqn,
                use_kelly=use_kelly,
            ),
            learner_config_dict=_learner_config(
                use_iqn=use_iqn,
                use_kelly=use_kelly,
            ),
        )
        .training(
            lr=3e-4 if context.smoke else LEARNING_RATE,
            gamma=gamma,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.0 if use_iqn else 0.5,
            entropy_coeff=0.003,
            train_batch_size_per_learner=(
                2_048 if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=256 if context.smoke else MINIBATCH_SIZE,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=_model_class(
                    use_iqn=use_iqn,
                    use_kelly=use_kelly,
                ),
                model_config=_model_config(
                    use_iqn=use_iqn,
                    use_kelly=use_kelly,
                ),
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
    gamma: float,
    use_iqn: bool,
    use_kelly: bool,
) -> dict[str, Any]:
    """Train one condition and run the action-aware transducer belief probe."""

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    recipe = {
        "condition": condition,
        "environment": ENV_CONFIG,
        "algorithm": "PPO",
        "gamma": gamma,
        "lambda": 0.95,
        "critic": "implicit_quantile_network" if use_iqn else "scalar_mean",
        "predictive_kelly": use_kelly,
        "token_correctness_coefficient": (
            TOKEN_CORRECTNESS_COEFFICIENT if use_kelly else 0.0
        ),
        "direct_kelly_loss_coefficient": (
            DIRECT_KELLY_LOSS_COEFFICIENT if use_kelly else 0.0
        ),
        "kelly_net_win_odds": 2.0 if use_kelly else None,
        "iqn_config": IQN_CONFIG if use_iqn else None,
        "iqn_loss_coefficient": IQN_LOSS_COEFFICIENT if use_iqn else 0.0,
        "total_env_steps": target_steps,
        "train_batch_size": (
            2_048 if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": 256 if context.smoke else MINIBATCH_SIZE,
        "learning_rate": 3e-4 if context.smoke else LEARNING_RATE,
        "model_config": _model_config(
            use_iqn=use_iqn,
            use_kelly=use_kelly,
        ),
        "belief_probe_target": "predictive_transducer_belief",
    }
    outputs.write_json("resolved_recipe.json", recipe)

    result_grid = run_tune(
        build_config(
            context,
            gamma=gamma,
            use_iqn=use_iqn,
            use_kelly=use_kelly,
        ),
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
        raise RuntimeError(f"{condition} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{condition} produced no final checkpoint")

    probe = checkpoint_probe.run(
        replace(
            context,
            resume_from=Path(result.checkpoint.path),
        )
    )
    reward_percentage = 100.0 * float(probe["occupancy_state_2_fraction"])
    greedy_reward_percentage = 100.0 * float(probe["reward_greedy"])
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "gamma": gamma,
        "critic": recipe["critic"],
        "predictive_kelly": use_kelly,
        "reward_percentage": reward_percentage,
        "greedy_reward_percentage": greedy_reward_percentage,
        "r2_global": float(probe["r2_global"]),
        "r2_fine": float(probe["r2_fine"]),
        "probe": probe,
    }
    outputs.write_json("condition_summary.json", summary)
    (context.results_dir / "findings.md").write_text(
        "\n".join(
            [
                f"# {condition.replace('_', ' ').title()}",
                "",
                f"- State-2 reward percentage: {reward_percentage:.2f}%",
                (
                    "- Greedy state-2 reward percentage: "
                    f"{greedy_reward_percentage:.2f}%"
                ),
                f"- Transducer belief global R²: {probe['r2_global']:.4f}",
                f"- Transducer belief fine R²: {probe['r2_fine']:.4f}",
                "",
            ]
        )
    )
    return summary
