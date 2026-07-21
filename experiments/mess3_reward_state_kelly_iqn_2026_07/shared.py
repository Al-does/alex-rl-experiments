"""Shared mechanics for the controlled reward-state PPO/Kelly/IQN battery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Real
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
CHECKPOINT_INTERVAL = 306
CHECKPOINT_MILESTONES = (10_000_000, 20_000_000, 30_000_000)
CHECKPOINT_TOLERANCE_STEPS = 1_000_000
ACTION_LIMIT = 5.0
TRAIN_BATCH_SIZE = 32_768
MINIBATCH_SIZE = 2_048
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
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=ENV_CONFIG)
        .framework(
            "torch",
            torch_compile_learner=False,
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


def _metric(metrics: Mapping[str, Any], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, Real):
        return float(direct)
    value: Any = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return float(value) if isinstance(value, Real) else None


def checkpoint_records(result: Any) -> list[dict[str, Any]]:
    """Return retained checkpoints ordered by sampled environment steps."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for checkpoint, metrics in result.best_checkpoints or []:
        checkpoint_path = str(checkpoint.path)
        if checkpoint_path in seen:
            continue
        steps = _metric(metrics, "env_runners/num_env_steps_sampled_lifetime")
        iteration = _metric(metrics, "training_iteration")
        if steps is None or iteration is None:
            continue
        seen.add(checkpoint_path)
        records.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_name": Path(checkpoint_path).name,
                "training_iteration": int(iteration),
                "agent_steps": int(steps),
            }
        )
    if not records and result.checkpoint is not None:
        steps = _metric(
            result.metrics or {},
            "env_runners/num_env_steps_sampled_lifetime",
        )
        iteration = _metric(result.metrics or {}, "training_iteration")
        if steps is not None and iteration is not None:
            records.append(
                {
                    "checkpoint": result.checkpoint,
                    "checkpoint_name": Path(result.checkpoint.path).name,
                    "training_iteration": int(iteration),
                    "agent_steps": int(steps),
                }
            )
    return sorted(records, key=lambda record: record["agent_steps"])


def _probe_checkpoints(
    context: RunContext,
    checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not checkpoints:
        raise RuntimeError("reward-state battery retained no checkpoints")
    if not context.smoke:
        if len(checkpoints) != len(CHECKPOINT_MILESTONES):
            raise RuntimeError(
                "reward-state battery expected checkpoints near 10M, 20M, "
                f"and 30M steps, got {len(checkpoints)}"
            )
        for record, milestone in zip(checkpoints, CHECKPOINT_MILESTONES):
            if (
                abs(record["agent_steps"] - milestone)
                > CHECKPOINT_TOLERANCE_STEPS
            ):
                raise RuntimeError(
                    f"checkpoint at {record['agent_steps']:,} steps is not "
                    f"within {CHECKPOINT_TOLERANCE_STEPS:,} steps of "
                    f"{milestone:,}"
                )

    probes: list[dict[str, Any]] = []
    for index, record in enumerate(checkpoints):
        is_final = index == len(checkpoints) - 1
        probe_results_dir = (
            context.results_dir
            if is_final
            else (
                context.results_dir
                / "checkpoint_probes"
                / f"steps_{record['agent_steps']:09d}"
            )
        )
        probe_results_dir.mkdir(parents=True, exist_ok=True)
        probe = checkpoint_probe.run(
            replace(
                context,
                results_dir=probe_results_dir,
                resume_from=Path(record["checkpoint"].path),
            )
        )
        probes.append(
            {
                "checkpoint_name": record["checkpoint_name"],
                "training_iteration": record["training_iteration"],
                "agent_steps": record["agent_steps"],
                "reward_percentage": (
                    100.0 * float(probe["occupancy_state_2_fraction"])
                ),
                "greedy_reward_percentage": (
                    100.0 * float(probe["reward_greedy"])
                ),
                "r2_global": float(probe["r2_global"]),
                "r2_fine": float(probe["r2_fine"]),
                "target_consistency_max_abs": float(
                    probe["target_consistency_max_abs"]
                ),
                "probe": probe,
            }
        )
    return probes


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
        "checkpoint_interval_iterations": (
            1 if context.smoke else CHECKPOINT_INTERVAL
        ),
        "checkpoint_milestones": (
            [target_steps] if context.smoke else list(CHECKPOINT_MILESTONES)
        ),
        "checkpoint_tolerance_steps": (
            target_steps if context.smoke else CHECKPOINT_TOLERANCE_STEPS
        ),
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
                num_to_keep=3,
                checkpoint_frequency=1 if context.smoke else CHECKPOINT_INTERVAL,
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
    checkpoints = checkpoint_records(result)
    probes = _probe_checkpoints(context, checkpoints)
    final = probes[-1]
    compact_checkpoints = [
        {
            key: value
            for key, value in record.items()
            if key != "checkpoint"
        }
        for record in checkpoints
    ]
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {
            "condition": condition,
            "checkpoints": probes,
        },
    )
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "gamma": gamma,
        "critic": recipe["critic"],
        "predictive_kelly": use_kelly,
        "reward_percentage": final["reward_percentage"],
        "greedy_reward_percentage": final["greedy_reward_percentage"],
        "r2_global": final["r2_global"],
        "r2_fine": final["r2_fine"],
        "checkpoint_index": compact_checkpoints,
        "checkpoint_probes": probes,
        "probe": final["probe"],
    }
    outputs.write_json("condition_summary.json", summary)
    (context.results_dir / "findings.md").write_text(
        "\n".join(
            [
                f"# {condition.replace('_', ' ').title()}",
                "",
                f"- State-2 reward percentage: {final['reward_percentage']:.2f}%",
                (
                    "- Greedy state-2 reward percentage: "
                    f"{final['greedy_reward_percentage']:.2f}%"
                ),
                f"- Transducer belief global R²: {final['r2_global']:.4f}",
                f"- Transducer belief fine R²: {final['r2_fine']:.4f}",
                (
                    "- Checkpoints probed at environment steps: "
                    + ", ".join(f"{probe['agent_steps']:,}" for probe in probes)
                ),
                "",
            ]
        )
    )
    return summary
