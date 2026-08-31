"""Split-network PPO with detached policy entropy in the reward stream."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from experiments.factored_representations_reproduction_PPO_2026_08.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.factored_representations_reproduction_PPO_2026_08.process import (
    FACTOR_COUNTS,
)
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    SMOKE_ENV_STEPS,
    checkpoint_records,
)
from experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08 import (
    shared as split_ppo,
)
from experiments.mess3_token_guess_cycle_1.entropy_reward import (
    COEFFICIENT_KEY,
    EntropyRewardPPOTorchLearner,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune


CONDITION = "ppo_max_entropy"
TOTAL_ENV_STEPS = 10_000_000
ENTROPY_REWARD_COEFFICIENT = 0.5
PPO_ENTROPY_COEFFICIENT = 0.0


def build_config(context: RunContext, *, factor_count: int) -> PPOConfig:
    """Build a fresh baseline split-PPO config with reward entropy shaping."""

    config = split_ppo.build_config(
        context,
        factor_count=factor_count,
        condition="ppo",
    )
    return config.training(
        entropy_coeff=PPO_ENTROPY_COEFFICIENT,
    ).learners(
        learner_class=EntropyRewardPPOTorchLearner,
        learner_config_dict={
            COEFFICIENT_KEY: ENTROPY_REWARD_COEFFICIENT,
        },
    )


def _resolved_recipe(
    *,
    factor_count: int,
    context: RunContext,
) -> dict[str, Any]:
    recipe = split_ppo._resolved_recipe(
        factor_count=factor_count,
        condition="ppo",
        context=context,
    )
    return {
        **recipe,
        "condition": CONDITION,
        "total_env_steps": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS,
        "objective": (
            "PPO with detached behavior-policy entropy added to rewards before GAE"
        ),
        "entropy_reward_coefficient": ENTROPY_REWARD_COEFFICIENT,
        "ppo_entropy_coefficient": PPO_ENTROPY_COEFFICIENT,
        "entropy_design": (
            "Detached behavior-policy entropy is added before GAE so the critic "
            "fits entropy-augmented returns. PPO's differentiable actor entropy "
            "coefficient remains zero to match the comparison runs. Because "
            "gamma=0, transitions are action-independent, and actor/critic "
            "parameters are disjoint, this is a reward-stream ablation rather "
            "than a direct actor maximum-entropy gradient."
        ),
    }


def _run_factor(context: RunContext, *, factor_count: int) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("the reproduction requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("split-PPO continuation is not defined for this recipe")

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        _resolved_recipe(factor_count=factor_count, context=context),
    )
    result_grid = run_tune(
        build_config(context, factor_count=factor_count),
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
    if len(results) != 1:
        raise RuntimeError(f"expected one Tune trial, found {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError("split PPO reward-entropy training failed") from result.error
    write_training_curves(context)

    records = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *checkpoint_records(
            result,
            checkpoint_root=context.artifacts_dir / "log_spaced_checkpoints",
        ),
    ]
    reports = []
    for record in records:
        probe_context = replace(
            context,
            results_dir=(
                context.results_dir
                / "checkpoint_probes"
                / f"steps_{record['agent_steps']:09d}"
            ),
            resume_from=Path(record["checkpoint_path"]),
        )
        reports.append(
            analyze_checkpoint(
                probe_context,
                checkpoint=Path(record["checkpoint_path"]),
                factor_count=factor_count,
                condition=CONDITION,
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    plot_probe_trajectory(
        reports,
        condition=CONDITION,
        factor_count=factor_count,
        path=context.results_dir / "probe_trajectory.png",
    )
    return {
        "condition": CONDITION,
        "factor_count": factor_count,
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoint_reports": [
            {
                "agent_steps": report["agent_steps"],
                "checkpoint": report["checkpoint"],
                "training_iteration": report["training_iteration"],
                "path": str(
                    context.results_dir
                    / "checkpoint_probes"
                    / f"steps_{report['agent_steps']:09d}"
                    / "probe_battery.json"
                ),
            }
            for report in reports
        ],
        "trajectory_figure": str(context.results_dir / "probe_trajectory.png"),
    }


def run(context: RunContext) -> dict[str, Any]:
    """Run matched two- and three-factor reward-entropy cells."""

    summaries = {}
    for factor_count in FACTOR_COUNTS:
        factor_context = replace(
            context,
            results_dir=context.results_dir / f"{factor_count}_factors",
            artifacts_dir=context.artifacts_dir / f"{factor_count}_factors",
            resume_from=None,
        )
        summaries[str(factor_count)] = _run_factor(
            factor_context,
            factor_count=factor_count,
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    result = {
        "condition": CONDITION,
        "seed": context.seed,
        "smoke": context.smoke,
        "factor_conditions": summaries,
    }
    outputs.write_json("arm_summary.json", result)
    return result
