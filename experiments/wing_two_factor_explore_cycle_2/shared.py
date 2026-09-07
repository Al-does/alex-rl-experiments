from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.shared import checkpoint_records
from experiments.storage.training_curves import write_training_curves
from experiments.wing_two_factor_explore_cycle_1 import shared as cycle_1
from experiments.wing_two_factor_explore_cycle_1.design import design_summary
from experiments.wing_two_factor_explore_cycle_2.analysis import analyze_checkpoint
from experiments.wing_two_factor_explore_cycle_2.process import (
    CONTROL_STRENGTH,
    REWARD_STATE,
    WING_ALPHA,
    WING_X,
    environment_config,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune


VALUE_CLIP_PARAM = 1e9
TOTAL_ENV_STEPS = dict(cycle_1.TOTAL_ENV_STEPS)
SMOKE_ENV_STEPS = cycle_1.SMOKE_ENV_STEPS


def build_config(context: RunContext, condition: str) -> PPOConfig:
    return (
        cycle_1.build_config(context, condition, REWARD_STATE)
        .environment(HMMEnv, env_config=environment_config(condition))
        .training(vf_clip_param=VALUE_CLIP_PARAM)
    )


def resolved_recipe(context: RunContext, condition: str) -> dict[str, Any]:
    return {
        **cycle_1.resolved_recipe(context, condition, REWARD_STATE),
        "study": "wing_two_factor_explore_cycle_2",
        "environment": environment_config(condition),
        "value_clip_param": VALUE_CLIP_PARAM,
        "value_clip_semantics": "effectively uncapped squared value error; global gradient clipping retained",
        "analytic_design": design_summary(
            alpha=WING_ALPHA, x=WING_X, strength=CONTROL_STRENGTH,
        ),
        "control_semantics": "hold or deterministically rotate the stochastic Wing edge destination by +1/-1 modulo 3",
        "changes_from_cycle_1": {
            "reward_placements": [REWARD_STATE],
            "control_strength": {"before": 0.15, "after": CONTROL_STRENGTH},
            "vf_clip_param": {"before": 10.0, "after": VALUE_CLIP_PARAM},
        },
        "total_env_steps": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS[condition],
    }


def run_condition(context: RunContext, condition: str) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("Wing PPO requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", resolved_recipe(context, condition))
    result_grid = run_tune(
        build_config(context, condition),
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
        raise RuntimeError(f"{condition}: cycle 2 PPO training failed")
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
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    summary = {
        "condition": condition,
        "reward_state": REWARD_STATE,
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
