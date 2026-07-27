"""Train matched arms at both operating points and compare the separation.

``process_design`` predicts, without training anything, that the shipped MESS3
parameters leave both reported axes almost fully occupied by floors. This leaf
tests that prediction the only way it can be tested: by holding the recipe,
architecture, budget, seeds, and probe fixed and changing nothing but the
process.

Two arms are enough, because the shipped study already showed them at opposite
ends: plain PPO landed below the untrained-network floor while IQN was the only
family to clear it. If the proposed point is the better instrument, the same
two arms should separate by more of the range that is actually available.

Every cell writes its own summary as it finishes, so a partial run is still
usable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_1.analysis import probe_checkpoint
from experiments.mess3_token_guess_cycle_1.comparison.experiment import (
    BASE_MODEL_CONFIG,
    _apply_runtime_resources,
)
from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    NAMESPACE as IQN_NAMESPACE,
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from experiments.mess3_token_guess_cycle_1.operating_points import (
    POINTS,
    OperatingPoint,
    point_by_name,
)
from experiments.mess3_token_guess_cycle_1.untrained_reference.experiment import (
    probe_untrained_module,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune
from learners.models.transformer import TransformerModel


TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
IQN_CONFIG = {"train_quantiles": 32, "value_quantiles": 64, "n_cosines": 64}
IQN_LOSS_COEFFICIENT = 0.5
IQN_HUBER_KAPPA = 1.0


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    critic: str


ARMS = (Arm("ppo", "mean"), Arm("iqn", "iqn"))


def build_config(
    context: RunContext,
    point: OperatingPoint,
    arm_name: str,
) -> PPOConfig:
    """Build one cell; only the environment differs between operating points."""

    arm = next(candidate for candidate in ARMS if candidate.name == arm_name)
    model_config = dict(BASE_MODEL_CONFIG)
    if arm.critic == "iqn":
        model_config[IQN_NAMESPACE] = dict(IQN_CONFIG)
    profile = context.hardware
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=point.env_config())
        .framework(
            "torch",
            torch_compile_learner=(
                not context.smoke
                and profile is not None
                and profile.learner_device == "cuda"
            ),
            torch_compile_learner_what_to_compile="forward_train",
            torch_compile_learner_dynamo_backend="inductor",
            torch_compile_learner_dynamo_mode="reduce-overhead",
            torch_compile_worker=False,
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.0 if arm.critic == "iqn" else 0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=2_048 if context.smoke else 32_768,
            minibatch_size=256 if context.smoke else 4_096,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=(
                    IQNTransformerModel
                    if arm.critic == "iqn"
                    else TransformerModel
                ),
                model_config=model_config,
            )
        )
        .debugging(seed=context.seed)
    )
    if arm.critic == "iqn":
        config = config.learners(
            learner_class=IQNPPOTorchLearner,
            learner_config_dict={
                LOSS_COEFFICIENT_KEY: IQN_LOSS_COEFFICIENT,
                HUBER_KAPPA_KEY: IQN_HUBER_KAPPA,
            },
        )
    return _apply_runtime_resources(config, context)


def _cell_context(
    context: RunContext,
    point: OperatingPoint,
    name: str,
) -> RunContext:
    return replace(
        context,
        results_dir=context.results_dir / point.name / name,
        artifacts_dir=context.artifacts_dir / point.name / name,
        resume_from=None,
    )


def _train_cell(
    context: RunContext,
    point: OperatingPoint,
    arm: Arm,
) -> dict[str, Any]:
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    result_grid = run_tune(
        build_config(context, point, arm.name),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1, checkpoint_at_end=True
            )
        },
    )
    (context.results_dir / "tune_summary.json").unlink(missing_ok=True)
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
        condition=f"{point.name}/{arm.name}",
    )
    summary = {
        "operating_point": point.name,
        "stay": point.stay,
        "alpha": point.alpha,
        "arm": arm.name,
        "seed": context.seed,
        "target_agent_steps": target_steps,
        "probe": probed.metrics,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


def _untrained_cell(
    context: RunContext,
    point: OperatingPoint,
) -> dict[str, Any]:
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    probe = probe_untrained_module(
        context,
        initialisation=0,
        env_config=point.env_config(),
    )
    summary = {
        "operating_point": point.name,
        "stay": point.stay,
        "alpha": point.alpha,
        "arm": "untrained",
        "seed": context.seed,
        "probe": probe,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


def run_points(context: RunContext, points: tuple[OperatingPoint, ...]):
    """Train and probe every arm at each supplied operating point."""

    if context.seed is None:
        raise ValueError("the operating-point validation requires a seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "operating_points": [
                {
                    "name": point.name,
                    "stay": point.stay,
                    "alpha": point.alpha,
                    "environment": point.env_config(),
                }
                for point in points
            ],
            "arms": [arm.name for arm in ARMS],
            "model": BASE_MODEL_CONFIG,
            "algorithm": "PPO",
            "gamma": 0.99,
            "total_env_steps": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
        },
    )

    cells = []
    for point in points:
        cells.append(
            _untrained_cell(_cell_context(context, point, "untrained"), point)
        )
        for arm in ARMS:
            cells.append(
                _train_cell(
                    _cell_context(context, point, arm.name), point, arm
                )
            )
        (context.results_dir / "cells.json").write_text(
            json.dumps(cells, indent=2) + "\n"
        )
    return {"seed": context.seed, "cells": cells}


def run(context: RunContext):
    return run_points(context, POINTS)


def load_cells(*results_roots: Path) -> list[dict[str, Any]]:
    """Collect every finished cell across seeds, tolerating partial runs."""

    return [
        json.loads(path.read_text())
        for root in results_roots
        for path in sorted(root.glob("*/*/*/condition_summary.json"))
    ]
