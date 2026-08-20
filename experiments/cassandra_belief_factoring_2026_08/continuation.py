"""Experiment-local mechanics for restoring a checkpoint with a new config."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ray.rllib.algorithms.algorithm import Algorithm

from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_algorithm


@dataclass(frozen=True)
class RestoringAlgorithmConfig:
    """Build an Algorithm from a checkpoint while overriding its config."""

    config: Any
    checkpoint: Path

    @property
    def algo_class(self):
        return self.config.algo_class

    def build_algo(self):
        return Algorithm.from_checkpoint(
            str(self.checkpoint.resolve()),
            config=self.config,
        )

    def to_dict(self):
        return self.config.to_dict()


def continue_from_checkpoint(
    config,
    context: RunContext,
    *,
    source_run_id: str,
    source_steps: int,
    additional_steps: int,
):
    """Run a fresh smoke or restore through an additional-step target."""

    if context.smoke:
        runtime_config = config
        runtime_context = context
        target_steps = 4_096
        recorded_source_steps = 0
    else:
        if context.resume_from is None:
            raise ValueError(
                "full continuation requires --resume-from with the source "
                "Algorithm checkpoint"
            )
        if source_run_id not in str(context.resume_from):
            raise ValueError(
                "resume checkpoint does not match declared source run "
                f"{source_run_id}: {context.resume_from}"
            )
        runtime_config = RestoringAlgorithmConfig(
            config=config,
            checkpoint=context.resume_from,
        )
        runtime_context = replace(context, resume_from=None)
        target_steps = source_steps + additional_steps
        recorded_source_steps = source_steps

    first_result = True

    def should_stop(metrics) -> bool:
        nonlocal first_result
        steps = _sampled_steps(metrics)
        if first_result and not context.smoke:
            expected = source_steps + int(
                config.train_batch_size_per_learner
            )
            if steps != expected:
                raise ValueError(
                    "restored checkpoint sampled-step counter does not match "
                    f"source run: expected first result at {expected}, got {steps}"
                )
        first_result = False
        return steps >= target_steps

    result = run_algorithm(
        runtime_config,
        runtime_context,
        should_stop=should_stop,
        checkpoint_at_end=True,
    )
    RunArtifacts.from_context(context).write_json(
        "continuation_summary.json",
        {
            "source_run_id": source_run_id,
            "source_checkpoint": context.resume_from,
            "source_steps": recorded_source_steps,
            "additional_requested_steps": (
                4_096 if context.smoke else additional_steps
            ),
            "target_steps": target_steps,
            "final_sampled_steps": _sampled_steps(result),
        },
    )
    return result


def _sampled_steps(result) -> int:
    """Read RLlib's lifetime sampled-step counter."""

    runners = result.get("env_runners", {})
    value = (
        runners.get("num_env_steps_sampled_lifetime")
        if isinstance(runners, dict)
        else None
    )
    if value is None:
        value = result.get("num_env_steps_sampled_lifetime")
    if value is None:
        raise KeyError("result has no lifetime sampled-step metric")
    return int(value)


__all__ = [
    "RestoringAlgorithmConfig",
    "continue_from_checkpoint",
]
