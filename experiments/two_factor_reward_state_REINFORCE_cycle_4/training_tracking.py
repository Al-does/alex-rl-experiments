"""Lightweight reward/occupancy tracking without checkpoint probe batteries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from harness.artifacts import RunArtifacts
from harness.context import RunContext

REWARD_OCCUPANCY_CURVE_FILENAME = "reward_occupancy_curve.jsonl"
CHECKPOINT_OCCUPANCY_FILENAME = "checkpoint_occupancy.jsonl"


def _metric(metrics: Mapping[str, Any], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, Real):
        number = float(direct)
        return number if number == number else None
    value: Any = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if not isinstance(value, Real):
        return None
    number = float(value)
    return number if number == number else None


def occupancy_fraction_from_metrics(
    metrics: Mapping[str, Any],
    *,
    condition: str,
) -> float | None:
    """Infer per-step reward occupancy from on-policy training rollouts."""
    del condition
    return_mean = _metric(metrics, "env_runners/episode_return_mean")
    episode_len = _metric(metrics, "env_runners/episode_len_mean")
    if return_mean is None or episode_len is None or episode_len <= 0:
        return None
    return float(return_mean) / float(episode_len)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def record_step_occupancy_snapshot(
    context: RunContext,
    result: Mapping[str, Any],
    *,
    condition: str,
    step_interval: int,
) -> None:
    """Append one occupancy row when lifetime env steps cross a checkpoint boundary."""
    steps_value = _metric(result, "env_runners/num_env_steps_sampled_lifetime")
    iteration_value = _metric(result, "training_iteration")
    if steps_value is None or iteration_value is None:
        return
    steps = int(steps_value)
    boundary = (steps // step_interval) * step_interval
    if boundary <= 0:
        return
    path = context.results_dir / CHECKPOINT_OCCUPANCY_FILENAME
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            if int(json.loads(line)["agent_steps"]) == boundary:
                return
    occupancy_fraction = occupancy_fraction_from_metrics(result, condition=condition)
    if occupancy_fraction is None:
        return
    _append_jsonl(
        path,
        {
            "agent_steps": boundary,
            "training_iteration": int(iteration_value),
            "episode_return_mean": _metric(result, "env_runners/episode_return_mean"),
            "episode_len_mean": _metric(result, "env_runners/episode_len_mean"),
            "occupancy_fraction": occupancy_fraction,
            "occupancy_pct": 100.0 * occupancy_fraction,
            "condition": condition,
            "source": "training_rollout",
        },
    )


def write_reward_occupancy_curve(
    context: RunContext | RunArtifacts,
    *,
    condition: str,
) -> Path | None:
    """Materialize per-iteration occupancy from verbose training metrics."""
    artifacts = (
        RunArtifacts.from_context(context)
        if isinstance(context, RunContext)
        else context
    )
    source_candidates = [
        artifacts.artifacts_dir / "metrics.jsonl",
        artifacts.results_dir / "progress.jsonl",
        artifacts.artifacts_dir / "progress.jsonl",
    ]
    source = next((path for path in source_candidates if path.is_file()), None)
    if source is None:
        return None

    curves_path = artifacts.results_dir / REWARD_OCCUPANCY_CURVE_FILENAME
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for raw in source.read_text().splitlines():
        if not raw.strip():
            continue
        flattened = json.loads(raw)
        occupancy_fraction = occupancy_fraction_from_metrics(
            flattened,
            condition=condition,
        )
        if occupancy_fraction is None:
            continue
        steps = _metric(flattened, "env_runners/num_env_steps_sampled_lifetime")
        iteration = _metric(flattened, "training_iteration")
        row = {
            "steps": int(steps) if steps is not None else None,
            "iteration": int(iteration) if iteration is not None else None,
            "return_mean": _metric(flattened, "env_runners/episode_return_mean"),
            "episode_len_mean": _metric(flattened, "env_runners/episode_len_mean"),
            "occupancy_fraction": occupancy_fraction,
            "occupancy_pct": 100.0 * occupancy_fraction,
            "condition": condition,
            "source": "training_rollout",
        }
        lines.append(json.dumps(row, sort_keys=True))
    curves_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return curves_path
