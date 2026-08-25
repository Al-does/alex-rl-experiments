"""Tests for experiment-owned storage conventions."""

from __future__ import annotations

import json

from experiments.storage.training_curves import (
    TRAINING_CURVES_FILENAME,
    compact_training_curve_row,
    write_training_curves,
)
from harness.artifacts import RunArtifacts


def test_compact_training_curve_row_projects_short_keys():
    flattened = {
        "training_iteration": 3,
        "num_env_steps_sampled_lifetime": 98304.0,
        "env_runners/episode_return_mean": 12.5,
        "learners/default_policy/entropy": 1.2,
        "time_this_iter_s": 4.5,
        "env_runners/env_step_timer": 0.001,
    }
    row = compact_training_curve_row(flattened)
    assert row == {
        "iteration": 3,
        "steps": 98304.0,
        "return_mean": 12.5,
        "entropy": 1.2,
        "time_iter_s": 4.5,
    }


def test_write_training_curves_materializes_from_artifact_metrics(tmp_path):
    results = tmp_path / "results" / "run"
    artifacts = tmp_path / "artifacts" / "run"
    run = RunArtifacts(results, artifacts)
    run.append_result(
        {
            "training_iteration": 1,
            "env_runners": {"episode_return_mean": 5.0},
            "num_env_steps_sampled_lifetime": 32768,
        }
    )
    path = write_training_curves(run)
    assert path == results / TRAINING_CURVES_FILENAME
    row = json.loads(path.read_text().strip())
    assert row["iteration"] == 1
    assert row["return_mean"] == 5.0
