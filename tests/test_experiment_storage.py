"""Tests for experiment-owned storage conventions."""

from __future__ import annotations

import json

import pytest

from experiments.storage import CurveField
from experiments.storage.training_curves import (
    TRAINING_CURVES_FILENAME,
    _COMPACT_FIELD_SOURCES,
    compact_training_curve_row,
    write_training_curves,
)
from harness.artifacts import RunArtifacts


def test_compact_field_sources_are_generic():
    # A failure means the shared table changed; study metrics belong in extra_fields.
    assert {target for _, target in _COMPACT_FIELD_SOURCES} == {
        "iteration",
        "steps",
        "return_mean",
        "return_min",
        "return_max",
        "entropy",
        "entropy_coeff",
        "policy_loss",
        "vf_loss",
        "total_loss",
        "value_explained_variance",
        "mean_kl",
        "time_iter_s",
        "time_total_s",
        "done",
        "trial_id",
    }


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


def test_compact_training_curve_row_projects_extra_fields():
    extra_fields: tuple[CurveField, ...] = (
        (("env_runners/study/value", "study/value"), "study_value"),
    )
    row = compact_training_curve_row(
        {"env_runners/study/value": None, "study/value": 3},
        extra_fields=extra_fields,
    )
    assert row == {"study_value": 3}


def test_compact_training_curve_row_rejects_duplicate_targets():
    with pytest.raises(ValueError, match="duplicate compact training-curve target"):
        compact_training_curve_row(
            {"custom/value": 1},
            extra_fields=((("custom/value",), "iteration"),),
        )
    with pytest.raises(ValueError, match="duplicate compact training-curve target"):
        compact_training_curve_row(
            {"custom/value": 1},
            extra_fields=(
                (("custom/value",), "study_value"),
                (("custom/other",), "study_value"),
            ),
        )


def test_compact_training_curve_row_does_not_project_study_fields_by_default():
    row = compact_training_curve_row(
        {
            "training_iteration": 2,
            "env_runners/guess_or_pass/passed": 0.25,
        }
    )
    assert row == {"iteration": 2}


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


def test_write_training_curves_validates_extra_targets_before_reading_rows(tmp_path):
    results = tmp_path / "results" / "run"
    artifacts = tmp_path / "artifacts" / "run"
    run = RunArtifacts(results, artifacts)
    artifacts.mkdir(parents=True)
    (artifacts / "metrics.jsonl").write_text("not-json\n")

    with pytest.raises(ValueError, match="duplicate compact training-curve target"):
        write_training_curves(
            run,
            extra_fields=((("custom/value",), "iteration"),),
        )
