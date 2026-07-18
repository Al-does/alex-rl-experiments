"""Experiment-repo integration checks that depend on named recipes."""

from __future__ import annotations

import json
from pathlib import Path

from harness.context import RunContext
from harness.hardware import PROFILES


def make_context(tmp_path: Path, name: str) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / name / "results",
        artifacts_dir=tmp_path / name / "artifacts",
        run_id=name,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_representative_supervised_experiment_run(tmp_path):
    from experiments.mess3_belief_geometry_2026_07.state_guess_supervised import (
        experiment,
    )

    context = make_context(tmp_path, "supervised")
    module = experiment.run(context)

    summary = json.loads(
        context.results_dir.joinpath("summary.json").read_text()
    )
    assert summary["env_steps"] >= 8192
    assert summary["target"] == "state"
    assert context.artifacts_dir.joinpath(
        "checkpoints", "module_state_final.pt"
    ).is_file()
    assert next(module.parameters()).device.type == "cpu"
