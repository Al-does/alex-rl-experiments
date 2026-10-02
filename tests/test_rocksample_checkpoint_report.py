import json

import pytest

from experiments.rocksample_jax_2026_10.report_checkpoint_beliefs import (
    load_trajectories,
)


@pytest.fixture
def trajectories(tmp_path):
    for repetition in (1, 2):
        for seed in range(4):
            directory = tmp_path / f"rep{repetition}_seed{seed}"
            directory.mkdir()
            for update in (0, 1, 2, 3, 4, 5, 6, 9, 14, 20, 29):
                counts = {"n_fit_episodes": 256, "n_test_episodes": 256}
                report = {
                    "env_steps": update * 1_048_576,
                    "update": update,
                    "seed": seed,
                    "run_id": f"run{repetition}",
                    "smoke": False,
                    "initialization_parameter_sha256": f"seed{seed}",
                    "on_policy": counts,
                    "common_checks": {**counts, "history_sha256": "fixed"},
                }
                (directory / f"update{update:03}.json").write_text(json.dumps(report))
    return tmp_path


def test_trajectory_comparison_rejects_changed_history(trajectories):
    assert len(load_trajectories(trajectories)) == 8
    path = trajectories / "rep2_seed3" / "update014.json"
    report = json.loads(path.read_text())
    report["common_checks"]["history_sha256"] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="common histories differ"):
        load_trajectories(trajectories)


def test_trajectory_comparison_rejects_different_initialization(trajectories):
    path = trajectories / "rep2_seed3" / "update000.json"
    report = json.loads(path.read_text())
    report["initialization_parameter_sha256"] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="initialization differs"):
        load_trajectories(trajectories)
