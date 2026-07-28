from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mess3_token_guess_cycle_2.mse_charts import (
    CONDITIONS,
    SEEDS,
    _validated_points,
    load_mse_curves,
    write_mse_bar_charts,
)


def _payload(offset: float = 0.0) -> dict:
    return {
        "checkpoints": [
            {
                "agent_steps": step,
                "mse": mse + offset,
                "training_iteration": None if step == 0 else index * 10,
                "probe": {
                    "mse": mse + offset,
                    "mse_ci_95_low": mse + offset - 0.00001,
                    "mse_ci_95_high": mse + offset + 0.00001,
                    "bootstrap_n": 1_000,
                    "bootstrap_cluster": "environment_episode",
                    "sampling_distribution": "process_weighted_rollout",
                    "representation": "post_final_layer_norm",
                    "n_fit": 60_000,
                    "n_test": 80_000,
                },
            }
            for index, (step, mse) in enumerate(
                ((0, 0.004), (330_000, 0.001), (660_000, 0.0005))
            )
        ]
    }


def test_validated_points_requires_episode_cluster_bootstrap():
    payload = _payload()
    payload["checkpoints"][1]["probe"]["bootstrap_cluster"] = "timestep"

    with pytest.raises(ValueError, match="episode-clustered"):
        _validated_points(payload)


def test_load_and_write_all_mse_charts(tmp_path: Path):
    results_root = tmp_path / "mess3_token_guess_cycle_2"
    for condition_index, condition in enumerate(CONDITIONS):
        for seed_index, seed in enumerate(SEEDS):
            run_name = f"mess3_token_guess_cycle_2-{condition}-seed{seed}"
            run_dir = results_root / condition / "results" / run_name
            run_dir.mkdir(parents=True)
            payload = _payload(offset=condition_index * 1e-5 + seed_index * 1e-6)
            (run_dir / "checkpoint_probe_curve.json").write_text(
                json.dumps(payload)
            )

    curves = load_mse_curves(results_root)
    output_dir = tmp_path / "charts"
    summary = write_mse_bar_charts(curves, output_dir=output_dir)

    assert len(summary["run_charts"]) == 15
    assert len(list((output_dir / "by_run").glob("*.png"))) == 15
    assert (output_dir / "mse_over_training_all_runs.png").is_file()
    assert (output_dir / "mse_over_training_condition_means.png").is_file()
    assessment = (output_dir / "bootstrap_assessment.md").read_text()
    assert "1,000 percentile-bootstrap resamples" in assessment
    assert "only three model seeds" in assessment
    assert summary["per_checkpoint_uncertainty"]["n_resamples"] == 1_000
    assert summary["model_seed_summary"]["bootstrap_used"] is False
