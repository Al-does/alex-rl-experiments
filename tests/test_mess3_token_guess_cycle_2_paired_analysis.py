from __future__ import annotations

from experiments.mess3_token_guess_cycle_2.paired_analysis import paired_ttest
from experiments.mess3_token_guess_cycle_2.shared import (
    CHECKPOINT_FREQUENCY,
    THIRD_CHECKPOINT_ENV_STEPS,
    _run_schedule,
    condition_by_name,
)
from harness.context import RunContext
from pathlib import Path


def test_third_checkpoint_budget_preserves_cadence():
    context = RunContext(
        experiment_dir=Path("."),
        results_dir=Path("/tmp/tg-c2-r"),
        artifacts_dir=Path("/tmp/tg-c2-a"),
        seed=42,
        run_id="test",
        smoke=False,
    )
    steps, frequency = _run_schedule(
        context,
        condition_by_name("ppo"),
        THIRD_CHECKPOINT_ENV_STEPS,
        preserve_checkpoint_cadence=True,
    )
    assert steps == 700_000
    assert frequency == CHECKPOINT_FREQUENCY


def test_paired_ttest_kelly_vs_ppo_shape():
    payload = paired_ttest(
        candidate="decoupled_kelly",
        control="ppo",
        candidate_values=[0.0003, 0.00025, 0.00028],
        control_values=[0.0004, 0.00035, 0.00033],
        seeds=[42, 43, 44],
        metric="held_out_affine_probe_mse",
        checkpoint_index=2,
        agent_steps=662_054,
    )
    assert payload["paired_t_test"]["df"] == 2
    assert payload["mean_paired_difference"] < 0
    assert "shapiro_wilk" in payload
