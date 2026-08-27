"""Recipe checks for the small IDAAC CartPole validation."""

from __future__ import annotations

from pathlib import Path

from harness.context import RunContext
from harness.hardware import PROFILES
from learners import IDAACConfig
from learners.models import IDAACModel

from experiments.idaac_cartpole_2026_08.cartpole.experiment import (
    MINIBATCH_SIZE,
    SMOKE_BATCH_SIZE,
    build_config,
)


def _context(tmp_path: Path, *, smoke: bool) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        run_id="idaac-cartpole-test",
        seed=42,
        smoke=smoke,
        hardware=PROFILES["cpu"],
    )


def test_idaac_cartpole_builds_fresh_paper_algorithm_configs(tmp_path):
    context = _context(tmp_path, smoke=False)

    first = build_config(context)
    second = build_config(context)

    assert isinstance(first, IDAACConfig)
    assert first is not second
    assert first.rl_module_spec.module_class is IDAACModel
    assert first.num_epochs == 1
    assert first.value_num_epochs == 9
    assert first.value_update_frequency == 1
    assert first.advantage_loss_coeff == 0.25
    assert first.invariance_loss_coeff == 0.001
    assert first.num_gpus_per_learner == 0


def test_idaac_cartpole_smoke_reduces_batch_and_value_epochs(tmp_path):
    config = build_config(_context(tmp_path, smoke=True))

    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.minibatch_size == MINIBATCH_SIZE
    assert config.value_num_epochs == 3
    assert config.num_env_runners == 0
