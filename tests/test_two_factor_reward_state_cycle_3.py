"""Scientific and architecture tests for two-factor cycle 3."""

from __future__ import annotations

import importlib

import pytest

from experiments.two_factor_reward_state_PPO_cycle_2.task import CONDITIONS
from experiments.two_factor_reward_state_REINFORCE_cycle_3.model import (
    TwoFactorRewardReinforce,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_3.shared import (
    Reinforce,
    ReinforceConfig,
    ReinforceTorchLearner,
    SMOKE_BATCH_SIZE,
    TOTAL_ENV_STEPS,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_each_cycle_3_leaf_builds_a_fresh_reinforce_recipe(tmp_path, condition):
    module = importlib.import_module(
        "experiments.two_factor_reward_state_REINFORCE_cycle_3."
        f"{condition}.experiment"
    )
    first = module.build_config(_context(tmp_path))
    second = module.build_config(_context(tmp_path))

    assert first is not second
    assert isinstance(first, ReinforceConfig)
    assert first.algo_class is Reinforce
    assert first.get_default_learner_class() is ReinforceTorchLearner
    assert first.rl_module_spec.module_class is TwoFactorRewardReinforce
    assert first.seed == 42
    assert first.num_env_runners == 0
    assert first.env_config["task"]["kwargs"]["condition"] == condition
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.lambda_ == 1.0
    assert first.num_epochs == 1
    assert first.use_kl_loss is False
    assert TOTAL_ENV_STEPS == 5_000_000
