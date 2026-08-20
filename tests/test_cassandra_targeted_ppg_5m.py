"""Configuration checks for the isolated four-arm Cassandra PPG campaign."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from harness.context import RunContext
from harness.hardware import PROFILES
from learners import PPG, PPGConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppg_5m.shared import (
    DISCOUNTED_VALUE_RANGE,
    MODEL_CONFIG,
    REWARD_RANGE,
    TOTAL_ENV_STEPS,
    VF_CLIP_PARAM,
    CassandraPPGTransformer,
)


INTERVENTIONS = {
    "npi16_value_0p003": (16, 0.003),
    "npi16_value_0p01": (16, 0.01),
    "npi32_value_0p003": (32, 0.003),
    "npi32_value_0p01": (32, 0.01),
}
MODULE_PREFIX = (
    "experiments.cassandra_belief_factoring_2026_08."
    "targeted_ppg_5m"
)


def _context(tmp_path: Path, *, smoke: bool) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        run_id="test",
        smoke=smoke,
        hardware=PROFILES["cpu"],
    )


@pytest.mark.parametrize(
    ("leaf", "expected"),
    INTERVENTIONS.items(),
)
def test_targeted_ppg_interventions_are_fresh_matched_configs(
    tmp_path,
    leaf,
    expected,
):
    module = importlib.import_module(f"{MODULE_PREFIX}.{leaf}.experiment")
    first = module.build_config(_context(tmp_path, smoke=False))
    second = module.build_config(_context(tmp_path, smoke=False))
    expected_npi, expected_coefficient = expected

    assert isinstance(first, PPGConfig)
    assert first is not second
    assert first.algo_class is PPG
    assert first.seed == 42
    assert first.env_config["action_scope"] == "targeted"
    assert first.env_config["initial_state_distribution"] == "all_good"
    assert first.policy_iterations_per_aux == expected_npi
    assert first.aux_epochs == 6
    assert first.aux_minibatch_size == 8_192
    assert first.beta_clone == 1.0
    assert first.aux_value_loss_coeff == expected_coefficient
    assert first.aux_true_value_loss_coeff == expected_coefficient
    assert first.rl_module_spec.module_class is CassandraPPGTransformer
    assert first.rl_module_spec.model_config == MODEL_CONFIG
    assert first.entropy_coeff == [[0, 0.03], [2_500_000, 0.008]]


@pytest.mark.parametrize("leaf", INTERVENTIONS)
def test_smoke_config_reaches_an_auxiliary_phase(tmp_path, leaf):
    module = importlib.import_module(f"{MODULE_PREFIX}.{leaf}.experiment")
    config = module.build_config(_context(tmp_path, smoke=True))

    assert config.train_batch_size_per_learner == 2_048
    assert config.minibatch_size == 256
    assert config.policy_iterations_per_aux == 2
    assert config.aux_epochs == 1
    assert config.aux_minibatch_size == 256
    assert config.num_env_runners == 0
    assert config.num_envs_per_env_runner == 1


def test_reward_scaled_value_hyperparameters_cover_targeted_bounds():
    assert TOTAL_ENV_STEPS == 5_000_000
    assert REWARD_RANGE == pytest.approx((-3.75, 0.9985**4))
    assert DISCOUNTED_VALUE_RANGE == pytest.approx(
        (
            -3.75 / (1.0 - 0.99),
            (0.9985**4) / (1.0 - 0.99),
        )
    )
    assert VF_CLIP_PARAM == pytest.approx(375.0**2)
