"""Configuration checks for the matched Cassandra PPG action-scope runs."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from harness.context import RunContext
from harness.hardware import PROFILES
from learners import PPG, PPGConfig

from experiments.cassandra_belief_factoring_2026_08.ppg_5m.shared import (
    AUX_VALUE_LOSS_COEFF,
    MODEL_CONFIG,
    POLICY_ITERATIONS_PER_AUX,
    REWARD_RANGES,
    TOTAL_ENV_STEPS,
    VF_CLIP_PARAM,
    CassandraPPGTransformer,
)


CONDITIONS = {
    "global_alias_ppg": "global_aliases",
    "targeted_ppg": "targeted",
}
MODULE_PREFIX = "experiments.cassandra_belief_factoring_2026_08.ppg_5m"


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


@pytest.mark.parametrize(("leaf", "action_scope"), CONDITIONS.items())
def test_ppg_conditions_build_fresh_configs(tmp_path, leaf, action_scope):
    module = importlib.import_module(f"{MODULE_PREFIX}.{leaf}.experiment")
    first = module.build_config(_context(tmp_path, smoke=False))
    second = module.build_config(_context(tmp_path, smoke=False))

    assert isinstance(first, PPGConfig)
    assert first is not second
    assert first.algo_class is PPG
    assert first.seed == 42
    assert first.env_config["action_scope"] == action_scope
    assert first.env_config["initial_state_distribution"] == "all_good"
    assert first.policy_iterations_per_aux == POLICY_ITERATIONS_PER_AUX == 32
    assert first.aux_epochs == 6
    assert first.aux_minibatch_size == 8_192
    assert first.beta_clone == 1.0
    assert first.aux_value_loss_coeff == AUX_VALUE_LOSS_COEFF == 0.003
    assert first.aux_true_value_loss_coeff == AUX_VALUE_LOSS_COEFF
    assert first.rl_module_spec.module_class is CassandraPPGTransformer
    assert first.rl_module_spec.model_config == MODEL_CONFIG
    assert first.entropy_coeff == [[0, 0.03], [2_500_000, 0.008]]


def test_action_scope_is_the_only_config_difference(tmp_path):
    modules = {
        leaf: importlib.import_module(f"{MODULE_PREFIX}.{leaf}.experiment")
        for leaf in CONDITIONS
    }
    configs = {
        leaf: module.build_config(_context(tmp_path, smoke=False))
        for leaf, module in modules.items()
    }
    global_alias = configs["global_alias_ppg"]
    targeted = configs["targeted_ppg"]

    assert global_alias.env_config != targeted.env_config
    assert {
        key: value
        for key, value in global_alias.env_config.items()
        if key != "action_scope"
    } == {
        key: value
        for key, value in targeted.env_config.items()
        if key != "action_scope"
    }
    for setting in (
        "lr",
        "gamma",
        "lambda_",
        "clip_param",
        "vf_loss_coeff",
        "vf_clip_param",
        "entropy_coeff",
        "train_batch_size_per_learner",
        "minibatch_size",
        "num_epochs",
        "policy_iterations_per_aux",
        "aux_epochs",
        "aux_minibatch_size",
        "aux_lr",
        "beta_clone",
        "aux_value_loss_coeff",
        "aux_true_value_loss_coeff",
    ):
        assert getattr(global_alias, setting) == getattr(targeted, setting)
    assert global_alias.rl_module_spec.model_config == (
        targeted.rl_module_spec.model_config
    )


@pytest.mark.parametrize("leaf", CONDITIONS)
def test_smoke_configs_would_reach_an_auxiliary_phase(tmp_path, leaf):
    module = importlib.import_module(f"{MODULE_PREFIX}.{leaf}.experiment")
    config = module.build_config(_context(tmp_path, smoke=True))

    assert config.train_batch_size_per_learner == 2_048
    assert config.minibatch_size == 256
    assert config.policy_iterations_per_aux == 2
    assert config.aux_epochs == 1
    assert config.aux_minibatch_size == 256
    assert config.num_env_runners == 0
    assert config.num_envs_per_env_runner == 1


def test_reward_bounds_cover_both_action_scopes():
    assert TOTAL_ENV_STEPS == 5_000_000
    assert REWARD_RANGES["global_aliases"] == pytest.approx(
        (-15.0, 0.9985**4)
    )
    assert REWARD_RANGES["targeted"] == pytest.approx((-3.75, 0.9985**4))
    assert VF_CLIP_PARAM == pytest.approx(1_500.0**2)
