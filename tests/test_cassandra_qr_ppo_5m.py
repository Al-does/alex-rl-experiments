"""Configuration tests for the matched Cassandra QR-PPO conditions."""

from __future__ import annotations

import pytest

from experiments.cassandra_belief_factoring_2026_08.qr_ppo_5m.global_alias_qr.experiment import (
    build_config as build_global_alias_config,
)
from experiments.cassandra_belief_factoring_2026_08.qr_ppo_5m.shared import (
    CassandraQRTransformer,
    ENTROPY_COEFF,
    GAMMA,
    MODEL_CONFIG,
    NUM_QUANTILES,
    QUANTILE_HUBER_KAPPA,
    QUANTILE_LOSS_COEFFICIENT,
    TOTAL_ENV_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.qr_ppo_5m.targeted_qr.experiment import (
    build_config as build_targeted_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners import QRPPOTorchLearner


def context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


@pytest.mark.parametrize(
    ("build_config", "action_scope"),
    [
        (build_global_alias_config, "global_aliases"),
        (build_targeted_config, "targeted"),
    ],
)
def test_qr_recipe_builds_matched_five_million_step_condition(
    tmp_path,
    build_config,
    action_scope,
):
    config = build_config(context(tmp_path))

    assert TOTAL_ENV_STEPS == 5_000_000
    assert config.seed == 42
    assert config.env_config["action_scope"] == action_scope
    assert config.env_config["initial_state_distribution"] == "all_good"
    assert config.gamma == GAMMA == 0.990
    assert config.entropy_coeff == ENTROPY_COEFF == 0.008
    assert config.use_kl_loss is False
    assert config.vf_loss_coeff == 0.0
    assert config.learner_class is QRPPOTorchLearner
    assert config.learner_config_dict == {
        "qr_value/loss_coefficient": QUANTILE_LOSS_COEFFICIENT,
        "qr_value/huber_kappa": QUANTILE_HUBER_KAPPA,
    }
    assert config.rl_module_spec.module_class is CassandraQRTransformer
    assert config.rl_module_spec.model_config == MODEL_CONFIG
    assert config.rl_module_spec.model_config["d_model"] == 64
    assert config.rl_module_spec.model_config["qr_value"] == {
        "num_quantiles": NUM_QUANTILES,
    }


def test_qr_recipe_builds_fresh_configs(tmp_path):
    run_context = context(tmp_path)

    first = build_targeted_config(run_context)
    second = build_targeted_config(run_context)

    assert first is not second
    assert first.rl_module_spec is not second.rl_module_spec
