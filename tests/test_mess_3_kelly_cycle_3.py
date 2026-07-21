"""Focused tests for the four-arm gamma-.99 comparison."""

from __future__ import annotations

from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from experiments.mess_3_kelly_cycle_2.learning import (
    IQNConditionalWagerTransformerModel,
    KellyIQNPPOTorchLearner,
    KellyMeanPPOTorchLearner,
)
from experiments.mess_3_kelly_cycle_3.shared import (
    ARMS,
    IQN_CONFIG,
    build_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models.transformer import TransformerModel


def test_four_cycle_three_arms_build_controlled_gamma_099_configs(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    configs = {
        arm.name: build_config(context, arm.name)
        for arm in ARMS
    }
    assert set(configs) == {
        "ppo",
        "iqn",
        "conditional_decoupled_kelly_mean",
        "conditional_decoupled_kelly_iqn",
    }
    for arm in ARMS:
        config = configs[arm.name]
        assert config.gamma == 0.99
        assert config.lambda_ == 0.95
        assert config.entropy_coeff == 0.0
        assert config.train_batch_size_per_learner == 2_048
        assert "next_token_aux" not in config.rl_module_spec.model_config
        if arm.critic_mode == "iqn":
            assert config.vf_loss_coeff == 0.0
            assert config.rl_module_spec.model_config["iqn_value"] == IQN_CONFIG
        else:
            assert config.vf_loss_coeff == 0.5

    assert configs["ppo"].rl_module_spec.module_class is TransformerModel
    assert configs["ppo"].learner_class is None
    assert configs["iqn"].rl_module_spec.module_class is IQNTransformerModel
    assert configs["iqn"].learner_class is IQNPPOTorchLearner
    assert (
        configs["conditional_decoupled_kelly_mean"].learner_class
        is KellyMeanPPOTorchLearner
    )
    assert (
        configs["conditional_decoupled_kelly_iqn"].rl_module_spec.module_class
        is IQNConditionalWagerTransformerModel
    )
    assert (
        configs["conditional_decoupled_kelly_iqn"].learner_class
        is KellyIQNPPOTorchLearner
    )
