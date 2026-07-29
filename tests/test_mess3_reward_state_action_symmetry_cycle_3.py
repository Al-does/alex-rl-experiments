"""Tests for action-symmetry cycle 3 auxiliary-loss arms."""

from __future__ import annotations

import importlib

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_3.learning import (
    KELLY_LOSS_COEFFICIENT_KEY,
    KellyPPOTorchLearner,
)
from experiments.mess3_reward_state_action_symmetry_cycle_3.shared import (
    DIRECT_KELLY_LOSS_WEIGHT,
    PREDICTIVE_LOSS_WEIGHT,
    TOTAL_ENV_STEPS,
    PredictiveLearner,
    PredictiveModel,
    condition_name,
    environment_config,
)
from experiments.mess3_reward_state_action_symmetry_cycle_3.shared import (
    KellyModel,
)
from harness.context import RunContext
from harness.hardware import PROFILES


ARMS = (
    ("variant_2_decoupled_kelly", 2, "decoupled_kelly"),
    ("variant_2_predictive_loss", 2, "predictive_loss"),
    ("variant_3_predictive_loss", 3, "predictive_loss"),
)


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_cycle_3_keeps_cycle_2_horizon_and_zero_delay():
    assert TOTAL_ENV_STEPS == 700_000
    assert environment_config(2)["delay"] == 0
    assert environment_config(3)["task"]["kwargs"]["variant"] == 3


def test_condition_names_match_leaf_folders():
    for folder, variant, objective in ARMS:
        assert condition_name(variant, objective) == folder


def test_arm_recipes_wire_expected_auxiliaries(tmp_path):
    context = _context(tmp_path)
    for folder, variant, objective in ARMS:
        module = importlib.import_module(
            "experiments.mess3_reward_state_action_symmetry_cycle_3."
            f"{folder}.experiment"
        )
        config = module.build_config(context)
        assert config is not module.build_config(context)
        assert config.env_config["task"]["kwargs"]["variant"] == variant
        assert config.train_batch_size_per_learner == 2_048
        spec = config.rl_module_spec
        learner_cfg = config.learner_config_dict
        if objective == "decoupled_kelly":
            assert spec.module_class is KellyModel
            assert config.learner_class is KellyPPOTorchLearner
            assert learner_cfg[KELLY_LOSS_COEFFICIENT_KEY] == DIRECT_KELLY_LOSS_WEIGHT
        else:
            assert spec.module_class is PredictiveModel
            assert config.learner_class is PredictiveLearner
            assert learner_cfg["next_token_aux/lambda"] == PREDICTIVE_LOSS_WEIGHT
            assert "next_token_aux/target_extractor" in learner_cfg
            assert spec.model_config["next_token_aux"]["num_classes"] == 3
        environment = HMMEnv(config.env_config)
        try:
            assert environment.action_space.n == 3
            assert environment.observation_space.shape == (6,)
        finally:
            environment.close()
