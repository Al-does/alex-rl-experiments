"""Focused composition tests for the reward-state Kelly/IQN battery."""

from __future__ import annotations

import importlib

import pytest
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from experiments.mess3_reward_state_cycle_1.iqn import (
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from experiments.mess3_reward_state_kelly_iqn_2026_07.kelly import (
    CORRECTNESS_COEFFICIENT_KEY,
    DIRECT_LOSS_COEFFICIENT_KEY,
    NAMESPACE as KELLY_NAMESPACE,
    predictive_kelly_metrics,
)
from experiments.mess3_reward_state_kelly_iqn_2026_07.remote import (
    CONDITION_MODULES,
    assigned_index,
)
from experiments.mess3_reward_state_kelly_iqn_2026_07.shared import (
    IQN_CONFIG,
    KellyIQNPPOTorchLearner,
    KellyIQNTransformerModel,
    KellyPPOTorchLearner,
    KellyTransformerModel,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models.transformer import TransformerModel


FAMILY = "experiments.mess3_reward_state_kelly_iqn_2026_07"
CONDITIONS = (
    ("ppo_gamma_0", 0.0, False, False),
    ("ppo_gamma_099", 0.99, False, False),
    ("iqn_gamma_0", 0.0, True, False),
    ("iqn_gamma_099", 0.99, True, False),
    ("kelly_gamma_0", 0.0, False, True),
    ("kelly_gamma_099", 0.99, False, True),
    ("kelly_iqn_gamma_0", 0.0, True, True),
    ("kelly_iqn_gamma_099", 0.99, True, True),
)


@pytest.mark.parametrize(
    ("condition", "gamma", "use_iqn", "use_kelly"),
    CONDITIONS,
)
def test_eight_conditions_build_controlled_configs(
    tmp_path,
    condition,
    gamma,
    use_iqn,
    use_kelly,
):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    module = importlib.import_module(f"{FAMILY}.{condition}.experiment")

    first = module.build_config(context)
    second = module.build_config(context)

    assert first is not second
    assert first.gamma == gamma
    assert first.lambda_ == 0.95
    assert first.env_config["task"]["kwargs"] == {"action_limit": 5.0}
    assert first.train_batch_size_per_learner == 2_048
    assert first.minibatch_size == 256
    assert first.vf_loss_coeff == (0.0 if use_iqn else 0.5)
    expected_model = {
        (False, False): TransformerModel,
        (True, False): IQNTransformerModel,
        (False, True): KellyTransformerModel,
        (True, True): KellyIQNTransformerModel,
    }[(use_iqn, use_kelly)]
    expected_learner = {
        (False, False): PPOTorchLearner,
        (True, False): IQNPPOTorchLearner,
        (False, True): KellyPPOTorchLearner,
        (True, True): KellyIQNPPOTorchLearner,
    }[(use_iqn, use_kelly)]
    assert first.rl_module_spec.module_class is expected_model
    assert first.learner_class is expected_learner
    if use_iqn:
        assert first.rl_module_spec.model_config["iqn_value"] == IQN_CONFIG
    if use_kelly:
        assert first.rl_module_spec.model_config[KELLY_NAMESPACE] == {
            "num_tokens": 3
        }
        assert first.learner_config_dict[CORRECTNESS_COEFFICIENT_KEY] == 1.0
        assert first.learner_config_dict[DIRECT_LOSS_COEFFICIENT_KEY] == 1.0


def test_predictive_kelly_loss_masks_padding_and_backpropagates():
    token_logits = torch.tensor(
        [[[0.0, 0.0, 2.0], [100.0, -100.0, -100.0]]],
        requires_grad=True,
    )
    wager_logits = torch.zeros_like(token_logits, requires_grad=True)
    targets = torch.tensor([[2, 1]])
    valid = torch.tensor([[True, False]])

    metrics = predictive_kelly_metrics(
        token_logits,
        wager_logits,
        targets,
        valid,
    )
    (metrics["cross_entropy"] + metrics["direct_loss"]).backward()

    assert metrics["accuracy"] == 1.0
    torch.testing.assert_close(
        metrics["log_growth_mean"],
        torch.log(torch.tensor(2.0)),
    )
    assert token_logits.grad is not None
    assert wager_logits.grad is not None
    assert wager_logits.grad[0, 0, 2] != 0.0
    torch.testing.assert_close(
        wager_logits.grad[0, 1],
        torch.zeros(3),
    )


def test_vast_shots_map_one_to_one_to_eight_conditions():
    assert len(CONDITION_MODULES) == 8
    for shot, module in enumerate(CONDITION_MODULES, start=1):
        assert assigned_index(f"rllib-reward-state-{shot}-a1b2c3") == shot - 1
        assert module.endswith(f"{CONDITIONS[shot - 1][0]}.experiment")
