"""Focused tests for gamma-zero Kelly credit assignment and IQN arms."""

from __future__ import annotations

import torch
from ray.rllib.core.columns import Columns

from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from experiments.mess_3_kelly_cycle_2.analysis import _STREAM_KEYS
from experiments.mess_3_kelly_cycle_2.learning import (
    BEHAVIOR_WAGER_KEY,
    CONDITIONAL_LAYOUT,
    CONDITIONAL_MODE,
    CORRECTNESS_KEY,
    COUPLED_MODE,
    DECOUPLED_MODE,
    SCALAR_LAYOUT,
    ConditionalWagerTransformerModel,
    IQNConditionalWagerTransformerModel,
    IQNScalarWagerTransformerModel,
    KellyIQNPPOTorchLearner,
    KellyMeanPPOTorchLearner,
    PrepareKellyBatch,
    TokenCategoricalWithConditionalWager,
    current_selected_wager,
    selected_wager,
)
from experiments.mess_3_kelly_cycle_2.shared import (
    ARMS,
    IQN_CONFIG,
    build_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models.transformer import TransformerModel


def test_scalar_and_conditional_wagers_select_expected_outputs():
    actions = torch.tensor([[0, 2]])
    scalar_inputs = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]]
    )
    scalar = selected_wager(
        action_dist_inputs=scalar_inputs,
        actions=actions,
        layout=SCALAR_LAYOUT,
    )
    torch.testing.assert_close(
        scalar,
        torch.sigmoid(torch.tensor([[0.0, 2.0]])),
    )

    conditional_inputs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -2.0, 0.0, 2.0],
            ]
        ]
    )
    conditional = selected_wager(
        action_dist_inputs=conditional_inputs,
        actions=actions,
        layout=CONDITIONAL_LAYOUT,
    )
    torch.testing.assert_close(
        conditional,
        torch.sigmoid(torch.tensor([[-1.0, 2.0]])),
    )


def test_conditional_wager_gradient_only_reaches_selected_action():
    inputs = torch.zeros((1, 2, 6), requires_grad=True)
    actions = torch.tensor([[0, 2]])
    wager = current_selected_wager(
        action_dist_inputs=inputs,
        actions=actions,
        layout=CONDITIONAL_LAYOUT,
    )
    wager.sum().backward()
    wager_gradients = inputs.grad[..., 3:]
    assert wager_gradients[0, 0, 0] != 0.0
    assert wager_gradients[0, 1, 2] != 0.0
    assert torch.count_nonzero(wager_gradients) == 2
    assert torch.count_nonzero(inputs.grad[..., :3]) == 0


def test_decoupled_connector_preserves_correctness_and_coupled_replaces_it():
    inputs = torch.zeros((1, 2, 4))
    actions = torch.tensor([[0, 1]])
    rewards = torch.tensor([[1.0, 0.0]])

    decoupled_batch = {
        "default": {
            Columns.REWARDS: rewards.clone(),
            Columns.ACTION_DIST_INPUTS: inputs,
            Columns.ACTIONS: actions,
        }
    }
    PrepareKellyBatch(
        actor_mode=DECOUPLED_MODE,
        wager_layout=SCALAR_LAYOUT,
    )(
        rl_module=None,
        episodes=None,
        batch=decoupled_batch,
    )
    torch.testing.assert_close(
        decoupled_batch["default"][Columns.REWARDS],
        rewards,
    )
    torch.testing.assert_close(
        decoupled_batch["default"][CORRECTNESS_KEY],
        rewards,
    )
    assert BEHAVIOR_WAGER_KEY in decoupled_batch["default"]

    coupled_batch = {
        "default": {
            Columns.REWARDS: rewards.clone(),
            Columns.ACTION_DIST_INPUTS: inputs,
            Columns.ACTIONS: actions,
        }
    }
    PrepareKellyBatch(
        actor_mode=COUPLED_MODE,
        wager_layout=SCALAR_LAYOUT,
    )(
        rl_module=None,
        episodes=None,
        batch=coupled_batch,
    )
    expected = torch.tensor([[torch.log(torch.tensor(2.0)), torch.log(torch.tensor(0.5))]])
    torch.testing.assert_close(
        coupled_batch["default"][Columns.REWARDS],
        expected,
    )


def test_conditional_distribution_ignores_wager_logits_for_token_choice():
    logits = torch.tensor([[0.0, 4.0, -2.0, 100.0, -100.0, 50.0]])
    distribution = TokenCategoricalWithConditionalWager.from_logits(logits)
    assert distribution.to_deterministic().sample().tolist() == [1]


def test_all_eight_arms_build_gamma_zero_mean_and_iqn_configs(tmp_path):
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
    assert len(configs) == 8
    for arm in ARMS:
        config = configs[arm.name]
        assert config.gamma == 0.0
        assert config.lambda_ == 0.0
        assert config.entropy_coeff == 0.0
        assert config.train_batch_size_per_learner == 2_048
        assert "next_token_aux" not in config.rl_module_spec.model_config
        if arm.critic_mode == "iqn":
            assert config.vf_loss_coeff == 0.0
            assert config.rl_module_spec.model_config["iqn_value"] == IQN_CONFIG
            assert config.learner_class in {
                IQNPPOTorchLearner,
                KellyIQNPPOTorchLearner,
            }
        else:
            assert config.vf_loss_coeff == 0.5
        if arm.name == "correctness_mean":
            assert config.rl_module_spec.module_class is TransformerModel
        elif arm.name == "correctness_iqn":
            assert config.rl_module_spec.module_class is IQNTransformerModel
        elif arm.name == "decoupled_kelly_mean":
            assert config.learner_class is KellyMeanPPOTorchLearner
        elif arm.name == "decoupled_kelly_iqn":
            assert config.rl_module_spec.module_class is IQNScalarWagerTransformerModel
        elif arm.name == "conditional_decoupled_kelly_mean":
            assert (
                config.rl_module_spec.module_class
                is ConditionalWagerTransformerModel
            )
        elif arm.name == "conditional_decoupled_kelly_iqn":
            assert (
                config.rl_module_spec.module_class
                is IQNConditionalWagerTransformerModel
            )


def test_cycle_two_reuses_cycle_one_probe_streams():
    assert _STREAM_KEYS == {
        "probe_train": (200,),
        "probe_test": (201,),
        "plot_sample": (202,),
    }
