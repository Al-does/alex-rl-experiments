"""Focused tests for the IQN distributional-value condition."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch
from ray.rllib.core.columns import Columns

from experiments.mess3_token_guess_cycle_1.average_reward import (
    AverageRewardIQNPPOTorchLearner,
    CenterRewardsByRunningAverage,
    center_rewards,
)
from experiments.mess3_token_guess_cycle_1.iqn_gamma_1_3m.experiment import (
    build_config as build_gamma_one_config,
)
from experiments.mess3_token_guess_cycle_1.iqn_value.experiment import (
    IQN_CONFIG,
    build_config,
)
from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    IQNPPOTorchLearner,
    IQNTransformerModel,
    IQNValueHead,
    quantile_huber_loss,
)
from experiments.mess3_token_guess_cycle_1.iqn_value_20m.experiment import (
    build_config as build_long_config,
    checkpoint_records,
    training_curve,
)
from experiments.mess3_token_guess_cycle_1.iqn_return_objectives.experiment import (
    build_config as build_return_objective_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def test_iqn_head_is_device_native_and_differentiable():
    head = IQNValueHead(embedding_dim=8, n_cosines=16)
    embeddings = torch.randn(2, 3, 8, requires_grad=True)
    taus = torch.rand(2, 3, 5)

    quantiles = head(embeddings, taus)
    assert quantiles.shape == (2, 3, 5)
    quantiles.mean().backward()
    assert embeddings.grad is not None
    assert embeddings.grad.shape == embeddings.shape


def test_quantile_huber_loss_matches_simple_case_and_masks_padding():
    quantiles = torch.zeros(1, 2, 2)
    taus = torch.tensor([[[0.25, 0.75], [0.25, 0.75]]])
    targets = torch.tensor([[1.0, 100.0]])
    valid = torch.tensor([[True, False]])

    loss = quantile_huber_loss(
        quantiles,
        taus,
        targets,
        kappa=1.0,
        valid=valid,
    )
    torch.testing.assert_close(loss, torch.tensor(0.25))


def test_iqn_recipe_builds_fresh_controlled_smoke_configs(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    first = build_config(context)
    second = build_config(context)

    assert first is not second
    assert first.seed == 42
    assert first.num_env_runners == 0
    assert first.train_batch_size_per_learner == 2_048
    assert first.lambda_ == 0.95
    assert first.vf_loss_coeff == 0.0
    assert first.learner_class is IQNPPOTorchLearner
    assert first.rl_module_spec.module_class is IQNTransformerModel
    assert first.rl_module_spec.model_config["iqn_value"] == IQN_CONFIG


def test_long_iqn_recipe_preserves_the_controlled_iqn_config(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    config = build_long_config(context)

    assert config.learner_class is IQNPPOTorchLearner
    assert config.rl_module_spec.module_class is IQNTransformerModel
    assert config.train_batch_size_per_learner == 2_048
    assert config.lambda_ == 0.95


def test_gamma_one_recipe_changes_only_discount_factor(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    standard = build_config(context)
    gamma_one = build_gamma_one_config(context)

    assert standard.gamma == 0.99
    assert gamma_one.gamma == 1.0
    assert gamma_one.lambda_ == standard.lambda_
    assert gamma_one.learner_class is standard.learner_class
    assert gamma_one.rl_module_spec.module_class is standard.rl_module_spec.module_class
    assert (
        gamma_one.train_batch_size_per_learner
        == standard.train_batch_size_per_learner
    )


def test_average_reward_centering_preserves_padding_and_updates_ema():
    rewards = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, False]])
    centered = center_rewards(
        rewards,
        torch.tensor(0.5),
        valid=valid,
    )
    torch.testing.assert_close(
        centered,
        torch.tensor([[0.5, -0.5, 0.0]]),
    )

    connector = CenterRewardsByRunningAverage(ema_decay=0.5)
    first = {"default_policy": {
        Columns.REWARDS: rewards.clone(),
        Columns.LOSS_MASK: valid,
    }}
    connector(rl_module=None, episodes=[], batch=first)
    torch.testing.assert_close(
        connector.estimates["default_policy"],
        torch.tensor(0.5),
    )
    second = {"default_policy": {
        Columns.REWARDS: torch.tensor([[1.0, 1.0, 0.0]]),
        Columns.LOSS_MASK: valid,
    }}
    connector(rl_module=None, episodes=[], batch=second)
    torch.testing.assert_close(
        connector.estimates["default_policy"],
        torch.tensor(0.75),
    )


def test_return_objective_configs_change_only_their_intended_value_logic(
    tmp_path,
):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    standard = build_config(context)
    gamma_zero = build_return_objective_config(context, "gamma_zero")
    average = build_return_objective_config(context, "average_reward")

    assert gamma_zero.gamma == 0.0
    assert gamma_zero.learner_class is standard.learner_class
    assert average.gamma == 1.0
    assert average.learner_class is AverageRewardIQNPPOTorchLearner
    for config in (gamma_zero, average):
        assert config.lambda_ == standard.lambda_
        assert config.rl_module_spec.module_class is standard.rl_module_spec.module_class
        assert (
            config.train_batch_size_per_learner
            == standard.train_batch_size_per_learner
        )


def test_longitudinal_records_order_checkpoints_and_compute_reward_percentage():
    result = SimpleNamespace(
        metrics_dataframe=pd.DataFrame(
            [
                {
                    "training_iteration": 2,
                    "env_runners/num_env_steps_sampled_lifetime": 200,
                    "env_runners/episode_return_mean": 40.0,
                    "env_runners/episode_len_mean": 100.0,
                },
                {
                    "training_iteration": 1,
                    "env_runners/num_env_steps_sampled_lifetime": 100,
                    "env_runners/episode_return_mean": 25.0,
                    "env_runners/episode_len_mean": 100.0,
                },
            ]
        ),
        best_checkpoints=[
            (
                SimpleNamespace(path="/tmp/checkpoint_000002"),
                {
                    "training_iteration": 2,
                    "env_runners": {
                        "num_env_steps_sampled_lifetime": 200,
                        "episode_return_mean": 40.0,
                        "episode_len_mean": 100.0,
                    },
                },
            ),
            (
                SimpleNamespace(path="/tmp/checkpoint_000001"),
                {
                    "training_iteration": 1,
                    "env_runners": {
                        "num_env_steps_sampled_lifetime": 100,
                        "episode_return_mean": 25.0,
                        "episode_len_mean": 100.0,
                    },
                },
            ),
        ],
    )

    rewards = training_curve(result)
    checkpoints = checkpoint_records(result)

    assert [record["reward_percentage"] for record in rewards] == [40.0, 25.0]
    assert [record["agent_steps"] for record in checkpoints] == [100, 200]
    assert [
        record["training_reward_percentage"] for record in checkpoints
    ] == [25.0, 40.0]
