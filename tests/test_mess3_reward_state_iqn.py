"""Focused tests for IQN reward-state control conditions."""

from __future__ import annotations

import importlib

import pytest
import torch

from experiments.mess3_reward_state_cycle_1.iqn import (
    IQNPPOTorchLearner,
    IQNTransformerModel,
    IQNValueHead,
    quantile_huber_loss,
)
from experiments.mess3_reward_state_cycle_1.iqn_control import IQN_CONFIG
from harness.context import RunContext
from harness.hardware import PROFILES


FAMILY = "experiments.mess3_reward_state_cycle_1"


def test_iqn_head_stays_on_device_and_is_differentiable():
    head = IQNValueHead(embedding_dim=8, n_cosines=16)
    embeddings = torch.randn(2, 3, 8, requires_grad=True)
    taus = torch.rand(2, 3, 5)

    quantiles = head(embeddings, taus)

    assert quantiles.shape == (2, 3, 5)
    assert quantiles.device == embeddings.device
    quantiles.mean().backward()
    assert embeddings.grad is not None


def test_quantile_huber_loss_masks_padded_sequence_items():
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


@pytest.mark.parametrize(
    ("condition", "expected_task_kwargs"),
    [
        ("iqn_occupancy_only", {"action_limit": 5.0}),
        (
            "iqn_transition_kl",
            {"action_limit": 5.0, "transition_kl_beta": 4.0},
        ),
        (
            "iqn_action_norm",
            {
                "action_limit": 5.0,
                "action_norm_coefficient": 0.05,
            },
        ),
    ],
)
def test_iqn_conditions_change_only_reward_cost(
    tmp_path,
    condition,
    expected_task_kwargs,
):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    module = importlib.import_module(f"{FAMILY}.{condition}.experiment")

    first = module.build_config(context)
    second = module.build_config(context)

    assert first is not second
    assert first.env_config["task"]["kwargs"] == expected_task_kwargs
    assert first.train_batch_size_per_learner == 2_048
    assert first.minibatch_size == 256
    assert first.vf_loss_coeff == 0.0
    assert first.learner_class is IQNPPOTorchLearner
    assert first.rl_module_spec.module_class is IQNTransformerModel
    assert first.rl_module_spec.model_config["iqn_value"] == IQN_CONFIG
