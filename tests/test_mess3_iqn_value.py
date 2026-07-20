"""Focused tests for the IQN distributional-value condition."""

from __future__ import annotations

import torch

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
