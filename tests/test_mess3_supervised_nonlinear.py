"""Wiring tests for nonlinear supervised MESS3 prediction heads."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.mess3_supervised.paper_supervised_replication.model import (
    PaperTransformer,
)
from experiments.mess3_supervised_nonlinear.model import (
    SwishMLPDecoderTransformer,
)
from experiments.mess3_supervised_nonlinear.shared import (
    FULL_TRAINING_CONFIG,
    SMOKE_TRAINING_CONFIG,
)


def test_linear_control_uses_the_paper_head():
    model = PaperTransformer()

    assert isinstance(model.unembedding, nn.Linear)
    assert (model.unembedding.in_features, model.unembedding.out_features) == (
        64,
        3,
    )


@pytest.mark.parametrize(
    ("depth", "expected_shapes"),
    [
        (2, [(64, 64), (64, 3)]),
        (4, [(64, 64), (64, 64), (64, 64), (64, 3)]),
    ],
)
def test_swish_decoder_depth_and_shape(depth, expected_shapes):
    model = SwishMLPDecoderTransformer(decoder_depth=depth)
    linear_layers = [
        layer for layer in model.unembedding if isinstance(layer, nn.Linear)
    ]
    swish_layers = [
        layer for layer in model.unembedding if isinstance(layer, nn.SiLU)
    ]

    assert [
        (layer.in_features, layer.out_features) for layer in linear_layers
    ] == expected_shapes
    assert len(swish_layers) == depth - 1
    assert model(torch.zeros((2, 10), dtype=torch.long)).shape == (2, 10, 3)


@pytest.mark.parametrize("depth", [2, 4])
def test_nonlinear_head_preserves_backbone_and_probe_location(depth):
    tokens = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]])
    torch.manual_seed(17)
    paper_model = PaperTransformer().eval()
    torch.manual_seed(17)
    nonlinear_model = SwishMLPDecoderTransformer(
        decoder_depth=depth,
    ).eval()

    _, paper_activations = paper_model(tokens, return_activations=True)
    _, nonlinear_activations = nonlinear_model(
        tokens,
        return_activations=True,
    )

    assert nonlinear_model.activation_names == paper_model.activation_names
    torch.testing.assert_close(
        nonlinear_activations["block_3"],
        paper_activations["block_3"],
    )
    torch.testing.assert_close(
        nonlinear_activations["pre_ln_final"],
        paper_activations["pre_ln_final"],
    )


def test_all_conditions_use_only_the_paper_sgd_recipe():
    assert FULL_TRAINING_CONFIG.optimizer_name == "sgd"
    assert FULL_TRAINING_CONFIG.total_steps == 1_000_000
    assert SMOKE_TRAINING_CONFIG.optimizer_name == "sgd"
    assert SMOKE_TRAINING_CONFIG.total_steps == 100
