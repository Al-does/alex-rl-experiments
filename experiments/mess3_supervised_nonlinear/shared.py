"""Shared controlled recipe for nonlinear supervised MESS3 decoders."""

from __future__ import annotations

from harness.context import RunContext

from experiments.mess3_supervised.large_batch_replication.experiment import (
    FULL_TRAINING_CONFIG,
    SMOKE_TRAINING_CONFIG,
)
from experiments.mess3_supervised.paper_supervised_replication.experiment import (
    run_replication,
)

from .model import SwishMLPDecoderTransformer


DECODER_HIDDEN_DIM = 64


def run_linear_control(context: RunContext):
    """Run the unchanged paper model as the linear-head control."""

    return run_replication(
        context,
        full_training_config=FULL_TRAINING_CONFIG,
        smoke_training_config=SMOKE_TRAINING_CONFIG,
        variant="linear-decoder-control",
    )


def run_decoder_replication(
    context: RunContext,
    *,
    decoder_depth: int,
):
    """Run the paper recipe with only its prediction decoder changed."""

    return run_replication(
        context,
        full_training_config=FULL_TRAINING_CONFIG,
        smoke_training_config=SMOKE_TRAINING_CONFIG,
        variant=f"swish-mlp-decoder-{decoder_depth}-layer",
        model_class=SwishMLPDecoderTransformer,
        model_kwargs={
            "decoder_depth": decoder_depth,
            "decoder_hidden_dim": DECODER_HIDDEN_DIM,
        },
    )
