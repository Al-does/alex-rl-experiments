"""Muon variant of the high-throughput supervised MESS3 replication."""

from __future__ import annotations

from harness.context import RunContext

from ..paper_supervised_replication.experiment import run_replication
from ..paper_supervised_replication.training import TrainingConfig


FULL_TRAINING_CONFIG = TrainingConfig(
    total_steps=62_500,
    analyzed_step=61_446,
    batch_size=16_384,
    optimizer_name="muon",
    learning_rate=0.02,
    weight_decay=0.0,
    momentum=0.95,
    auxiliary_learning_rate=3e-4,
    auxiliary_weight_decay=0.0,
    log_every=250,
    checkpoint_every=5_000,
    retain_periodic_checkpoints=True,
    validation_every=5_000,
    validation_batch_size=16_384,
)

SMOKE_TRAINING_CONFIG = TrainingConfig(
    total_steps=100,
    analyzed_step=100,
    batch_size=64,
    optimizer_name="muon",
    learning_rate=0.02,
    weight_decay=0.0,
    momentum=0.95,
    auxiliary_learning_rate=3e-4,
    auxiliary_weight_decay=0.0,
    log_every=10,
    checkpoint_every=50,
    validation_every=50,
    validation_batch_size=1_024,
)


def run(context: RunContext):
    return run_replication(
        context,
        full_training_config=FULL_TRAINING_CONFIG,
        smoke_training_config=SMOKE_TRAINING_CONFIG,
        variant="large-batch-muon",
    )
