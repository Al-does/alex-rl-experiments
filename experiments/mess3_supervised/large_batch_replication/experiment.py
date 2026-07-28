"""High-throughput large-batch variant of the supervised MESS3 replication."""

from __future__ import annotations

from harness.context import RunContext

from ..paper_supervised_replication.experiment import run_replication
from ..paper_supervised_replication.training import TrainingConfig


# A 256x larger batch with square-root learning-rate scaling. Reducing the
# update budget by the same 16x factor preserves the original recipe's
# cumulative learning-rate exposure (1_000_000 * 0.01 == 62_500 * 0.16).
FULL_TRAINING_CONFIG = TrainingConfig(
    total_steps=62_500,
    analyzed_step=61_446,
    batch_size=16_384,
    learning_rate=0.16,
    weight_decay=0.0,
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
    learning_rate=0.01,
    weight_decay=0.0,
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
        variant="large-batch-sqrt-scaled",
    )
