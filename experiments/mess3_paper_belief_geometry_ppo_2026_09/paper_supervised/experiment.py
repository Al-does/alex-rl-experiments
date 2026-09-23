"""Paper recipe: next-token cross-entropy, SGD lr 0.01, batch 64, 1M updates."""

from harness.context import RunContext

from ..shared import run_paper_supervised
from ..supervised import TrainingConfig


# The paper's recipe. Probed checkpoints: init, update 3,125 (2M agent steps,
# the PPO arm's full budget), 50K, 250K, the analyzed 983,140, and 1M.
FULL_TRAINING_CONFIG = TrainingConfig(
    extra_checkpoint_steps=(3_125, 50_000, 250_000),
)
SMOKE_TRAINING_CONFIG = TrainingConfig.smoke()


def run(context: RunContext):
    return run_paper_supervised(
        context,
        full_config=FULL_TRAINING_CONFIG,
        smoke_config=SMOKE_TRAINING_CONFIG,
    )
