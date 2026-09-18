"""Train through the step-45,000 checkpoint used for the article figures."""

from harness.context import RunContext

from experiments.nonergodic_mess3_supervised.shared import run_condition
from experiments.nonergodic_mess3_supervised.training import TrainingConfig


def run(context: RunContext):
    return run_condition(
        context,
        condition="geometry_checkpoint",
        full_training_config=TrainingConfig.figure_checkpoint(),
    )
