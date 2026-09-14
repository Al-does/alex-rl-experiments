"""Train through the article's reported 10,000-step loss convergence point."""

from harness.context import RunContext

from experiments.nonergodic_mess3_supervised.alpha_params.shared import (
    run_condition,
)
from experiments.nonergodic_mess3_supervised.alpha_params.training import (
    TrainingConfig,
)


def run(context: RunContext):
    return run_condition(
        context,
        condition="loss_convergence",
        full_training_config=TrainingConfig.convergence(),
    )
