"""Paired truncated arms isolating whether the guess is used, not just seen.

``strong_feedback`` and ``strong_feedback_blind`` share the same kappa = 0.7
dynamics and differ only in whether the previous guess appears in the
observation. Any gap in action awareness between them therefore comes from
the guess being readable, not from the dynamics.
"""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.analysis import ProbeBudget
from experiments.mess3_feedback_cycle_1.shared import (
    CONTRAST_ENV_STEPS,
    CONTRAST_PROBE_BUDGET,
    run_contrast,
)


CONTRAST_CONDITIONS = ("strong_feedback", "strong_feedback_blind")
SMOKE_ENV_STEPS = 4_096
SMOKE_PROBE_BUDGET = ProbeBudget(
    calibration=512,
    train=2_048,
    test=2_048,
    resamples=50,
)


def run(context: RunContext):
    return run_contrast(
        context,
        CONTRAST_CONDITIONS,
        target_steps=SMOKE_ENV_STEPS if context.smoke else CONTRAST_ENV_STEPS,
        probe_budget=(
            SMOKE_PROBE_BUDGET if context.smoke else CONTRAST_PROBE_BUDGET
        ),
    )
