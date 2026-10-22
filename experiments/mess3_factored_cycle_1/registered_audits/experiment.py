"""Run the preregistered A1--A6 QMDP campaign."""

from harness.context import RunContext

from experiments.mess3_factored_cycle_1.reference_campaign import (
    run as run_campaign,
)


def run(context: RunContext):
    return run_campaign(context)
