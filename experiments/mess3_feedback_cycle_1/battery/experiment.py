"""Run every feedback strength once for pre-sweep validation."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_battery


def run(context: RunContext):
    return run_battery(context)
