"""Run all five controlled conditions for pre-sweep validation."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_battery


def run(context: RunContext):
    return run_battery(context)
