"""Run all five controlled conditions for pre-sweep validation."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_battery


def run(context: RunContext):
    # The ``a2c`` arm uses the corrected 672-sample, update-matched recipe.
    return run_battery(context)
