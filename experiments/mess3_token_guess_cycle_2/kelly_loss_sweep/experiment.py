"""Four-point decoupled Kelly-loss coefficient sweep."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.sweeps import run_sweep


def run(context: RunContext):
    return run_sweep(context, "kelly_loss_coefficient")
