"""Four-point next-token predictive-loss coefficient sweep."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.sweeps import run_sweep


def run(context: RunContext):
    return run_sweep(context, "predictive_loss_coefficient")
