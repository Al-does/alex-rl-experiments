"""Predictive-loss coefficient 1.0 extension point."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.sweeps import run_sweep


def run(context: RunContext):
    return run_sweep(
        context,
        "predictive_loss_coefficient",
        values=(1.0,),
    )
