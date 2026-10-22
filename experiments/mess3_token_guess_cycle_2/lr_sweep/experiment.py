"""Four-point PPO learning-rate sweep spanning over one order of magnitude."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.sweeps import run_sweep


def run(context: RunContext):
    return run_sweep(context, "learning_rate")
