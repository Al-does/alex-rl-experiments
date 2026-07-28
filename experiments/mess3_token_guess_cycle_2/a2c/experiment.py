"""One-million-step, update-matched gamma-zero A2C condition.

Batch 672 gives A2C approximately the same Adam-update count as PPO while
preserving fresh-data, one-pass A2C training. ``vf_loss_coeff=1.0`` makes
A2C's half-MSE critic gradient match PPO's raw-MSE coefficient of 0.5.
"""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "a2c")
