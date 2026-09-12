"""PPO on the Simplex two-component non-ergodic MESS3 process."""

from harness.context import RunContext

from experiments.nonergodic_mess3_token_guess_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context)
