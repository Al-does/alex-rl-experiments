"""Run all five controlled conditions for pre-sweep validation.

The battery uses ``a2c_frequent_updates``. The legacy ``a2c`` leaf is
update-starved and remains available only to reproduce the original run.
"""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_battery


def run(context: RunContext):
    # ``run_battery`` deliberately points at the corrected A2C experiment.
    return run_battery(context)
