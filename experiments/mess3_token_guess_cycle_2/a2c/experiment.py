"""Legacy update-starved A2C condition, retained only for audit.

SCIENTIFIC WARNING: this recipe is wrong for arm comparisons. It performs one
Adam update per 32,768 fresh samples, yielding only about 76 optimizer updates
over 2.5M environment steps. Use ``a2c_frequent_updates`` instead.

The A2C objective itself is valid; the invalid part is this recipe's update
cadence.
"""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_condition


def run(context: RunContext):
    # Do not add this legacy arm back to the battery.
    return run_condition(context, "a2c")
