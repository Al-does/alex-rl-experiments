"""Run the cycle-5 variant-2 state-token swap diagnostic."""

from experiments.mess3_reward_state_action_symmetry_cycle_5.token_swap_diagnostic.analysis import (
    run_token_swap_diagnostic,
)
from harness.context import RunContext

CYCLE = 5
VARIANT = 2


def run(context: RunContext):
    return run_token_swap_diagnostic(context, cycle=CYCLE)
