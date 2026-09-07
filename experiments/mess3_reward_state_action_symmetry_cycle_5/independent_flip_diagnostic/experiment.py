"""Run the Cycle-5 Variant-2 independent token-flip diagnostic."""

from experiments.mess3_reward_state_action_symmetry_cycle_5.independent_flip_diagnostic.analysis import (
    run_independent_flip_diagnostic,
)
from harness.context import RunContext

CYCLE = 5
VARIANT = 2


def run(context: RunContext):
    return run_independent_flip_diagnostic(context)
