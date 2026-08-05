"""Analysis-only cycle 5, variant 2 belief-symmetry probe."""

from experiments.mess3_reward_state_action_symmetry_cycle_5.belief_symmetry_probes.analysis import run_probe_condition
from harness.context import RunContext

CYCLE = 5
VARIANT = 2


def run(context: RunContext):
    return run_probe_condition(context, cycle=CYCLE, variant=VARIANT)
