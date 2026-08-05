"""Analysis-only cycle 4, variant 2 longitudinal belief-symmetry probe."""

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes_0040.analysis import run_probe_condition
from harness.context import RunContext

CYCLE = 4
VARIANT = 2


def run(context: RunContext):
    return run_probe_condition(context, cycle=CYCLE, variant=VARIANT)
