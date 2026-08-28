"""Run reward-only cycle-2 SAC at 30% of maximum categorical entropy."""

from experiments.factored_representations_reproduction_SAC_cycle_2_2026_08.shared import (
    run_arm,
)


def run(context):
    return run_arm(
        context,
        condition="sac",
        target_entropy_fraction=0.3,
    )
