"""Run reward-only cycle-2 SAC at 30% entropy on two independent factors."""

from experiments.factored_representations_reproduction_SAC_cycle_2_2026_08.shared import (
    run_factor_condition,
)


def run(context):
    return run_factor_condition(
        context,
        factor_count=2,
        condition="sac",
        target_entropy_fraction=0.3,
    )
