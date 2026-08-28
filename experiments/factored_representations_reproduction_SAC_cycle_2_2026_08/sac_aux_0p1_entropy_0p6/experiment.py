"""Run cycle-2 SAC with CE=0.1 and 60% maximum categorical entropy."""

from experiments.factored_representations_reproduction_SAC_cycle_2_2026_08.shared import (
    run_arm,
)


def run(context):
    return run_arm(
        context,
        condition="sac_aux_ce",
        target_entropy_fraction=0.6,
        auxiliary_coefficient=0.1,
    )
