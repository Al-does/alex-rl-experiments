"""Run both factor counts with discrete SAC plus next-token CE."""

from experiments.factored_representations_reproduction_SAC_2026_08.shared import (
    run_arm,
)


def run(context):
    return run_arm(context, "sac_aux_ce")
