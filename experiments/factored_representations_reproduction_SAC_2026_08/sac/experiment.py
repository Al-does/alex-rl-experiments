"""Run both factor counts with reward-only discrete SAC."""

from experiments.factored_representations_reproduction_SAC_2026_08.shared import (
    run_arm,
)


def run(context):
    return run_arm(context, "sac")
