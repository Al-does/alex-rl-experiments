"""Run reward-only discrete SAC on two independent factors."""

from experiments.factored_representations_reproduction_SAC_2026_08.shared import (
    run_factor_condition,
)


def run(context):
    return run_factor_condition(context, factor_count=2, condition="sac")
