"""PPO-only arm on three independent MESS3 factors."""

from harness.context import RunContext

from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    run_factor_condition,
)


def run(context: RunContext):
    return run_factor_condition(context, factor_count=3, condition="ppo")
