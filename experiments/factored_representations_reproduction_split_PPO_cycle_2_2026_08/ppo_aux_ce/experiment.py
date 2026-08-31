"""Run split-network PPO with actor-only next-token cross entropy."""

from harness.context import RunContext

from experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.shared import (
    run_arm,
)


def run(context: RunContext):
    return run_arm(context, "ppo_aux_ce")
