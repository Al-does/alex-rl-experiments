"""PPO-only arm over two and three independent MESS3 factors."""

from harness.context import RunContext

from experiments.factored_representations_reproduction_2026_08.shared import (
    run_arm,
)


def run(context: RunContext):
    return run_arm(context, "ppo")
