"""PPO plus coefficient-one next-token CE over both factor environments."""

from harness.context import RunContext

from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    run_arm,
)


def run(context: RunContext):
    return run_arm(context, "ppo_aux_ce")
