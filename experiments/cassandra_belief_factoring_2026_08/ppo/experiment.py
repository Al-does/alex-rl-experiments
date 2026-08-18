"""Train and probe a transformer on Cassandra's global-action POMDP."""

from harness.context import RunContext

from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config,
    run_condition,
)


def run(context: RunContext):
    return run_condition(context)


__all__ = ["build_config", "run"]
