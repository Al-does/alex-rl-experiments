"""Two independent MESS3 factors, 64d, pure next-token CE."""

from harness.context import RunContext

from ..shared import run_condition


def run(context: RunContext):
    return run_condition(context, n_factors=2, d_model=64)


__all__ = ["run"]
