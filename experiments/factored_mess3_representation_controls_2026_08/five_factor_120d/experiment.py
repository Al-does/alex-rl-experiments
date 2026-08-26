"""Five independent MESS3 factors, 120d, pure next-token CE."""

from harness.context import RunContext

from ..shared import run_condition


def run(context: RunContext):
    return run_condition(context, n_factors=5, d_model=120)


__all__ = ["run"]
