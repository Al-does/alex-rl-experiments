"""Pure next-token training on three independent MESS3 factors."""

from harness.context import RunContext

from ..shared import run_factor_count


def run(context: RunContext):
    return run_factor_count(context, factor_count=3)
