"""Pure next-token training over two and three independent MESS3 factors."""

from harness.context import RunContext

from ..shared import run_study


def run(context: RunContext):
    return run_study(context)
