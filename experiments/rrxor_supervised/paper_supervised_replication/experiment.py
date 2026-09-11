"""Reproduce the paper's supervised RRXOR belief-geometry experiment."""

from harness.context import RunContext

from ..shared import run_condition


def run(context: RunContext):
    return run_condition(context)
