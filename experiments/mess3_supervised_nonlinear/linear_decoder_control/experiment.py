"""Unchanged paper MESS3 transformer with its linear 64 -> 3 decoder."""

from __future__ import annotations

from harness.context import RunContext

from ..shared import run_linear_control


def run(context: RunContext):
    return run_linear_control(context)
