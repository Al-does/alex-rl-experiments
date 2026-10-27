from __future__ import annotations

from harness.context import RunContext

from ..sac import run_sac


def run(context: RunContext) -> dict[str, object]:
    return run_sac(context, cost=2.0)
