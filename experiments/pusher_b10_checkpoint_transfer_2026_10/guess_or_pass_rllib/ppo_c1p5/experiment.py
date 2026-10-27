from __future__ import annotations

from harness.context import RunContext

from ..ppo import run_ppo


def run(context: RunContext) -> dict[str, object]:
    return run_ppo(context, cost=1.5)
