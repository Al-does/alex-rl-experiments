from harness.context import RunContext

from ..ppo import run_ppo


def run(context: RunContext):
    return run_ppo(context, cost=1.0)
