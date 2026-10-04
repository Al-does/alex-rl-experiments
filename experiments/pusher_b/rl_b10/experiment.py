from harness.context import RunContext

from experiments.pusher_b.rl import run_ppo


def run(context: RunContext):
    return run_ppo(context, preset="b10")
