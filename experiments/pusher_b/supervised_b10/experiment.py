from harness.context import RunContext

from experiments.pusher_b.supervised import run_supervised


def run(context: RunContext):
    return run_supervised(context, preset="b10")
