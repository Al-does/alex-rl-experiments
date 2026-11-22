from ..training import run_ladder


def run(context):
    return run_ladder(context, variant=2, weight=0.1)
