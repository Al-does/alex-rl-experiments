from ..training import run_ladder


def run(context):
    return run_ladder(context, variant=3, weight=0.01)
