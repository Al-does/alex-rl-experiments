from ..training import run_factors


def run(context):
    return run_factors(context, condition="reward_both", weight=0.01)
