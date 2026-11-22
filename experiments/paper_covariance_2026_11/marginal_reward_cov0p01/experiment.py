from ..training import run_factors


def run(context):
    return run_factors(context, condition="reward_factor_1", weight=0.01)
