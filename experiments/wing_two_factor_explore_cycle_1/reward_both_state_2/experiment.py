from experiments.wing_two_factor_explore_cycle_1 import shared


CONDITION = "reward_both"
REWARD_STATE = 2


def build_config(context):
    return shared.build_config(context, CONDITION, REWARD_STATE)


def run(context):
    return shared.run_condition(context, CONDITION, REWARD_STATE)
