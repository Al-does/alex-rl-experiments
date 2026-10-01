from experiments.strata_two_factor_explore_cycle_3.context_length_64 import shared


CONDITION = "reward_both"
REWARD_STATE = 0


def build_config(context):
    return shared.build_config(context, CONDITION)


def run(context):
    return shared.run_condition(context, CONDITION)
