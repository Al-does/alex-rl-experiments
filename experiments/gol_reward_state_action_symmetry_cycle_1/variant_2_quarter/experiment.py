from experiments.gol_reward_state_action_symmetry_cycle_1 import shared


def build_config(context):
    return shared.build_config(context, variant=2, speed="quarter")


def run(context):
    return shared.run_condition(context, variant=2, speed="quarter")
