from experiments.gol_reward_state_action_symmetry_cycle_1_entropy_0_01_20m import shared


def build_config(context):
    return shared.build_config(context, variant=3)


def run(context):
    return shared.run_condition(context, variant=3)
