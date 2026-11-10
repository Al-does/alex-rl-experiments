from experiments.reward_factors_history_jax_lumped_temp15_2026_11.shared import run_condition


def run(context):
    return run_condition(context, condition="reward_factor_1", architecture="memoryless")
