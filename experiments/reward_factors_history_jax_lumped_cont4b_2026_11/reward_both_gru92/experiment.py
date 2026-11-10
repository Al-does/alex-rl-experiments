from experiments.reward_factors_history_jax_lumped_cont4b_2026_11.shared import run_condition


def run(context):
    return run_condition(context, condition="reward_both", architecture="gru92")
