from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_2026_11.shared import run_condition


def run(context):
    return run_condition(context, condition="reward_both", architecture="gru92")
