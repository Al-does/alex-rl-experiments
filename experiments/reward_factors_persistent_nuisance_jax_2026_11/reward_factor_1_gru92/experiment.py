from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import run_condition


def run(context):
    return run_condition(context, condition="reward_factor_1", architecture="gru92")
