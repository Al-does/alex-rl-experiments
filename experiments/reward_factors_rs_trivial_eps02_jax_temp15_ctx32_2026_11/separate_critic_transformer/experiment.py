from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.shared import run_condition


def run(context):
    return run_condition(context, arm="separate_critic")
