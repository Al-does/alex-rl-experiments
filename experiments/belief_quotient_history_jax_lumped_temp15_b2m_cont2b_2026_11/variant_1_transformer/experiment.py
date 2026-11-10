from experiments.belief_quotient_history_jax_lumped_temp15_b2m_cont2b_2026_11.shared import run_condition


def run(context):
    return run_condition(context, variant=1, architecture="transformer")
