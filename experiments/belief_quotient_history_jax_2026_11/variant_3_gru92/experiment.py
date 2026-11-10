from experiments.belief_quotient_history_jax_2026_11.shared import run_condition


def run(context):
    return run_condition(context, variant=3, architecture="gru92")
