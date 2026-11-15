from experiments.belief_quotient_persistent_nuisance_jax_2b_2026_11.shared import run_condition


def run(context):
    return run_condition(context, variant=2, architecture="memoryless")
