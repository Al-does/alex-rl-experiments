from experiments.belief_quotient_persistent_nuisance_jax_egreedy_2026_11.shared import run_condition


def run(context):
    return run_condition(context, variant=3, architecture="transformer", epsilon="eps0p2")
