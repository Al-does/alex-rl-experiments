from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11.shared import run_condition


def run(context):
    return run_condition(context, variant=1, architecture="transformer")
