from experiments.belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11.shared import run_condition


def run(context):
    return run_condition(context, variant=3, architecture="transformer", entropy="ent0p005")
