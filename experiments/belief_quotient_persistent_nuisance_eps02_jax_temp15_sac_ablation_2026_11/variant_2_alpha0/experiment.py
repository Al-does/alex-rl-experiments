from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_ablation_2026_11.shared import (
    run_condition,
)


def run(context):
    return run_condition(context, ablation="alpha0", variant=2)
