"""Lumped-baseline study rerun from scratch for 4B steps with no entropy bonus
(coefficient 0.0 throughout) and softmax sampling temperature 1.5, applied
identically to rollout sampling and the PPO likelihoods. Greedy evaluation and
probes are unchanged (argmax is temperature-invariant)."""

from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as base

STUDY = "belief_quotient_history_jax_lumped_temp15_2026_11"
TOTAL_ENV_STEPS = 4_000_000_000
ENTROPY_SCHEDULE = ((0, 0.0),)
TEMPERATURE = 1.5


def run_condition(context, *, variant, architecture):
    return base.run_condition(
        context, variant=variant, architecture=architecture, total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY, entropy_schedule_points=ENTROPY_SCHEDULE, temperature=TEMPERATURE,
    )
