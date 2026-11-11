"""Persistent-nuisance ladder transformer arms rerun from scratch for 2B steps with no
entropy bonus, sampling temperature 1.0 and epsilon-greedy exploration: actions are
sampled from ``(1 - epsilon) * softmax(logits) + epsilon / n_actions`` and the PPO ratio
uses that same mixture's log-likelihoods. Greedy evaluation and probes are unchanged."""

from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_2b_2026_11 import shared as budget

STUDY = "belief_quotient_persistent_nuisance_jax_egreedy_2026_11"
TOTAL_ENV_STEPS = budget.TOTAL_ENV_STEPS
ENTROPY_SCHEDULE = ((0, 0.0),)
TEMPERATURE = 1.0
EPSILONS = {"eps0p05": 0.05, "eps0p2": 0.2}


def run_condition(context, *, variant, architecture, epsilon):
    return base.run_condition(
        context, variant=variant, architecture=architecture, total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY, entropy_schedule_points=ENTROPY_SCHEDULE, temperature=TEMPERATURE,
        epsilon=EPSILONS[epsilon],
    )
