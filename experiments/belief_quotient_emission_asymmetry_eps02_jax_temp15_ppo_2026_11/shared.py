"""Emission-asymmetry follow-up using the existing JAX PPO backend."""
from dataclasses import replace

from experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11 import (
    environment,
    shared as baseline,
)
from experiments.belief_quotient_history_jax_2026_11.training import run_study
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo

STUDY = "belief_quotient_emission_asymmetry_eps02_jax_temp15_ppo_2026_11"
ENTROPY_SCHEDULE = ((0, 0.03), (1_800_000_000, 0.0))
LEVELS = {"000": 0.0, **environment.LEVELS}


def run_condition(context, *, level):
    kappa = LEVELS[level]
    process = environment.ladder_process(kappa)
    config = ppo.configure(replace(
        baseline.base.train_config(context.smoke, baseline.source.TOTAL_ENV_STEPS),
        entropy_schedule=ENTROPY_SCHEDULE,
        temperature=baseline.source.TEMPERATURE,
        exploration_epsilon=0.0,
    ))
    return run_study(
        context, study=STUDY, condition=f"asym_{level}",
        architecture=baseline.ARCHITECTURE, process=process,
        spec=baseline.base.model_spec(baseline.ARCHITECTURE, process), config=config,
        targets=baseline.belief_targets, representation="embedding",
        recipe_extra=environment.environment_metadata(kappa), trainer=ppo.train,
    )
