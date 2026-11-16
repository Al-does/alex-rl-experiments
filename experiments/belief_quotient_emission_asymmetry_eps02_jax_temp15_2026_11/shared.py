"""PR #234 Reward-State transformer recipe on emission-asymmetric processes."""
from dataclasses import replace

from experiments.belief_quotient_history_jax_2026_11.training import run_study
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import (
    shared as source,
)
from experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11 import environment

STUDY = "belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11"
ARCHITECTURE = "transformer"
base = source.base


def belief_targets(belief):
    return {"belief": belief, "reward_class": belief[:, 2:],
            "identity": belief[:, :1] - belief[:, 1:2]}


def run_condition(context, *, level):
    kappa = environment.LEVELS[level]
    process = environment.ladder_process(kappa)
    config = replace(
        base.train_config(context.smoke, source.TOTAL_ENV_STEPS),
        entropy_schedule=tuple(source.ENTROPY_SCHEDULE), temperature=source.TEMPERATURE,
        exploration_epsilon=0.0,
    )
    return run_study(
        context, study=STUDY, condition=f"asym_{level}", architecture=ARCHITECTURE,
        process=process, spec=base.model_spec(ARCHITECTURE, process), config=config,
        targets=belief_targets, representation="embedding",
        recipe_extra=environment.environment_metadata(kappa),
    )
