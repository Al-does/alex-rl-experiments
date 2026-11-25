"""Reward-State x Trivial factors: actor-only vs shared vs separate critic."""
from dataclasses import replace

from experiments.belief_quotient_history_jax_2026_11.training import run_study
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)
from experiments.reward_factors_history_jax_2026_11 import shared as source
from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11 import algorithms, design
from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.process import (
    environment_metadata, make_targets, rs_trivial_process,
)

STUDY = "reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11"
CONTEXT_LEN = 32
ARCHITECTURE = "transformer"
ARMS = algorithms.ARMS


def run_condition(context, *, arm):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    backend = algorithms.BACKENDS[arm]
    process = rs_trivial_process()
    config = replace(source.train_config(context.smoke, TOTAL_ENV_STEPS),
                     entropy_schedule=tuple(ENTROPY_SCHEDULE), temperature=TEMPERATURE,
                     exploration_epsilon=0.0)
    config = backend.configure(config)
    spec = replace(source.model_spec(ARCHITECTURE, process), context_len=CONTEXT_LEN)
    return run_study(
        context, study=STUDY, condition=arm, architecture=ARCHITECTURE, process=process,
        spec=spec, config=config, targets=make_targets(design.value_star),
        representation="residual", trainer=backend.train,
        recipe_extra={**environment_metadata(), "arm": arm, **backend.extra,
                      "value_star_target": "(1-gamma) V*(b), gamma 0.99, per-factor full-simplex grid VI (design.py)",
                      "representation": "pre_final_layer_norm"},
    )
