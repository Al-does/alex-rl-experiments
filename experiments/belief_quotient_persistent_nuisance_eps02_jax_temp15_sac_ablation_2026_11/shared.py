"""SAC Reward-State delta ablations: alpha=0 (no entropy) and tau=0.05 (10x faster Polyak)."""
from __future__ import annotations

import math

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_2026_11 import sac
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11 import (
    shared as autoalpha,
)
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE,
    TEMPERATURE,
    TOTAL_ENV_STEPS,
)

STUDY = "belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_ablation_2026_11"

ABLATIONS = ("alpha0", "tau05")


def hyperparameters(ablation: str, n_actions: int) -> dict:
    if ablation == "alpha0":
        return {
            "auto_entropy_tuning": False,
            "alpha": 0.0,
            "tau": 0.005,
            "lr": autoalpha.HYPERPARAMETERS["lr"],
            "critic_lr": autoalpha.HYPERPARAMETERS["critic_lr"],
            "q_head_init": autoalpha.HYPERPARAMETERS["q_head_init"],
        }
    if ablation == "tau05":
        return {
            **autoalpha.hyperparameters(n_actions),
            "tau": 0.05,
        }
    raise ValueError(f"ablation must be one of {ABLATIONS}")


def run_condition(context, *, ablation: str, variant: int = 2):
    if variant != 2:
        raise ValueError("this SAC ablation study targets Reward-State Quotient (variant 2)")
    process = environment.ladder_process(variant)
    backend = sac.for_context(context, **hyperparameters(ablation, process.n_actions))
    return base.run_condition(
        context,
        variant=variant,
        architecture="transformer",
        study=STUDY,
        total_env_steps=TOTAL_ENV_STEPS,
        entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE,
        environment=environment,
        training_backend=backend,
    )
