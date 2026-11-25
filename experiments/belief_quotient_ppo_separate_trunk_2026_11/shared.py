"""eps02 ladder replay of the 2B-step shared recipe and its trunk ablations."""
from dataclasses import replace
from functools import partial

import jax

from experiments.belief_quotient_history_jax_2026_11 import shared as ladder, training
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)

from . import probes, reference, split_ppo

STUDY = "belief_quotient_ppo_separate_trunk_2026_11"


def run_condition(context, *, variant, arm):
    if not context.smoke and jax.default_backend() != "gpu":
        raise RuntimeError("full PPO ladder runs require a JAX GPU backend")
    process = environment.ladder_process(variant)
    config = ppo.configure(replace(
        ladder.train_config(context.smoke, TOTAL_ENV_STEPS),
        entropy_schedule=ENTROPY_SCHEDULE, temperature=TEMPERATURE,
    ))
    spec = ladder.model_spec("transformer", process)
    if arm == "shared_wide":
        spec = replace(spec, width=90)
    divisions = 16 if context.smoke else 160
    reward_reference = reference.REWARD_REFERENCES[variant]
    values = reference.value_grid(process, config.gamma, config.episode_length, divisions)
    return training.run_study(
        context, study=STUDY, condition=f"variant_{variant}", architecture="transformer",
        process=process, spec=spec, config=config, targets=probes.belief_targets,
        representation="embedding", trainer=split_ppo.train if arm == "separate" else ppo.train,
        analyze_checkpoint=partial(probes.probe_checkpoint, values=values, divisions=divisions,
                                   reward_reference=reward_reference),
        recipe_extra={
            **environment.environment_metadata(), "variant": variant, "arm": arm,
            "condition_name": ladder.CONDITION_NAMES[variant],
            "reward": "pre-transition state-2 occupancy",
            "representation": "post_final_layer_norm",
            "reward_reference": reward_reference,
            "reward_reference_revision": "ebc16c672f317a4b652580daf256591bd30ab3ef",
            "value_target": {"kind": "finite_horizon_discounted_belief_grid",
                             "divisions": divisions, "gamma": config.gamma},
        },
    )
