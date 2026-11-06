"""Architecture-only REINFORCE controls for the two belief batteries."""

from __future__ import annotations

import importlib
import math
from functools import partial

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_7 import shared as ladder_source
from experiments.two_factor_reward_state_REINFORCE_cycle_4 import shared as reward_source
from learners.models import (
    GRUModel, GRUModelConfig, ResidualMLPModel, ResidualMLPModelConfig,
)


class ZeroBaselineTemperature:
    def setup(self):
        super().setup()
        self.sampling_temperature = float(self.model_config.get("sampling_temperature", 1.0))
        if not math.isfinite(self.sampling_temperature) or self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")

    def compute_values(self, batch, embeddings=None):
        reference = embeddings if embeddings is not None else batch[Columns.OBS]
        return torch.zeros(reference.shape[:-1], dtype=reference.dtype, device=reference.device)

    def action_distribution_inputs(self, embeddings):
        return super().action_distribution_inputs(embeddings) / self.sampling_temperature

    def _outputs(self, embeddings, state_out, *, training):
        outputs = super()._outputs(embeddings, state_out, training=training)
        outputs[Columns.ACTION_DIST_INPUTS] = self.action_distribution_inputs(embeddings)
        return outputs


class ReinforceGRU(ZeroBaselineTemperature, GRUModel):
    pass


class ReinforceResidualMLP(ZeroBaselineTemperature, ResidualMLPModel):
    pass


def model_spec(architecture, *, ladder):
    if architecture == "memoryless":
        config = ResidualMLPModelConfig(
            d_model=64, n_layers=6,
            obs_feature_indices=tuple(range(3 if ladder else 9)),
        ).to_dict()
        cls = ReinforceResidualMLP
    elif architecture == "gru":
        config = GRUModelConfig(hidden_dim=128, n_layers=2, max_seq_len=32).to_dict()
        cls = ReinforceGRU
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    return RLModuleSpec(module_class=cls, model_config={**config, "sampling_temperature": 1.0})


def build_config(context, condition, *, architecture, ladder):
    if ladder:
        source = importlib.import_module(
            "experiments.mess3_reward_state_action_symmetry_cycle_7."
            f"variant_{condition}_ctx32_ent.experiment"
        )
        config = source.build_config(context)
    else:
        config = reward_source.build_config(context, condition)
    return config.rl_module(rl_module_spec=model_spec(architecture, ladder=ladder))


def recipe_overrides(context, condition, *, study, architecture, ladder):
    config = build_config(context, condition, architecture=architecture, ladder=ladder)
    env = HMMEnv(config.env_config)
    try:
        spec = config.get_rl_module_spec()
        module = spec.module_class(observation_space=env.observation_space,
                                   action_space=env.action_space, model_config=spec.model_config)
        count = sum(parameter.numel() for parameter in module.parameters())
    finally:
        env.close()
    recipe = {
        "study": study, "architecture": architecture,
        "model_config": spec.model_config, "parameter_count": count,
    }
    if ladder:
        source = importlib.import_module(
            "experiments.mess3_reward_state_action_symmetry_cycle_7."
            f"variant_{condition}_ctx32_ent.experiment"
        )
        recipe.update({"experiment_arm": architecture,
                       "sampling_temperature": 1.0,
                       "entropy_coeff_schedule": source.entropy_schedule(context),
                       "entropy_anneal_remaining_steps": source.ENTROPY_ANNEAL_REMAINING_STEPS})
    else:
        recipe.update({"model": spec.model_config, "transformer_raw_observation_lookback": None})
    return recipe


def run_condition(context, condition, *, study, architecture, ladder):
    source_run = ladder_source.run_condition if ladder else reward_source.run_condition
    return source_run(
        context, condition,
        config_builder=partial(build_config, architecture=architecture, ladder=ladder),
        recipe_overrides=recipe_overrides(context, condition, study=study,
                                          architecture=architecture, ladder=ladder),
    )
