"""250M desynced previous-reward PPO with d_model=120 transformer."""

from __future__ import annotations

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    run_recipe,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted_previous_reward_desynced.experiment import (
    ACTION_SCOPE,
    HYPOTHESIS,
    PRIMARY_COMPARISON,
    build_config as build_desynced_config,
)
from harness.context import RunContext
from learners.models.transformer import TransformerModel


CONDITION = "best_critic_bptt64_250m_targeted_previous_reward_desynced_d120"
D_MODEL = 120


def build_config(context: RunContext) -> PPOConfig:
    config = build_desynced_config(context)
    model_config = dict(config.rl_module_spec.model_config)
    model_config["d_model"] = D_MODEL
    return config.rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=TransformerModel,
            model_config=model_config,
        )
    )


def run(context: RunContext):
    return run_recipe(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
        previous_reward_visible=True,
        config_builder=build_config,
        recipe_metadata={
            "hypothesis": (
                f"{HYPOTHESIS} Scale transformer width to d_model={D_MODEL} "
                "while holding desync and best-critic settings fixed."
            ),
            "primary_comparison": (
                f"d_model={D_MODEL} desynced previous-reward BPTT-64 versus "
                "d_model=64 desynced previous-reward BPTT-64 at 250M steps"
            ),
            "episode_desync": {
                "enabled": True,
                "mode": "deterministic_one_time_initial_horizon",
                "subsequent_episode_length": 1_000,
            },
            "transformer_d_model": D_MODEL,
        },
    )


__all__ = ["CONDITION", "D_MODEL", "build_config", "run"]
