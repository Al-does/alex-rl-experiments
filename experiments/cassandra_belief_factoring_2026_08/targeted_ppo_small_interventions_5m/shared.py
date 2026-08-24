"""Shared recipe for one-at-a-time targeted PPO interventions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config as build_shared_config,
    environment_config,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


Intervention = Literal[
    "vf_clip_100",
    "lambda_098",
    "bptt_64",
    "previous_reward",
]

TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 4_096
EXPERIMENT_SEED = 42
ENTROPY_COEFF = 0.03
BASELINE_VF_CLIP = 10.0
BASELINE_LAMBDA = 0.95
BASELINE_MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()


class CassandraPreviousRewardObservationEnv(CassandraActionObservationEnv):
    """Append the immediately preceding scalar reward to the policy input."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        base_space = self.observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("Cassandra policy observations must use a Box space")
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [base_space.low, np.array([-np.inf], dtype=np.float32)]
            ),
            high=np.concatenate(
                [base_space.high, np.array([np.inf], dtype=np.float32)]
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _with_reward(observation: np.ndarray, reward: float) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(observation, dtype=np.float32),
                np.asarray([reward], dtype=np.float32),
            ]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        return self._with_reward(observation, 0.0), info

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        return (
            self._with_reward(observation, reward),
            reward,
            terminated,
            truncated,
            info,
        )


def _require_comparison_seed(context: RunContext) -> None:
    if context.seed != EXPERIMENT_SEED:
        raise ValueError(
            "Cassandra intervention comparisons require seed 42; "
            f"received {context.seed!r}"
        )


def build_config(
    context: RunContext,
    *,
    intervention: Intervention,
) -> PPOConfig:
    """Build one controlled targeted intervention from the dim-64 baseline."""

    _require_comparison_seed(context)
    env_config = environment_config(action_scope="targeted")
    env_config["initial_state_distribution"] = "all_good"
    model_config = dict(BASELINE_MODEL_CONFIG)
    env_class = CassandraActionObservationEnv

    if intervention == "bptt_64":
        model_config["context_len"] = 64
        model_config["max_seq_len"] = 64
    elif intervention == "previous_reward":
        env_class = CassandraPreviousRewardObservationEnv
    elif intervention not in {"vf_clip_100", "lambda_098"}:
        raise ValueError(f"unknown intervention: {intervention}")

    return (
        build_shared_config(context, action_scope="targeted")
        .environment(env_class, env_config=env_config)
        .training(
            entropy_coeff=ENTROPY_COEFF,
            gamma=0.990,
            lambda_=(
                0.98 if intervention == "lambda_098" else BASELINE_LAMBDA
            ),
            vf_clip_param=(
                100.0
                if intervention == "vf_clip_100"
                else BASELINE_VF_CLIP
            ),
            use_kl_loss=False,
            kl_coeff=0.0,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=model_config,
            )
        )
    )


def run_intervention(
    context: RunContext,
    *,
    intervention: Intervention,
    hypothesis: str,
) -> dict[str, Any]:
    """Train one intervention and emit a compact, comparable run summary."""

    config = build_config(context, intervention=intervention)
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": f"targeted_ppo_small_{intervention}",
            "hypothesis": hypothesis,
            "primary_comparison": (
                "one intervention versus the otherwise fixed targeted dim-64 "
                "transformer PPO baseline"
            ),
            "seed": EXPERIMENT_SEED,
            "algorithm": "PPO",
            "environment": environment_config(action_scope="targeted")
            | {"initial_state_distribution": "all_good"},
            "transformer": dict(config.rl_module_spec.model_config),
            "gamma": config.gamma,
            "lambda": config.lambda_,
            "vf_clip_param": config.vf_clip_param,
            "entropy_coeff": config.entropy_coeff,
            "use_kl_loss": config.use_kl_loss,
            "previous_reward_visible": intervention == "previous_reward",
            "total_env_steps": target_steps,
        },
    )
    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(
            f"{intervention} expected one trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{intervention} training failed") from result.error
    summary = {
        "condition": f"targeted_ppo_small_{intervention}",
        "seed": EXPERIMENT_SEED,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "status": "completed",
        "checkpoint": (
            str(result.checkpoint.path)
            if result.checkpoint is not None
            else None
        ),
    }
    outputs.write_json("intervention_summary.json", summary)
    return summary


__all__ = [
    "BASELINE_LAMBDA",
    "BASELINE_MODEL_CONFIG",
    "BASELINE_VF_CLIP",
    "ENTROPY_COEFF",
    "EXPERIMENT_SEED",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "CassandraPreviousRewardObservationEnv",
    "build_config",
    "run_intervention",
]
