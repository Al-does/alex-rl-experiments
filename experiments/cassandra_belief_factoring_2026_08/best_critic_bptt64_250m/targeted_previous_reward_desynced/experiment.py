"""250M targeted previous-reward PPO with one-time episode desynchronization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    build_config as build_recipe_config,
    run_recipe,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    CassandraPreviousRewardObservationEnv,
)
from harness.context import RunContext


ACTION_SCOPE = "targeted"
CONDITION = "best_critic_bptt64_250m_targeted_previous_reward_desynced"
DESYNC_SEED_KEY = "initial_episode_desync_seed"
DESYNC_ENVS_PER_RUNNER_KEY = "initial_episode_desync_envs_per_runner"

HYPOTHESIS = (
    "A deterministic one-time stagger of the first episode will spread episode "
    "ages across each PPO batch, reducing phase-correlated gradient updates "
    "while preserving every subsequent 1,000-step episode."
)
PRIMARY_COMPARISON = (
    "desynchronized versus synchronized targeted previous-reward BPTT-64 PPO "
    "under otherwise identical 250M-step best-critic settings"
)


def initial_episode_horizon(
    *,
    episode_length: int,
    seed: int,
    worker_index: int,
    vector_index: int,
    num_workers: int,
    envs_per_runner: int,
) -> int:
    """Return an evenly spaced, seed-shifted first-episode horizon."""

    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if envs_per_runner <= 0:
        raise ValueError("envs_per_runner must be positive")
    worker_slot = max(worker_index - 1, 0)
    slot = worker_slot * envs_per_runner + vector_index
    total_slots = max(1, num_workers * envs_per_runner)
    phase = (slot % total_slots) * episode_length // total_slots
    return 1 + ((phase + seed) % episode_length)


class CassandraDesyncedPreviousRewardObservationEnv(
    CassandraPreviousRewardObservationEnv
):
    """Stagger only the first horizon, then retain the canonical fixed horizon."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        try:
            desync_seed = int(values.pop(DESYNC_SEED_KEY))
            envs_per_runner = int(values.pop(DESYNC_ENVS_PER_RUNNER_KEY))
        except KeyError as error:
            raise ValueError(
                "desynchronized Cassandra env requires desync seed and topology"
            ) from error

        episode_length = int(values.get("episode_length", 1_000))
        worker_index = int(getattr(config, "worker_index", 0))
        vector_index = int(getattr(config, "vector_index", 0))
        num_workers = int(getattr(config, "num_workers", 0) or 0)
        self.initial_episode_horizon = initial_episode_horizon(
            episode_length=episode_length,
            seed=desync_seed,
            worker_index=worker_index,
            vector_index=vector_index,
            num_workers=num_workers,
            envs_per_runner=envs_per_runner,
        )
        self._initial_episode_active = True
        self._episode_steps = 0
        self._desync_needs_reset = False
        super().__init__(values)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self._episode_steps = 0
        self._desync_needs_reset = False
        return observation, info

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._desync_needs_reset:
            raise RuntimeError("reset must be called after episode truncation")
        observation, reward, terminated, truncated, info = super().step(action)
        self._episode_steps += 1
        if (
            self._initial_episode_active
            and self._episode_steps >= self.initial_episode_horizon
        ):
            self._initial_episode_active = False
            self._desync_needs_reset = True
            truncated = True
        return observation, reward, terminated, truncated, info


def build_config(context: RunContext) -> PPOConfig:
    """Build the synchronized recipe with only its environment adapter changed."""

    config = build_recipe_config(
        context,
        action_scope=ACTION_SCOPE,
        previous_reward_visible=True,
    )
    env_config = dict(config.env_config)
    env_config[DESYNC_SEED_KEY] = context.seed
    env_config[DESYNC_ENVS_PER_RUNNER_KEY] = config.num_envs_per_env_runner
    return config.environment(
        CassandraDesyncedPreviousRewardObservationEnv,
        env_config=env_config,
    )


def run(context: RunContext):
    return run_recipe(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
        previous_reward_visible=True,
        config_builder=build_config,
        recipe_metadata={
            "hypothesis": HYPOTHESIS,
            "primary_comparison": PRIMARY_COMPARISON,
            "episode_desync": {
                "enabled": True,
                "mode": "deterministic_one_time_initial_horizon",
                "subsequent_episode_length": 1_000,
            },
        },
    )


__all__ = [
    "ACTION_SCOPE",
    "CONDITION",
    "CassandraDesyncedPreviousRewardObservationEnv",
    "build_config",
    "initial_episode_horizon",
    "run",
]
