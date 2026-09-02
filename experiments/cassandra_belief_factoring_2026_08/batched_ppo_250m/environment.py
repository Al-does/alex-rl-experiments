"""GPU-friendly batched Cassandra dynamics for the fully observable study."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.nn import functional as F

from envs.cassandra_machine.model import (
    GLOBAL_ALIAS_ACTION_COST,
    GLOBAL_ALIAS_COMPONENT_TRANSITIONS,
    N_COMPONENTS,
    N_CONDITIONS,
    OPERATE_COMPONENT_REWARD,
    TARGETED_ACTION_COST,
    TARGETED_COMPONENT_TRANSITIONS,
)


ActionScope = Literal["global_aliases", "targeted"]
OBSERVATION_DIM = N_COMPONENTS * N_CONDITIONS + 1


class BatchedCassandraEnv:
    """Simulate synchronized Cassandra episodes directly on one torch device.

    The original fully observable recipe does not consume Cassandra's noisy
    observation symbol or Bayesian belief. This implementation therefore
    samples only the state transition that can affect future rewards and emits
    the same four component one-hots plus preceding reward.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        action_scope: ActionScope,
        episode_length: int,
        seed: int,
        device: torch.device,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if action_scope not in {"global_aliases", "targeted"}:
            raise ValueError(
                "action_scope must be 'global_aliases' or 'targeted'"
            )

        self.num_envs = num_envs
        self.action_scope = action_scope
        self.episode_length = episode_length
        self.device = device
        transitions = (
            GLOBAL_ALIAS_COMPONENT_TRANSITIONS
            if action_scope == "global_aliases"
            else TARGETED_COMPONENT_TRANSITIONS
        )
        costs = (
            GLOBAL_ALIAS_ACTION_COST
            if action_scope == "global_aliases"
            else TARGETED_ACTION_COST
        )
        self.transitions = torch.tensor(
            transitions, dtype=torch.float32, device=device
        )
        self.action_costs = torch.tensor(
            costs, dtype=torch.float32, device=device
        )
        self.operate_rewards = torch.tensor(
            OPERATE_COMPONENT_REWARD,
            dtype=torch.float32,
            device=device,
        )
        self.component_indices = torch.arange(
            N_COMPONENTS, device=device
        ).expand(num_envs, -1)
        self._transition_generator = torch.Generator(device=device)
        self._transition_generator.manual_seed(seed)
        self._initial_generator = torch.Generator(device=device)
        self._initial_generator.manual_seed(seed + 1_000_003)
        self.components = torch.empty(
            (num_envs, N_COMPONENTS), dtype=torch.long, device=device
        )
        self.previous_rewards = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.episode_returns = torch.zeros_like(self.previous_rewards)
        self.episode_step = 0

    @property
    def action_count(self) -> int:
        return int(self.action_costs.shape[0])

    def _observation(self) -> torch.Tensor:
        components = F.one_hot(
            self.components, num_classes=N_CONDITIONS
        ).reshape(self.num_envs, N_COMPONENTS * N_CONDITIONS)
        return torch.cat(
            [components.to(torch.float32), self.previous_rewards[:, None]],
            dim=-1,
        )

    def reset(self) -> torch.Tensor:
        self.components = torch.randint(
            N_CONDITIONS,
            (self.num_envs, N_COMPONENTS),
            generator=self._initial_generator,
            device=self.device,
        )
        self.previous_rewards.zero_()
        self.episode_returns.zero_()
        self.episode_step = 0
        return self._observation()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"actions must have shape ({self.num_envs},), "
                f"received {tuple(actions.shape)}"
            )

        operate_reward = self.operate_rewards[self.components].prod(dim=-1)
        rewards = torch.where(
            actions == 0,
            operate_reward,
            self.action_costs[actions],
        )
        rows = self.transitions[
            actions[:, None], self.component_indices, self.components
        ]
        cumulative = rows.cumsum(dim=-1)
        draws = torch.rand(
            (self.num_envs, N_COMPONENTS),
            generator=self._transition_generator,
            device=self.device,
        )
        self.components = (
            (draws[..., None] > cumulative).sum(dim=-1).clamp_max(
                N_CONDITIONS - 1
            )
        )
        self.previous_rewards = rewards
        self.episode_returns.add_(rewards)
        self.episode_step += 1
        truncated = self.episode_step >= self.episode_length
        return self._observation(), rewards, truncated

    def state_dict(self) -> dict[str, Any]:
        return {
            "components": self.components,
            "previous_rewards": self.previous_rewards,
            "episode_returns": self.episode_returns,
            "episode_step": self.episode_step,
            "transition_rng": self._transition_generator.get_state(),
            "initial_rng": self._initial_generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.components.copy_(state["components"].to(self.device))
        self.previous_rewards.copy_(state["previous_rewards"].to(self.device))
        self.episode_returns.copy_(state["episode_returns"].to(self.device))
        self.episode_step = int(state["episode_step"])
        self._transition_generator.set_state(state["transition_rng"].cpu())
        self._initial_generator.set_state(state["initial_rng"].cpu())


__all__ = [
    "ActionScope",
    "BatchedCassandraEnv",
    "OBSERVATION_DIM",
]
