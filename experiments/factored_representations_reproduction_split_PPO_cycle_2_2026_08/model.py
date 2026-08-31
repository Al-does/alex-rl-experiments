"""PPO module with independent actor and critic transformer parameters."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
    FactoredReproductionModelConfig,
    ReproductionResidualEncoder,
)
from learners.models.next_token import NextTokenAuxHead


class SplitFactoredReproductionActorCritic(FactoredReproductionActorCritic):
    """Paper transformers for PPO policy and value with no shared parameters."""

    def _build_encoder(self) -> int:
        self.reproduction_config = FactoredReproductionModelConfig.from_dict(
            dict(self.model_config)
        )
        self._obs_dim = int(np.prod(self.observation_space.shape))
        self.actor_encoder = ReproductionResidualEncoder(
            self._obs_dim,
            self.reproduction_config,
        )
        self.critic_encoder = ReproductionResidualEncoder(
            self._obs_dim,
            self.reproduction_config,
        )
        return self.reproduction_config.d_model

    @property
    def encoder(self) -> ReproductionResidualEncoder:
        """Expose only the actor transformer to the shared probe battery."""

        return self.actor_encoder

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> torch.Tensor:
        # PPO supplies actor embeddings here as an optimization. Ignoring them is
        # essential: the value loss must traverse only the independent critic.
        del embeddings
        observations = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        critic_embeddings = self.critic_encoder(
            state["ctx"],
            state["len"].reshape(-1),
            observations,
            apply_final_norm=True,
        )
        return self.heads.values(critic_embeddings)


class SplitActorCriticWithNextJointTokenAux(
    NextTokenAuxHead,
    SplitFactoredReproductionActorCritic,
):
    """Split PPO module with an actor-only next-token auxiliary head."""

