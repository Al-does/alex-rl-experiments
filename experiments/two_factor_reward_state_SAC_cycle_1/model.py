"""Action-aware split transformer actor and critics for discrete SAC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.torch.base import TorchModel

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
    ReproductionResidualEncoder,
)
from experiments.factored_representations_reproduction_SAC_2026_08.model import (
    FactoredReproductionSAC,
    ReproductionSACCatalog,
    ReproductionSACEncoder,
    ReproductionSACEncoderConfig,
)
from experiments.two_factor_reward_state_SAC_cycle_1.process import (
    CONTEXT_LENGTH,
    JOINT_TOKEN_COUNT,
)
from experiments.two_factor_reward_state_SAC_cycle_1.task import N_ACTIONS


FRAME_WIDTH = JOINT_TOKEN_COUNT + N_ACTIONS
FLAT_OBSERVATION_WIDTH = CONTEXT_LENGTH * FRAME_WIDTH


@dataclass
class TwoFactorSACEncoderConfig(ReproductionSACEncoderConfig):
    """Build an encoder that interleaves generic token/action history fields."""

    def build(self, framework: str) -> TwoFactorSACEncoder:
        if framework != "torch":
            raise ValueError("the two-factor SAC study supports only PyTorch")
        return TwoFactorSACEncoder(self)


class TwoFactorSACEncoder(ReproductionSACEncoder):
    """Encode aligned joint-token and preceding-action history frames."""

    def __init__(self, config: TwoFactorSACEncoderConfig) -> None:
        if config.token_count != FRAME_WIDTH:
            raise ValueError(f"frame width must be {FRAME_WIDTH}")
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.reproduction_config = FactoredReproductionModelConfig.from_dict(
            config.transformer
        )
        if self.reproduction_config.context_length != CONTEXT_LENGTH:
            raise ValueError(
                "transformer context length must match the observation history"
            )
        self.token_count = config.token_count
        self.encoder = ReproductionResidualEncoder(
            config.token_count,
            self.reproduction_config,
        )

    @staticmethod
    def _frames(observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != FLAT_OBSERVATION_WIDTH:
            raise ValueError(
                f"expected observation width {FLAT_OBSERVATION_WIDTH}, "
                f"received {observations.shape[-1]}"
            )
        flat = observations.reshape(-1, FLAT_OBSERVATION_WIDTH)
        token_end = CONTEXT_LENGTH * JOINT_TOKEN_COUNT
        tokens = flat[:, :token_end].reshape(
            -1,
            CONTEXT_LENGTH,
            JOINT_TOKEN_COUNT,
        )
        actions = flat[:, token_end:].reshape(-1, CONTEXT_LENGTH, N_ACTIONS)
        return torch.cat((tokens, actions), dim=-1)

    def _pre_final_norm(self, observations: torch.Tensor) -> torch.Tensor:
        prefix = observations.shape[:-1]
        newest_first = self._frames(observations)
        oldest_first = newest_first.flip(dims=(-2,))
        visible_count = (
            oldest_first[..., :JOINT_TOKEN_COUNT].abs().sum(dim=-1) > 0.5
        ).sum(dim=-1)
        residual = self.encoder(
            oldest_first[:, :-1, :],
            visible_count,
            oldest_first[:, -1:, :],
            apply_final_norm=False,
        )[:, 0, :]
        return residual.reshape(*prefix, self.reproduction_config.d_model)

    def _forward(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        residual = self._pre_final_norm(inputs[Columns.OBS].float())
        return {ENCODER_OUT: self.encoder.final_norm(residual)}


class TwoFactorSACCatalog(ReproductionSACCatalog):
    """Build independent action-aware transformers for actor and critics."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict[str, Any],
        view_requirements: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            model_config_dict,
            view_requirements,
        )
        if tuple(observation_space.shape) != (FLAT_OBSERVATION_WIDTH,):
            raise ValueError(
                "two-factor SAC observation does not match its history contract"
            )
        # The parent catalog targets the nine-position reproduction study.
        # This study has 64 aligned token/action frames instead.
        self.token_count = FRAME_WIDTH

    def _make_encoder_config(self, *, actor: bool) -> TwoFactorSACEncoderConfig:
        return TwoFactorSACEncoderConfig(
            input_dims=tuple(self.observation_space.shape),
            token_count=self.token_count,
            transformer=dict(self._model_config_dict),
            actor=actor,
            auxiliary_classes=None,
        )


class TwoFactorRewardSAC(FactoredReproductionSAC):
    """Discrete SAC with fully separate actor, critic, and twin critic."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("catalog_class", TwoFactorSACCatalog)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def actor_hidden(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the actor residual used by the offline probe pipeline."""

        return self.pi_encoder.encode_pre_final_norm(observations)
