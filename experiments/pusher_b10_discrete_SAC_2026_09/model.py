"""Shared-trunk and fully split transformer modules for Pusher discrete SAC."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import MLPHeadConfig, ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.typing import ModuleID, ParamDict, TensorType

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
    ReproductionResidualEncoder,
)
from experiments.factored_representations_reproduction_SAC_2026_08.model import (
    FactoredReproductionSAC,
)
from experiments.pusher_b.process import CONTEXT_LENGTH, TOKEN_COUNT


@dataclass
class PusherSACEncoderConfig(ModelConfig):
    token_count: int = TOKEN_COUNT
    transformer: dict[str, object] = field(default_factory=dict)

    @property
    def output_dims(self) -> tuple[int, ...]:
        config = FactoredReproductionModelConfig.from_dict(self.transformer)
        return (config.d_model,)

    def build(self, framework: str) -> PusherSACEncoder:
        if framework != "torch":
            raise ValueError("the Pusher SAC study supports only PyTorch")
        return PusherSACEncoder(self)


class PusherSACEncoder(TorchModel, Encoder):
    """Encode one newest-first Pusher token-history window."""

    def __init__(self, config: PusherSACEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.reproduction_config = FactoredReproductionModelConfig.from_dict(
            config.transformer
        )
        if self.reproduction_config.context_length != CONTEXT_LENGTH:
            raise ValueError(
                "transformer context length must match Pusher history depth"
            )
        if config.token_count != TOKEN_COUNT:
            raise ValueError(f"Pusher frames must contain {TOKEN_COUNT} tokens")
        self.token_count = config.token_count
        self.encoder = ReproductionResidualEncoder(
            config.token_count,
            self.reproduction_config,
        )

    def encode_pre_final_norm(self, observations: torch.Tensor) -> torch.Tensor:
        expected_width = CONTEXT_LENGTH * self.token_count
        if observations.shape[-1] != expected_width:
            raise ValueError(
                f"expected observation width {expected_width}, "
                f"received {observations.shape[-1]}"
            )
        prefix = observations.shape[:-1]
        newest_first = observations.reshape(
            -1,
            CONTEXT_LENGTH,
            self.token_count,
        )
        oldest_first = newest_first.flip(dims=(-2,))
        visible_count = (
            oldest_first.abs().sum(dim=-1) > 0.5
        ).sum(dim=-1)
        residual = self.encoder(
            oldest_first[:, :-1, :],
            visible_count,
            oldest_first[:, -1:, :],
            apply_final_norm=False,
        )[:, 0, :]
        return residual.reshape(*prefix, self.reproduction_config.d_model)

    def _forward(
        self,
        inputs: dict[str, object],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        observations = inputs[Columns.OBS]
        if not isinstance(observations, torch.Tensor):
            raise TypeError("Pusher SAC observations must be torch tensors")
        residual = self.encode_pre_final_norm(observations.float())
        return {ENCODER_OUT: self.encoder.final_norm(residual)}


class PusherSACCatalog(SACCatalog):
    """Build Pusher transformer encoders and linear actor/critic heads."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict[str, object],
        view_requirements: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            model_config_dict,
            view_requirements,
        )
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("Pusher SAC expects a flat Box observation")
        if tuple(observation_space.shape) != (
            CONTEXT_LENGTH * TOKEN_COUNT,
        ):
            raise ValueError("Pusher SAC observation has the wrong history width")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("Pusher SAC requires a discrete action space")
        self.reproduction_config = FactoredReproductionModelConfig.from_dict(
            model_config_dict
        )
        self.latent_dims = (self.reproduction_config.d_model,)
        self.pi_and_qf_head_hiddens = []
        self.qf_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=[],
            output_layer_dim=action_space.n,
            output_layer_activation="linear",
        )

    def _make_encoder_config(self) -> PusherSACEncoderConfig:
        return PusherSACEncoderConfig(
            input_dims=tuple(self.observation_space.shape),
            token_count=TOKEN_COUNT,
            transformer=dict(self._model_config_dict),
        )

    def build_encoder(self, framework: str) -> Encoder:
        return self._make_encoder_config().build(framework)

    def build_qf_encoder(self, framework: str) -> Encoder:
        return self._make_encoder_config().build(framework)


class PusherSplitSAC(FactoredReproductionSAC):
    """Independent actor, critic, and twin-critic transformers."""


class PusherSharedTrunkSAC(FactoredReproductionSAC):
    """One transformer feeding separate actor and twin-critic heads."""

    def setup(self) -> None:
        self.twin_q = self.model_config["twin_q"]
        shared_encoder = self.catalog.build_encoder(framework=self.framework)
        self.pi_encoder = shared_encoder
        if not self.inference_only or self.framework != "torch":
            self.qf_encoder = shared_encoder
            if self.twin_q:
                self.qf_twin_encoder = shared_encoder

        self.pi = self.catalog.build_pi_head(framework=self.framework)
        if not self.inference_only or self.framework != "torch":
            self.qf = self.catalog.build_qf_head(framework=self.framework)
            if self.twin_q:
                self.qf_twin = self.catalog.build_qf_head(
                    framework=self.framework
                )


class SingleBackwardSACTorchLearner(SACTorchLearner):
    """Backpropagate the summed discrete-SAC loss once per minibatch."""

    def compute_gradients(
        self,
        loss_per_module: dict[ModuleID, TensorType],
        **kwargs: object,
    ) -> ParamDict:
        try:
            return TorchLearner.compute_gradients(
                self,
                loss_per_module,
                **kwargs,
            )
        finally:
            self._temp_losses.clear()


class SharedTrunkSACTorchLearner(SingleBackwardSACTorchLearner):
    """Optimize a shared encoder once using actor and both critic losses."""

    def configure_optimizers_for_module(
        self,
        module_id: ModuleID,
        config: AlgorithmConfig | None = None,
    ) -> None:
        if config is None:
            raise ValueError("shared-trunk SAC requires an algorithm config")
        module = self._module[module_id]
        shared_lr = float(
            config.learner_config_dict["shared_encoder_learning_rate"]
        )
        optimizer_specs = (
            (
                "shared",
                self.get_parameters(module.pi_encoder),
                shared_lr,
            ),
            (
                "policy",
                self.get_parameters(module.pi),
                config.actor_lr,
            ),
            (
                "qf",
                self.get_parameters(module.qf),
                config.critic_lr,
            ),
            (
                "qf_twin",
                self.get_parameters(module.qf_twin),
                config.critic_lr,
            ),
        )
        for name, parameters, learning_rate in optimizer_specs:
            optimizer = torch.optim.Adam(parameters, eps=1e-7)
            self.register_optimizer(
                module_id=module_id,
                optimizer_name=name,
                optimizer=optimizer,
                params=parameters,
                lr_or_lr_schedule=learning_rate,
            )

        temperature = self.curr_log_alpha[module_id]
        alpha_optimizer = torch.optim.Adam([temperature], eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="alpha",
            optimizer=alpha_optimizer,
            params=[temperature],
            lr_or_lr_schedule=config.alpha_lr,
        )

def trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
