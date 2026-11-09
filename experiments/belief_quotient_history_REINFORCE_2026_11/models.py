import math

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from learners.models import (
    GRUModel,
    GRUModelConfig,
    ResidualMLPModel,
    ResidualMLPModelConfig,
    TransformerModel,
    TransformerModelConfig,
)


class ZeroBaselineTemperature:
    def setup(self):
        super().setup()
        self.sampling_temperature = float(
            self.model_config.get("sampling_temperature", 1.0)
        )
        if not math.isfinite(self.sampling_temperature) or self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")

    def compute_values(self, batch, embeddings=None):
        reference = embeddings if embeddings is not None else batch[Columns.OBS]
        return torch.zeros(
            reference.shape[:-1], dtype=reference.dtype, device=reference.device
        )

    def action_distribution_inputs(self, embeddings):
        return super().action_distribution_inputs(embeddings) / self.sampling_temperature

    def _outputs(self, embeddings, state_out, *, training):
        outputs = super()._outputs(embeddings, state_out, training=training)
        outputs[Columns.ACTION_DIST_INPUTS] = self.action_distribution_inputs(embeddings)
        return outputs


class ReinforceTransformer(ZeroBaselineTemperature, TransformerModel):
    pass


class ReinforceGRU(ZeroBaselineTemperature, GRUModel):
    pass


class ReinforceMemoryless(ZeroBaselineTemperature, ResidualMLPModel):
    pass


def model_spec(architecture, *, ladder):
    if architecture == "transformer":
        config = TransformerModelConfig(
            d_model=64, n_layers=4, n_heads=1, context_len=32 if ladder else 10
        ).to_dict()
        if ladder:
            config["grad_checkpointing"] = True
        cls = ReinforceTransformer
    elif architecture == "memoryless":
        config = ResidualMLPModelConfig(
            d_model=64,
            n_layers=6,
            obs_feature_indices=tuple(range(3 if ladder else 9)),
        ).to_dict()
        cls = ReinforceMemoryless
    elif architecture in ("gru128", "gru92"):
        config = GRUModelConfig(
            hidden_dim=128 if architecture == "gru128" else 92,
            n_layers=2 if architecture == "gru128" else 4,
            max_seq_len=32,
        ).to_dict()
        cls = ReinforceGRU
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    return RLModuleSpec(
        module_class=cls, model_config={**config, "sampling_temperature": 1.0}
    )
