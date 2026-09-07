from __future__ import annotations

import math

from ray.rllib.core.columns import Columns

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)


class WingActorCritic(FactoredReproductionActorCritic):
    def setup(self):
        super().setup()
        self.sampling_temperature = float(self.model_config["sampling_temperature"])
        if not math.isfinite(self.sampling_temperature) or self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be finite and positive")

    def action_distribution_inputs(self, embeddings):
        return super().action_distribution_inputs(embeddings) / self.sampling_temperature

    def _outputs(self, embeddings, state_out, *, training):
        outputs = super()._outputs(embeddings, state_out, training=training)
        outputs[Columns.ACTION_DIST_INPUTS] = self.action_distribution_inputs(embeddings)
        return outputs
