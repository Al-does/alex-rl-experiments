"""Wing actor-critic with a memory-bounded value pass for the 256-wide encoder.

RLlib's GAE connector calls ``compute_values`` on the entire sampled batch with
autograd enabled. The windowed encoder materializes ``[rows, context, d_mlp]``
activations, which at ``d_model=256`` exceeds a single 24 GiB GPU. The value
pass needs no gradients, so it runs in fixed-size chunks under ``no_grad``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from ray.rllib.core.columns import Columns

from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic

VALUE_CHUNK_ROWS = 512


class WideWingActorCritic(WingActorCritic):
    def compute_values(
        self, batch: dict[str, Any], embeddings: Optional[Any] = None
    ):
        if embeddings is not None:
            return self.heads.values(embeddings)
        observations = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        rows = observations.shape[0]
        values = []
        with torch.no_grad():
            for start in range(0, rows, VALUE_CHUNK_ROWS):
                stop = min(start + VALUE_CHUNK_ROWS, rows)
                chunk = {
                    Columns.OBS: observations[start:stop],
                    Columns.STATE_IN: {
                        key: value[start:stop] for key, value in state.items()
                    },
                }
                chunk_embeddings, _ = self._encode_train(chunk)
                values.append(self.heads.values(chunk_embeddings))
        return torch.cat(values, dim=0)
