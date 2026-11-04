import numpy as np
import torch
from torch import nn

from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_checkpoint_tv import (
    replay_pair,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_pattern_checkpoint import (
    observations,
)


class AccumulatingPolicy(nn.Module):
    def get_initial_state(self):
        return {"history": np.zeros(3, dtype=np.float32)}

    def encode_step(self, obs, state):
        weights = torch.arange(18, dtype=obs.dtype).reshape(6, 3) / 10
        history = 0.5 * state["history"] + obs @ weights
        return history, {"history": history}

    def action_distribution_inputs(self, hidden):
        return hidden


def test_replay_keeps_stream_histories_and_partial_batch_episodes_separate():
    rng = np.random.default_rng(42)
    tokens = rng.integers(3, size=(17, 5))
    randomized = np.where(tokens == 2, 2, 1 - tokens)
    actions = rng.integers(3, size=tokens.shape)
    actions[:, 0] = -1
    grid = {
        "token": tokens, "randomized_token": randomized,
        "previous_action": actions,
    }
    actual = replay_pair(AccumulatingPolicy(), grid)
    weights = np.arange(18, dtype=np.float32).reshape(6, 3) / 10
    for stream, key in enumerate(("token", "randomized_token")):
        obs = observations(grid, key)
        for episode in range(17):
            history = np.zeros(3, dtype=np.float32)
            for step in range(5):
                history = 0.5 * history + obs[episode, step] @ weights
                np.testing.assert_allclose(
                    actual[episode, step, stream], history, rtol=0, atol=1e-6,
                )
