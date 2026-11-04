import numpy as np
import pytest
import torch
from torch import nn

from experiments.mess3_reward_state_action_symmetry_cycle_7 import ab_on_policy_tv as adapter


def test_inverse_cdf_sampling_preserves_zero_probability_actions_and_boundaries():
    p = np.tile([0.25, 0.0, 0.75], (5, 1))
    uniforms = np.array([0.0, 0.2499, 0.25, 0.9, 0.99999])
    np.testing.assert_array_equal(adapter.choose_actions(p, uniforms, "sampled"), [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(adapter.choose_actions(p, uniforms, "greedy"), [2] * 5)
    with pytest.raises(ValueError):
        adapter.choose_actions(p, uniforms, "unknown")


class Policy(nn.Module):
    def get_initial_state(self):
        return {"steps": np.zeros(1, dtype=np.float32)}

    def encode_step(self, obs, state):
        scores = 2 * obs[:, :3] + 0.1 * obs[:, 3:] + 0.01 * state["steps"]
        return scores, {"steps": state["steps"] + 1}

    def action_distribution_inputs(self, hidden):
        return hidden


@pytest.mark.parametrize("mode", ["greedy", "sampled"])
def test_collector_executes_original_actions_and_preserves_feedback_for_shadow(monkeypatch, mode):
    instances = []

    class Environment:
        def __init__(self, config):
            self.actions = []
            instances.append(self)

        def reset(self, seed):
            self.step_index = 0
            return np.array([1, 0, 0, 0, 0, 0], dtype=np.float32), {}

        def step(self, action):
            self.actions.append(action)
            self.step_index += 1
            obs = np.concatenate((
                np.eye(3)[self.step_index % 3], np.eye(3)[action],
            )).astype(np.float32)
            return obs, 0, False, False, {}

        def close(self):
            pass

    monkeypatch.setattr(adapter, "HMMEnv", Environment)
    seed, horizon = 123, 8
    data = adapter.collect(Policy(), mode=mode, episodes=3, horizon=horizon, seed=seed, batch_size=2)
    for episode, env in enumerate(instances):
        selected = {key: value[data["episode"] == episode] for key, value in data.items()}
        np.testing.assert_array_equal(env.actions, selected["action"][:-1])
        np.testing.assert_array_equal(selected["previous_action"][1:], selected["action"][:-1])
        assert selected["previous_action"][0] == -1
        uniforms = np.random.default_rng(np.random.SeedSequence([seed, episode, 2])).random(horizon)
        expected = adapter.choose_actions(adapter.softmax(selected["logits"]), uniforms, mode)
        np.testing.assert_array_equal(selected["action"], expected)
        previous = np.zeros((horizon, 3))
        previous[1:] = np.eye(3)[selected["action"][:-1]]
        shadow = (
            2 * np.eye(3)[selected["randomized_token"]]
            + 0.1 * previous + 0.01 * np.arange(horizon)[:, None]
        )
        np.testing.assert_allclose(selected["shadow_logits"], shadow, atol=3e-7)
        np.testing.assert_array_equal(
            selected["randomized_token"][selected["token"] == 2], 2,
        )
