from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from envs.gol.model import AGGREGATION, controlled_kernels, gol_model
from envs.gol.tasks.reward_state import GolRewardTask
from experiments.gol_reward_state_action_symmetry_cycle_1 import analysis
from harness.seeding import named_seed_sequences
from learners.models.transformer import TransformerModel
from ray.rllib.core.columns import Columns


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def module():
    with torch.random.fork_rng():
        torch.manual_seed(81)
        return TransformerModel(
            observation_space=gym.spaces.Box(0.0, 1.0, (6,), dtype=np.float32),
            action_space=gym.spaces.Discrete(4),
            model_config={"d_model": 8, "n_layers": 1, "n_heads": 1, "context_len": 8, "max_seq_len": 4},
        ).eval()


@pytest.mark.parametrize("variant", [2, 3])
@pytest.mark.parametrize("speed", ["half", "quarter"])
def test_targets_align_with_decision_embedding_and_independent_filter(module, variant, speed):
    data = analysis.collect_probe_data(module, variant=variant, speed=speed, n_steps=80, seed=13, warmup=0, n_envs=4)
    kernels = controlled_kernels(variant, speed)
    initial = gol_model(variant, speed).initial_distribution
    beliefs = np.tile(initial, (4, 1))
    previous_actions = np.full(4, -1)
    for row, env_index in enumerate(data.env_indices):
        observation = data.observations[row]
        if data.episode_steps[row] == 0:
            np.testing.assert_array_equal(observation, np.zeros(6))
        else:
            action, token = int(observation[2:].argmax()), int(observation[:2].argmax())
            assert action == previous_actions[env_index]
            updated = np.array([
                sum(beliefs[env_index, source] * kernels[action, token, source, destination] for source in range(4))
                for destination in range(4)
            ])
            beliefs[env_index] = updated / updated.sum()
        np.testing.assert_allclose(data.beliefs[row], beliefs[env_index], atol=1e-13)
        previous_actions[env_index] = data.actions[row]
    assert data.activations.shape == (80, 8)
    assert data.beliefs.dtype == np.float64
    replay = analysis.replay_representations(module, data.history_observations)
    np.testing.assert_array_equal(replay[data.sample_indices], data.activations)
    targets = analysis.belief_targets(data.beliefs)
    assert {key: value.shape for key, value in targets.items()} == {
        "fine_belief": (80, 4), "coarse_belief": (80, 3), "m1_minus_m2": (80, 1), "e_minus_s": (80, 1),
    }
    np.testing.assert_allclose(targets["coarse_belief"], data.beliefs @ AGGREGATION)
    np.testing.assert_allclose(targets["m1_minus_m2"][:, 0], data.beliefs[:, 0] - data.beliefs[:, 1])
    np.testing.assert_allclose(targets["e_minus_s"][:, 0], data.beliefs[:, 2] - data.beliefs[:, 3])


def test_encode_step_is_actual_policy_embedding(module):
    observations = torch.tensor([[0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 1., 0.]])
    state = analysis._initial_state(module, 2)
    with torch.no_grad():
        embeddings, _ = module.encode_step(observations, state)
        forward = module.forward_inference({Columns.OBS: observations[:, None], Columns.STATE_IN: state})
        torch.testing.assert_close(module.action_distribution_inputs(embeddings), forward[Columns.ACTION_DIST_INPUTS][:, 0])


def test_rewards_do_not_change_actor_inputs_actions_or_filter(module, monkeypatch):
    kwargs = dict(variant=3, n_steps=128, seed=112, warmup=4, n_envs=4)
    original = analysis.collect_probe_data(module, **kwargs)
    reward = GolRewardTask.reward

    def changed_reward(self, event, decision):
        value, components = reward(self, event, decision)
        return 1000.0 - value, components

    monkeypatch.setattr(GolRewardTask, "reward", changed_reward)
    changed = analysis.collect_probe_data(module, **kwargs)
    assert not np.array_equal(original.rewards, changed.rewards)
    for field in ("actions", "observations", "activations", "beliefs", "history_observations"):
        np.testing.assert_array_equal(getattr(original, field), getattr(changed, field))
    assert np.any((original.beliefs[:, 2] > 0) & (original.beliefs[:, 2] < 1))


def test_reproducible_but_independent_fit_test_histories(module):
    streams = named_seed_sequences(44, analysis.STREAM_KEYS)
    kwargs = dict(variant=2, n_steps=65, warmup=3, n_envs=4)
    fit = analysis.collect_probe_data(module, seed=streams["fit"], **kwargs)
    repeated = analysis.collect_probe_data(module, seed=streams["fit"], **kwargs)
    test = analysis.collect_probe_data(module, seed=streams["test"], **kwargs)
    np.testing.assert_array_equal(fit.history_observations, repeated.history_observations)
    np.testing.assert_array_equal(fit.activations, repeated.activations)
    assert not np.array_equal(fit.history_observations, test.history_observations)
    assert not np.array_equal(fit.beliefs, test.beliefs)
    assert fit.episode_steps.min() == 3
    assert fit.episode_steps.max() == 19
    assert fit.activations.shape == (65, 8)
    assert fit.history_observations.shape == (20, 4, 6)
    assert len(fit.sample_indices) == 65


def test_branch_baseline_is_joint_categorical_and_training_only():
    observations = np.array([
        np.r_[np.eye(2)[token], np.eye(4)[action]] for action in range(4) for token in range(2)
    ] + [np.zeros(6)])
    np.testing.assert_array_equal(analysis.branch_keys(observations), np.arange(9))
    train_target = np.array([[1., 0.], [0., 1.], [.2, .8]])
    predicted = analysis.branch_prediction(np.array([1, 1, 3]), np.array([1, 3, 8]), train_target)
    np.testing.assert_allclose(predicted, [[.5, .5], [.2, .8], [.4, .6]])
    with pytest.raises(ValueError, match="only two token"):
        analysis.branch_keys(np.zeros((4, 7)))
    with pytest.raises(ValueError, match="one token and one previous action"):
        analysis.branch_keys(np.array([[1, 0, 0, 0, 0, 0]]))


@pytest.mark.parametrize("variant", [2, 3])
def test_marginal_controls_predict_next_step_but_cannot_identify_e_minus_s(variant):
    kernels = controlled_kernels(variant, "quarter")
    beliefs = np.array([[.2, .3, .4, .1], [.2, .3, .1, .4]])
    features = analysis.marginal_features(beliefs, kernels)
    assert features["next_reward"].shape == (2, 4)
    assert features["next_token"].shape == (2, 8)
    assert features["combined_marginals"].shape == (2, 12)
    for feature in features.values():
        np.testing.assert_allclose(feature[0], feature[1], atol=1e-15)
    for action in range(4):
        expected_reward = sum(beliefs[0, i] * kernels[action, x, i, 2] for i in range(4) for x in range(2))
        assert features["next_reward"][0, action] == pytest.approx(expected_reward)
        for token in range(2):
            expected_token = sum(beliefs[0, i] * kernels[action, token, i, j] for i in range(4) for j in range(4))
            assert features["next_token"][0, 2 * action + token] == pytest.approx(expected_token)
    assert analysis.belief_targets(beliefs)["e_minus_s"][0] != analysis.belief_targets(beliefs)["e_minus_s"][1]


def test_history_shuffle_replays_pairs_and_retains_original_targets(module):
    data = analysis.collect_probe_data(module, variant=3, n_steps=80, seed=22, warmup=3, n_envs=4)
    before = data.beliefs.copy()
    shuffled = analysis.shuffled_histories(data.history_observations, 33)
    np.testing.assert_array_equal(shuffled[0], 0)
    for env_index in range(4):
        np.testing.assert_array_equal(
            np.sort(analysis.branch_keys(shuffled[:, env_index])),
            np.sort(analysis.branch_keys(data.history_observations[:, env_index])),
        )
    replayed = analysis.replay_representations(module, shuffled)[data.sample_indices]
    assert replayed.shape == data.activations.shape
    assert not np.array_equal(replayed, data.activations)
    np.testing.assert_array_equal(data.beliefs, before)
    np.testing.assert_array_equal(shuffled, analysis.shuffled_histories(data.history_observations, 33))


def test_affine_metrics_are_held_out_and_json_native(module):
    streams = named_seed_sequences(19, analysis.STREAM_KEYS)
    fit = analysis.collect_probe_data(module, variant=3, n_steps=80, seed=streams["fit"], warmup=4)
    test = analysis.collect_probe_data(module, variant=3, n_steps=80, seed=streams["test"], warmup=4)
    fit = replace(fit, activations=fit.beliefs)
    test = replace(test, activations=test.beliefs)
    report = analysis.analyze_samples(
        fit, test, shuffled_fit=np.zeros_like(fit.beliefs), shuffled_test=np.zeros_like(test.beliefs),
        kernels=controlled_kernels(3), streams=streams, smoke=True,
    )
    for metrics in report["post_final_layer_norm"].values():
        assert metrics["mse"] < 1e-10
        assert metrics["r_squared"] > .99999
        assert metrics["n_evaluated"] == 80
        assert metrics["target_variance"] > 0
        assert len(metrics["mse_environment_bootstrap_ci95"]) == 2
    metrics = analysis._metrics(np.zeros((5, 1)), np.zeros((5, 1)), np.zeros(5))
    converted = analysis._json_native(metrics)
    assert converted["r_squared"] is None
    assert converted["global_mse_ratio"] is None
    assert converted["fine_mse_ratio"] is None
    json.dumps(converted, allow_nan=False)


@pytest.mark.parametrize("algorithm_root", [False, True])
def test_public_checkpoint_smoke_pipeline(module, tmp_path, algorithm_root):
    checkpoint = tmp_path / "checkpoint"
    module_path = checkpoint / "learner_group" / "learner" / "rl_module" / "default_policy" if algorithm_root else checkpoint
    module.save_to_path(str(module_path))
    context = SimpleNamespace(seed=42, smoke=True, results_dir=tmp_path / "results")
    report = analysis.analyze_checkpoint(
        context, checkpoint=checkpoint, variant=2, speed="half", checkpoint_label="initialization",
    )
    assert report["n_fit"] == report["n_test"] == 256
    assert set(report["probe_fits"]) == {"fine_belief", "coarse_belief", "m1_minus_m2", "e_minus_s"}
    assert report["agent_steps"] == report["training_iteration"] == 0
    metadata = report["metadata"]
    assert metadata["rollout_seed_spawn_keys"]["fit"] != metadata["rollout_seed_spawn_keys"]["test"]
    assert metadata["representation"] == "post_final_layer_norm"
    assert metadata["device"] == "cpu"
    assert metadata["rewards_condition_actor_or_filter"] is False
    assert "ORIGINAL" in metadata["shuffled_history_control"]
    assert "not a causal" in metadata["shuffled_history_control"]
    assert "not an optimized long-run" in report["previous_action_latest_token_controller"]["method"]
    assert len(report["previous_action_latest_token_controller"]["action_table_by_branch"]) == 9
    assert 0 <= report["policy"]["expected_next_reward_mean"] <= 1
    assert sum(report["policy"]["action_fractions"]) == pytest.approx(1)
    assert set(report["controls"]) == {
        "shuffled_history_broken_alignment", "next_reward", "next_token", "combined_marginals",
        "categorical_plus_marginals", "previous_action_latest_token", "single_permuted_training_labels",
    }
    assert json.loads((context.results_dir / "probe_initialization.json").read_text()) == report
    json.dumps(report, allow_nan=False)
