from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from analysis.rollouts import PolicyRandomness
from envs.hmm import product_distribution
from envs.wing.model import wing_model
from experiments.wing_two_factor_explore_cycle_1 import analysis
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
from harness.seeding import named_seed_sequences


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _module():
    return WingActorCritic(
        observation_space=gym.spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
        action_space=gym.spaces.Discrete(9),
        model_config={
            "d_model": 64,
            "n_layers": 3,
            "n_heads": 1,
            "d_mlp": 256,
            "context_length": 32,
            "sampling_temperature": 1.5,
            "positional_embedding": "rope",
        },
    )


def _dataset(seed, count=96):
    rng = np.random.default_rng(seed)
    factors = rng.dirichlet(np.ones(3), size=(count, 2))
    hidden = factors.reshape(count, 6)
    tokens = rng.integers(4, size=count)
    actions = rng.integers(9, size=count)
    return analysis.ProbeData(
        activations=np.stack([hidden, hidden * 2, hidden * 3], axis=1),
        joint_beliefs=product_distribution([factors[:, 0], factors[:, 1]]),
        factor_beliefs=factors,
        observations=np.concatenate([np.eye(4)[tokens], np.eye(3)[actions // 3], np.eye(3)[actions % 3]], axis=1),
        states=rng.integers(9, size=count),
        actions=actions,
        rewards=rng.random(count),
        episode_ids=np.repeat(np.arange(8), count // 8),
        episode_steps=np.tile(np.arange(32, 32 + count // 8), 8),
        product_consistency_max_abs=0.0,
    )


def test_seeded_sampling_uses_scaled_logits_without_rescaling():
    logits = torch.tensor([[0.0, 1.5]]).repeat(20_000, 1)

    def sample(seed):
        sequence = np.random.SeedSequence(seed)
        randomness = PolicyRandomness(sequence, np.random.default_rng(sequence))
        return analysis._sample_actions(logits, randomness)

    first = sample(9)
    np.testing.assert_array_equal(first, sample(9))
    assert first.mean() == pytest.approx(float(torch.sigmoid(torch.tensor(1.5))), abs=0.01)
    assert not np.array_equal(first, sample(10))
    assert set(first) == {0, 1}


def test_layerwise_report_and_controls_use_held_out_samples(tmp_path):
    context = SimpleNamespace(seed=42, smoke=True, results_dir=tmp_path)
    train, test = _dataset(10), _dataset(20)
    report = analysis._analyze_samples(
        context,
        condition="reward_both",
        reward_state=0,
        checkpoint_label="initialization",
        agent_steps=0,
        training_iteration=0,
        train=train,
        test=test,
        streams=named_seed_sequences(42, analysis._STREAM_KEYS),
        emission_matrix=wing_model().emission_matrix,
    )
    assert report["n_fit"] == report["n_test"] == 96
    for layer in report["layers"].values():
        for factor in ("factor_1", "factor_2"):
            assert layer["probe_fits"][factor]["mse"] < 1e-20
            assert layer["probe_fits"][factor]["delta"]["mse"] < 1e-20
            assert layer["probe_fits"][factor]["normalized_mse"] < 1e-20
            assert layer["shuffled_training_labels"][factor]["r_squared"] < 0.3
    for factor in ("factor_1", "factor_2"):
        assert report["controls"]["ntp"][factor]["delta"]["r_squared"] < 0.3
        assert report["controls"]["log_ntp"][factor]["delta"]["r_squared"] < 0.3
        assert sum(report["policy"]["factor_state_fractions"][factor]) == pytest.approx(1)
    assert sum(report["policy"]["action_fractions"]) == pytest.approx(1)
    assert json.loads((tmp_path / "probe_battery.json").read_text())["metadata"]["temperature"] == 1.5


def test_branch_baseline_uses_training_means_and_unseen_fallback():
    train, test = _dataset(1), _dataset(2)
    observed = np.zeros_like(train.observations)
    observed[:, 0] = observed[:, 4] = observed[:, 7] = 1
    train = replace(train, observations=observed)
    expected = np.broadcast_to(train.factor_beliefs[:, 0].mean(axis=0), (96, 3))
    np.testing.assert_allclose(analysis._branch_prediction(train, test, 0), expected)
    changed = replace(test, factor_beliefs=np.zeros_like(test.factor_beliefs))
    np.testing.assert_array_equal(
        analysis._branch_prediction(train, test, 0),
        analysis._branch_prediction(train, changed, 0),
    )


def test_stateful_collection_resets_hooks_and_never_conditions_on_rewards(monkeypatch, single_thread):
    original_config = analysis.environment_config

    def short_config(condition, reward_state):
        config = original_config(condition, reward_state)
        config["episode_length"] = 40
        config["randomize_first_episode_length"] = False
        return config

    monkeypatch.setattr(analysis, "environment_config", short_config)
    module = _module()

    def collect(reward_state):
        return analysis.collect_probe_data(
            module,
            condition="reward_both",
            reward_state=reward_state,
            n_steps=256,
            seed=np.random.SeedSequence(71),
            device=torch.device("cpu"),
        )

    first, repeated, different_rewards = collect(0), collect(0), collect(2)
    assert first.activations.shape == (256, 3, 64)
    assert first.factor_beliefs.shape == (256, 2, 3)
    assert first.episode_steps.min() == 32
    assert first.episode_steps.max() == 39
    assert len(np.unique(first.episode_ids)) == 32
    assert first.product_consistency_max_abs < 1e-10
    assert len(np.unique(first.actions)) > 1
    for other in (repeated, different_rewards):
        np.testing.assert_array_equal(first.actions, other.actions)
        np.testing.assert_array_equal(first.activations, other.activations)
        np.testing.assert_array_equal(first.factor_beliefs, other.factor_beliefs)
    assert not np.array_equal(first.rewards, different_rewards.rewards)
    assert all(not block._forward_hooks for block in module.encoder.blocks)


def test_checkpoint_analysis_smoke_runs_complete_battery(tmp_path, monkeypatch, single_thread):
    module = _module()
    monkeypatch.setattr(
        analysis, "load_algorithm",
        lambda checkpoint: nullcontext(SimpleNamespace(get_module=lambda: module)),
    )
    context = SimpleNamespace(seed=42, smoke=True, results_dir=tmp_path, hardware=None)
    report = analysis.analyze_checkpoint(
        context,
        checkpoint=tmp_path / "initial_checkpoint",
        condition="reward_factor_1",
        reward_state=1,
        checkpoint_label="initialization",
        agent_steps=0,
        training_iteration=0,
    )
    assert report["n_fit"] == report["n_test"] == 256
    assert report["product_consistency_max_abs"] < 1e-10
    assert set(report["layers"]) == {"layer_1", "layer_2", "layer_3"}
    np.testing.assert_allclose(
        report["metadata"]["ntp_emission_matrix"],
        [[0.03, 0.97], [0.391, 0.609], [0.03, 0.97]],
    )
    seeds = report["metadata"]["rollout_seed_spawn_keys"]
    assert seeds["probe_train"] != seeds["probe_test"]
    assert report["controls"]["current_token_preceding_action"]["factor_1"]["mse"] >= 0


def test_collection_removes_hooks_on_failure(monkeypatch, single_thread):
    module = _module()

    def fail(*args, **kwargs):
        raise RuntimeError("injected collection failure")

    monkeypatch.setattr(analysis, "collect_batched_rollout_data", fail)
    with pytest.raises(RuntimeError, match="injected collection failure"):
        analysis.collect_probe_data(
            module,
            condition="reward_both",
            reward_state=0,
            n_steps=256,
            seed=np.random.SeedSequence(1),
            device=torch.device("cpu"),
        )
    assert all(not block._forward_hooks for block in module.encoder.blocks)
