from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv, factor_marginals
from envs.wing.model import controlled_kernels, wing_model
from experiments.factored_representations_reproduction_PPO_2026_08.probe import _initial_state
from experiments.wing_token_guess_cycle_1.process import environment_config as passive_config
from experiments.wing_two_factor_explore_cycle_1.control_data import (
    ControlData,
    candidate_parameters,
    collect_control_data,
    replay_activations,
    replay_beliefs,
    shuffled_history,
)
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
from experiments.wing_two_factor_explore_cycle_2.process import environment_config as controlled_config


DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(passive, *, length=9, randomized=True):
    config = passive_config() if passive else controlled_config("reward_both")
    config["episode_length"] = length
    config["randomize_first_episode_length"] = randomized
    return config


def _module(config):
    env = HMMEnv(config)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(9)
            return WingActorCritic(
                observation_space=env.observation_space,
                action_space=env.action_space,
                model_config={
                    "d_model": 12,
                    "n_layers": 2,
                    "n_heads": 1,
                    "d_mlp": 24,
                    "context_length": 8,
                    "sampling_temperature": 1.5,
                    "positional_embedding": "rope",
                },
            )
    finally:
        env.close()


def _forced_data(passive):
    config = _config(passive, length=11, randomized=False)
    config["diagnostics"] = {"belief": True, "tokens": True, "transitions": True}
    observations, beliefs, tokens, preceding, episodes, steps = [], [], [], [], [], []
    env = HMMEnv(config)
    try:
        for episode in range(4):
            observation, info = env.reset(seed=100 + episode)
            for step in range(11):
                observations.append(observation)
                beliefs.append(np.stack(factor_marginals(info["belief_current"], (3, 3))))
                visible = info["visible_token_current"]
                tokens.append((-1, -1) if visible is None else divmod(visible, 2))
                preceding.append((-1, -1) if step == 0 else divmod(info["executed_action"], 2 if passive else 3))
                episodes.append(episode)
                steps.append(step)
                action = (step * 5 + episode * 2) % env.action_space.n
                observation, _, _, _, info = env.step(action)
    finally:
        env.close()
    count = len(steps)
    return ControlData(
        activations=np.zeros((count, 2, 12), dtype=np.float32),
        beliefs=np.asarray(beliefs),
        policy_probabilities=np.full((count, 4 if passive else 9), 1 / (4 if passive else 9)),
        observations=np.asarray(observations),
        episode_ids=np.asarray(episodes),
        episode_steps=np.asarray(steps),
        mask=np.asarray(steps) >= 3,
        tokens=np.asarray(tokens),
        preceding_actions=np.asarray(preceding),
    )


@pytest.mark.parametrize("passive", [True, False])
def test_forced_action_replay_matches_public_exact_beliefs(passive):
    data = _forced_data(passive)
    parameters = {"alpha": 0.94, "x": 0.4, "strength": None if passive else 1.0}
    beliefs, ntp = replay_beliefs(data, **parameters)
    np.testing.assert_allclose(beliefs, data.beliefs, atol=2e-14, rtol=0)
    np.testing.assert_allclose(beliefs.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_allclose(ntp.sum(axis=-1), 1, atol=1e-14)
    for suffix in (11, 100):
        suffix_beliefs, suffix_ntp = replay_beliefs(data, **parameters, suffix_length=suffix)
        np.testing.assert_allclose(suffix_beliefs, beliefs, atol=2e-14, rtol=0)
        np.testing.assert_allclose(suffix_ntp, ntp, atol=2e-14, rtol=0)
    changed_mask = replace(data, mask=~data.mask)
    np.testing.assert_array_equal(replay_beliefs(changed_mask, **parameters)[0], beliefs)
    action_perturbed = data.preceding_actions.copy()
    action_perturbed[data.episode_steps > 0] = (action_perturbed[data.episode_steps > 0] + 1) % 3
    perturbed = replay_beliefs(replace(data, preceding_actions=action_perturbed), **parameters)[0]
    if passive:
        np.testing.assert_array_equal(perturbed, beliefs)
    else:
        assert np.max(np.abs(perturbed - beliefs)) > 0.2


@pytest.mark.parametrize("passive", [True, False])
def test_suffix_prior_and_short_suffix_operator_timing(passive):
    data = _forced_data(passive)
    parameters = {"alpha": 0.94, "x": 0.4, "strength": None if passive else 1.0}
    model = wing_model()
    uniform = np.full((len(data.tokens), 2, 3), 1 / 3)
    beliefs, ntp = replay_beliefs(data, **parameters, suffix_length=0)
    np.testing.assert_allclose(beliefs, uniform @ model.transition_matrix if passive else uniform)
    np.testing.assert_allclose(ntp, uniform @ model.emission_matrix)
    kernels = controlled_kernels(strength=1.0)
    for suffix in (1, 3):
        beliefs, ntp = replay_beliefs(data, **parameters, suffix_length=suffix)
        for index in (0, 1, 5, 11, 21, 32):
            source = np.full((2, 3), 1 / 3)
            start = max(index - suffix + 1, index - int(data.episode_steps[index]))
            for member in range(start, index + 1):
                for factor in range(2):
                    token = data.tokens[member, factor]
                    if token < 0:
                        continue
                    edge = model.edge_transition_matrices[token] if passive or data.episode_steps[member] == 0 else kernels[data.preceding_actions[member, factor], token]
                    source[factor] = source[factor] @ edge
                    source[factor] /= source[factor].sum()
            np.testing.assert_allclose(beliefs[index], source @ model.transition_matrix if passive else source, atol=2e-14)
            np.testing.assert_allclose(ntp[index], source @ model.emission_matrix, atol=2e-14)


def test_passive_ntp_predicts_pending_token_not_one_transition_too_far():
    data = _forced_data(True)
    beliefs, ntp = replay_beliefs(data, alpha=0.94, x=0.4, strength=None)
    model = wing_model()
    np.testing.assert_allclose(ntp[0], np.tile(model.initial_distribution @ model.emission_matrix, (2, 1)))
    assert np.max(np.abs(ntp - beliefs @ model.emission_matrix)) > 0.005
    source = np.tile(model.initial_distribution, (2, 1))
    for index in range(1, 11):
        for factor in range(2):
            source[factor] = source[factor] @ model.edge_transition_matrices[data.tokens[index, factor]]
            source[factor] /= source[factor].sum()
        np.testing.assert_allclose(ntp[index], source @ model.emission_matrix, atol=2e-14)


@pytest.mark.parametrize("passive", [True, False])
@pytest.mark.parametrize("warmup", [0, 3, 8])
def test_collection_keeps_complete_histories_and_exact_scored_budget(passive, warmup):
    config = _config(passive)
    original_config = deepcopy(config)
    module = _module(config)
    module.train()
    hook_counts = [len(block._forward_hooks) for block in module.encoder.blocks]
    kwargs = dict(env_config=config, n_steps=19, seed=72, device=DEVICE, n_envs=3, warmup=warmup)
    first = collect_control_data(module, **kwargs)
    assert module.training
    assert [len(block._forward_hooks) for block in module.encoder.blocks] == hook_counts
    second = collect_control_data(module, **kwargs)
    for name in ("observations", "activations", "beliefs", "policy_probabilities", "episode_ids", "episode_steps", "mask", "tokens", "preceding_actions"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert config == original_config
    assert first.mask.sum() == 19
    np.testing.assert_array_equal(first.mask, first.episode_steps >= warmup)
    assert first.activations.shape[1:] == (2, 12)
    assert first.observations.shape[1:] == ((4,) if passive else (10,))
    np.testing.assert_allclose(first.policy_probabilities.sum(axis=-1), 1, atol=1e-14)
    for episode in np.unique(first.episode_ids):
        np.testing.assert_array_equal(first.episode_steps[first.episode_ids == episode], np.arange(np.sum(first.episode_ids == episode)))
    if warmup:
        assert not first.mask[first.episode_steps == 0].any()
    replayed, _ = replay_beliefs(first, alpha=0.94, x=0.4, strength=None if passive else 1.0)
    np.testing.assert_allclose(replayed, first.beliefs, atol=3e-14, rtol=0)
    activations, probabilities = replay_activations(module, first, device=DEVICE)
    np.testing.assert_allclose(activations, first.activations, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(probabilities, first.policy_probabilities, atol=2e-7, rtol=1e-5)
    assert module.training
    assert [len(block._forward_hooks) for block in module.encoder.blocks] == hook_counts
    with torch.inference_mode():
        module.eval()
        residual, _ = module.encode_step_pre_final_norm(torch.as_tensor(first.observations[:1]), _initial_state(module, 1, DEVICE))
        expected = torch.softmax(module.action_distribution_inputs(module.encoder.final_norm(residual)), dim=-1).numpy()
    np.testing.assert_allclose(first.policy_probabilities[:1], expected, atol=1e-7)


@pytest.mark.parametrize("passive", [True, False])
def test_collector_composes_with_belief_geometry_facade(passive):
    import json

    from analysis.belief_geometry import evaluate_belief_geometry
    from analysis.probes.controls import fit_grouped_affine

    config = _config(passive, length=9, randomized=False)
    module = _module(config)
    train, test = (
        collect_control_data(module, env_config=config, n_steps=budget, seed=seed,
                             device=DEVICE, n_envs=2, warmup=3)
        for budget, seed in ((24, 101), (18, 102))
    )
    assert not np.array_equal(train.observations[:len(test.observations)], test.observations)
    replayed = []
    for data, budget in ((train, 24), (test, 18)):
        assert data.mask.sum() == budget < len(data.mask)
        np.testing.assert_array_equal(data.mask, data.episode_steps >= 3)
        beliefs, pending = replay_beliefs(data, alpha=0.94, x=0.4, strength=None if passive else 1.0)
        np.testing.assert_allclose(beliefs, data.beliefs, atol=3e-14, rtol=0)
        replayed.append((beliefs[data.mask, 0], pending[data.mask, 0]))
    train_beliefs, train_pending = replayed[0]
    test_beliefs, test_pending = replayed[1]
    name = "last_layer_pre_final_norm"
    train_features = {name: train.activations[train.mask, -1]}
    test_features = {name: test.activations[test.mask, -1]}
    train_groups = np.array([f"train/{group}" for group in train.episode_ids[train.mask]])
    test_groups = np.array([f"test/{group}" for group in test.episode_ids[test.mask]])
    result = evaluate_belief_geometry(
        train_features, test_features, train_beliefs, test_beliefs,
        train_groups=train_groups, test_groups=test_groups,
        nuisance_features={"pending_token": (train_pending, test_pending)},
        contrasts={"state_0_minus_2": np.array([1.0, 0.0, -1.0]) / np.sqrt(2)},
        seed=42, n_null_repeats=1, n_resamples=3,
    )
    json.dumps(result.report, allow_nan=False)
    assert set(result.predictions) == {name}
    assert (result.report["n_train"], result.report["n_test"], result.report["n_states"]) == (24, 18, 3)
    assert result.report["n_train_groups"] == len(np.unique(train_groups)) == 4
    assert result.report["n_test_groups"] == len(np.unique(test_groups)) == 4
    assert result.report["n_null_repeats"] == 1 and result.report["n_resamples"] == 3
    record = result.report["representations"][name]
    assert record["fit"]["fit_source"] == "train"
    assert sorted(group for fold in record["fit"]["fold_validation_groups"] for group in fold) == sorted(np.unique(train_groups))
    weight, bias, fit = fit_grouped_affine(train_features[name], train.beliefs[train.mask, 0], train_groups, seed=42)
    np.testing.assert_allclose(result.predictions[name], test_features[name] @ weight + bias, atol=1e-10)
    assert record["fit"]["method"] == fit["method"] == "grouped_svd_cutoff_cv"
    assert result.predictions[name].shape == (18, 3)
    assert np.isfinite(result.predictions[name]).all()
    np.testing.assert_allclose(result.predictions[name].sum(axis=1), 1, atol=1e-10)
    assert record["metrics"]["mse"] == pytest.approx(np.square(result.predictions[name] - test.beliefs[test.mask, 0]).mean())
    np.testing.assert_allclose(result.baseline_predictions["train_mean"], np.broadcast_to(train_beliefs.mean(axis=0), test_beliefs.shape))
    comparison = record["comparisons"]["baselines"]["nuisance/pending_token"]
    assert comparison["bootstrap_unit"] == "group" and comparison["n_groups"] == 4
    assert comparison["bootstrap_refit"] is False
    assert set(record["nulls"]) == {"permuted_labels", "gaussian_features"}
    for repeats in record["nulls"].values():
        assert len(repeats) == 1
        assert result.baseline_predictions[repeats[0]["prediction_key"]].shape == test_beliefs.shape


@pytest.mark.parametrize("passive", [True, False])
def test_shuffling_preserves_joint_rows_resets_and_replays_observations(passive):
    config = _config(passive, length=13, randomized=False)
    module = _module(config)
    data = collect_control_data(module, env_config=config, n_steps=26, seed=77, device=DEVICE, n_envs=2, warmup=2)
    shuffled = shuffled_history(data, seed=10)
    again = shuffled_history(data, seed=10)
    reset_rows = data.episode_steps == 0
    for name in ("observations", "tokens", "preceding_actions"):
        np.testing.assert_array_equal(getattr(shuffled, name), getattr(again, name))
        np.testing.assert_array_equal(getattr(shuffled, name)[reset_rows], getattr(data, name)[reset_rows])
    for name in ("episode_ids", "episode_steps", "mask", "beliefs", "activations"):
        np.testing.assert_array_equal(getattr(shuffled, name), getattr(data, name))
    for episode in np.unique(data.episode_ids):
        members = data.episode_ids == episode
        original_rows = np.concatenate([data.observations[members], data.tokens[members], data.preceding_actions[members]], axis=1)
        shuffled_rows = np.concatenate([shuffled.observations[members], shuffled.tokens[members], shuffled.preceding_actions[members]], axis=1)
        assert sorted(map(tuple, original_rows)) == sorted(map(tuple, shuffled_rows))
    assert not np.array_equal(shuffled.observations, data.observations)
    activations, probabilities = replay_activations(module, shuffled, device=DEVICE)
    assert np.max(np.abs(activations - data.activations)) > 0.01
    assert np.max(np.abs(probabilities - data.policy_probabilities)) > 1e-5
    np.testing.assert_allclose(activations[reset_rows], data.activations[reset_rows], atol=2e-6)
    np.testing.assert_array_equal(shuffled_history(data, seed=10).observations, shuffled.observations)


@pytest.mark.parametrize("passive", [True, False])
def test_candidates_are_valid_distinct_and_change_posteriors(passive):
    strength = None if passive else 1.0
    parameters = {"alpha": 0.94, "x": 0.4, "strength": strength}
    candidates = candidate_parameters(**parameters)
    assert 10 <= len(candidates) <= 12
    assert candidates == candidate_parameters(**parameters)
    assert len({tuple(candidate.items()) for candidate in candidates}) == len(candidates)
    assert parameters not in candidates
    data = _forced_data(passive)
    true = replay_beliefs(data, **parameters)[0]
    discrepancies = [np.max(np.abs(replay_beliefs(data, **candidate)[0] - true)) for candidate in candidates]
    assert max(discrepancies) > 0.05
    if not passive:
        routing = [candidate for candidate in candidates if candidate["alpha"] == 0.94 and candidate["x"] == 0.4]
        assert routing
        for candidate in routing:
            actual = controlled_kernels(**candidate).sum(axis=-1)
            np.testing.assert_allclose(actual, np.broadcast_to(wing_model().emission_matrix.T, actual.shape), atol=1e-14)


def test_cleanup_on_forward_failure_and_invalid_warmup():
    config = _config(False)
    module = _module(config)
    module.train()
    counts = [len(block._forward_hooks) for block in module.encoder.blocks]

    def fail(block, inputs):
        raise RuntimeError("intentional forward failure")

    handle = module.encoder.blocks[0].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="intentional forward failure"):
            collect_control_data(module, env_config=config, n_steps=3, seed=1, device=DEVICE, warmup=2)
        assert module.training
        assert [len(block._forward_hooks) for block in module.encoder.blocks] == counts
        with pytest.raises(RuntimeError, match="intentional forward failure"):
            replay_activations(module, _forced_data(False), device=DEVICE)
        assert module.training
        assert [len(block._forward_hooks) for block in module.encoder.blocks] == counts
    finally:
        handle.remove()
    with pytest.raises(ValueError, match="warmup"):
        collect_control_data(module, env_config=config, n_steps=3, seed=1, device=DEVICE, warmup=9)
    data = _forced_data(False)
    with pytest.raises(ValueError, match="complete ordered history"):
        replay_beliefs(replace(data, episode_steps=data.episode_steps + 1), alpha=0.94, x=0.4, strength=1.0)
    with pytest.raises(ValueError, match="suffix_length"):
        replay_beliefs(data, alpha=0.94, x=0.4, strength=1.0, suffix_length=-1)
