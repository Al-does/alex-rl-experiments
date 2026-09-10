from __future__ import annotations

from dataclasses import fields, replace
import inspect

import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv
from envs.hmm.belief import condition_edge
from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_1 import analysis, geometry_data
from experiments.strata_token_guess_cycle_1.geometry_data import (
    GeometryData,
    collect_geometry_data,
    replay_beliefs,
)
from experiments.strata_token_guess_cycle_1.process import environment_config


class _ProbeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.blocks = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        )
        self.encoder.final_norm = torch.nn.Identity()

    def get_initial_state(self):
        return {"steps": np.zeros(1, dtype=np.int64)}

    def encode_step_pre_final_norm(self, observations, state):
        residual = torch.cat([observations, state["steps"].float()], dim=1)[:, None]
        for block in self.encoder.blocks:
            residual = block(residual)
        return residual[:, 0], {"steps": state["steps"] + 1}

    def action_distribution_inputs(self, residual):
        return residual[:, :2]


def _public_trace(*, guess=0, changed_reward=False):
    envs = [
        HMMEnv(
            {
                **environment_config(),
                "episode_length": 7 + 4 * index,
                "randomize_first_episode_length": False,
                "diagnostics": {"belief": True, "tokens": True},
            }
        )
        for index in range(3)
    ]
    observations, beliefs, episode_ids, episode_steps, rewards = [], [], [], [], []
    current_ids = [40, 5, 90]
    next_id = 100
    try:
        current = [env.reset(seed=19 + index) for index, env in enumerate(envs)]
        if changed_reward:
            for env in envs:
                env.task.reward = lambda event, decision: (7.0, {})
        for _ in range(24):
            for index, env in enumerate(envs):
                observation, info = current[index]
                observations.append(observation)
                beliefs.append(info["belief_current"])
                episode_ids.append(current_ids[index])
                episode_steps.append(info["decision_step"])
                next_observation, reward, terminated, truncated, next_info = env.step(guess)
                np.testing.assert_array_equal(
                    next_observation, np.eye(2)[info["raw_token_current"]]
                )
                rewards.append(reward)
                if terminated or truncated:
                    current_ids[index] = next_id
                    next_id += 1
                    current[index] = env.reset(seed=next_id)
                else:
                    current[index] = next_observation, next_info
    finally:
        for env in envs:
            env.close()
    count = len(observations)
    return GeometryData(
        activations=np.zeros((count, 2, 3)),
        beliefs=np.asarray(beliefs),
        observations=np.asarray(observations),
        episode_ids=np.asarray(episode_ids),
        episode_steps=np.asarray(episode_steps),
        mask=np.ones(count, dtype=bool),
    ), np.asarray(rewards)


def _reference_replay(data, *, suffix_length=None, **parameters):
    model = strata_model(**(environment_config()["model"]["kwargs"] | parameters))
    beliefs = np.empty((len(data.observations), 3))
    ntp = np.empty((len(data.observations), 2))
    for episode in np.unique(data.episode_ids):
        members = np.flatnonzero(data.episode_ids == episode)
        for position, row in enumerate(members):
            source = model.initial_distribution.copy()
            length = position if suffix_length is None else suffix_length
            start = max(1, position - length + 1)
            for member in members[start : position + 1]:
                source = condition_edge(
                    source,
                    model.edge_transition_matrices,
                    int(data.observations[member].argmax()),
                )
            beliefs[row] = source @ model.transition_matrix
            ntp[row] = source @ model.emission_matrix
    return beliefs, ntp


@pytest.mark.parametrize("n_steps", [128, 20_000])
def test_geometry_collection_exact_budget_full_histories_and_resets(monkeypatch, n_steps):
    collected = []

    def capture(module, **kwargs):
        assert kwargs["warmup"] == 0
        raw = analysis.collect_probe_data(module, **kwargs)
        collected.append(raw)
        return raw

    monkeypatch.setattr(geometry_data, "collect_probe_data", capture)
    module = _ProbeModule()
    data = collect_geometry_data(
        module,
        n_steps=n_steps,
        seed=np.random.SeedSequence(71),
        device=torch.device("cpu"),
    )
    assert len(collected) == 1
    raw = collected[0]
    scored = np.flatnonzero(raw.episode_steps >= 32)
    stop = int(scored[n_steps - 1]) + 1
    assert len(data.mask) == stop < len(raw.episode_steps)
    assert data.mask.dtype == bool
    assert data.mask.sum() == n_steps
    assert data.mask[-1]
    np.testing.assert_array_equal(data.mask, data.episode_steps >= 32)
    assert data.activations.shape == (stop, 2, 3)
    assert data.beliefs.shape == (stop, 3)
    assert data.observations.shape == (stop, 2)
    np.testing.assert_array_equal(data.activations, raw.activations[:stop])
    np.testing.assert_array_equal(data.beliefs, raw.joint_beliefs[:stop])
    np.testing.assert_array_equal(data.observations, raw.observations[:stop])
    np.testing.assert_array_equal(data.episode_ids, raw.episode_ids[:stop])
    np.testing.assert_array_equal(data.activations[:, 0, 2], data.episode_steps)
    for episode in np.unique(data.episode_ids):
        rows = np.flatnonzero(data.episode_ids == episode)
        np.testing.assert_array_equal(data.episode_steps[rows], np.arange(len(rows)))
        np.testing.assert_array_equal(data.observations[rows[0]], np.zeros(2))
    assert all(len(block._forward_hooks) == 0 for block in module.encoder.blocks)
    beliefs, ntp = replay_beliefs(data)
    np.testing.assert_allclose(beliefs, data.beliefs, atol=1e-10, rtol=0)
    np.testing.assert_allclose(ntp.sum(axis=1), 1, atol=1e-14)
    if n_steps == 20_000:
        assert len(np.unique(data.episode_ids)) > analysis.N_ENVS
        assert not np.array_equal(
            raw.env_indices[scored[:n_steps]], np.arange(n_steps) % analysis.N_ENVS
        )
        first_lengths = []
        for env_index in range(analysis.N_ENVS):
            env_rows = np.flatnonzero(raw.env_indices[:stop] == env_index)
            episodes = np.unique(data.episode_ids[env_rows])
            lengths = [np.count_nonzero(data.episode_ids == episode) for episode in episodes]
            first_lengths.append(lengths[0])
            assert all(length == 1024 for length in lengths[1:-1])
            assert any(length == 1024 for length in lengths[1:-1])
        assert len(set(first_lengths)) > 1
        assert all(1 <= length <= 1024 for length in first_lengths)


def test_legacy_collector_default_is_unchanged_and_matches_scored_rows():
    signature = inspect.signature(analysis.collect_probe_data)
    assert signature.parameters["warmup"].default == 32
    optional = {field.name: field.default for field in fields(analysis.ProbeData)}
    assert optional["episode_ids"] is None
    assert optional["env_indices"] is None
    arguments = {
        "n_steps": 128,
        "seed": np.random.SeedSequence(71),
        "device": torch.device("cpu"),
    }
    legacy = analysis.collect_probe_data(_ProbeModule(), **arguments)
    data = collect_geometry_data(_ProbeModule(), **arguments)
    np.testing.assert_array_equal(legacy.activations, data.activations[data.mask])
    np.testing.assert_array_equal(legacy.joint_beliefs, data.beliefs[data.mask])
    np.testing.assert_array_equal(legacy.observations, data.observations[data.mask])
    np.testing.assert_array_equal(legacy.episode_steps, data.episode_steps[data.mask])
    assert legacy.episode_ids is not None
    assert legacy.env_indices is not None


def test_geometry_retry_reuses_seed_and_preserves_chronological_prefix(monkeypatch):
    original = analysis.collect_probe_data
    calls = []
    seed = np.random.SeedSequence(71)

    def short_first_collection(module, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            kwargs["n_steps"] = 8
        return original(module, **kwargs)

    monkeypatch.setattr(geometry_data, "collect_probe_data", short_first_collection)
    arguments = {"n_steps": 128, "seed": seed, "device": torch.device("cpu")}
    retried = collect_geometry_data(_ProbeModule(), **arguments)
    assert len(calls) == 2
    assert calls[1]["n_steps"] == 2 * calls[0]["n_steps"]
    assert all(call["seed"] is seed and call["warmup"] == 0 for call in calls)
    monkeypatch.setattr(geometry_data, "collect_probe_data", original)
    regular = collect_geometry_data(_ProbeModule(), **arguments)
    for name in ("activations", "beliefs", "observations", "episode_steps", "mask"):
        np.testing.assert_array_equal(getattr(retried, name), getattr(regular, name))
    np.testing.assert_allclose(replay_beliefs(retried)[0], regular.beliefs, atol=1e-10)


@pytest.mark.parametrize("n_steps", [0, -1, 1.5, True, np.bool_(True), "128", None])
def test_geometry_collection_rejects_invalid_budgets(n_steps):
    with pytest.raises(ValueError, match="strictly positive integer"):
        collect_geometry_data(
            None, n_steps=n_steps, seed=np.random.SeedSequence(1), device=torch.device("cpu")
        )


def test_collector_removes_only_its_hooks_on_failure(monkeypatch):
    module = _ProbeModule()
    existing = module.encoder.blocks[0].register_forward_hook(lambda *args: None)

    def fail(residual):
        raise RuntimeError("policy failure")

    monkeypatch.setattr(module, "action_distribution_inputs", fail)
    try:
        with pytest.raises(RuntimeError, match="policy failure"):
            collect_geometry_data(
                module,
                n_steps=1,
                seed=np.random.SeedSequence(1),
                device=torch.device("cpu"),
            )
        assert len(module.encoder.blocks[0]._forward_hooks) == 1
        assert len(module.encoder.blocks[1]._forward_hooks) == 0
    finally:
        existing.remove()


@pytest.mark.parametrize("guess", [0, 1])
def test_exact_replay_matches_public_environment_on_forced_guesses(guess):
    data, _ = _public_trace(guess=guess)
    beliefs, ntp = replay_beliefs(data)
    expected_beliefs, expected_ntp = _reference_replay(data)
    np.testing.assert_allclose(beliefs, data.beliefs, atol=1e-10, rtol=0)
    np.testing.assert_allclose(beliefs, expected_beliefs, atol=1e-14, rtol=0)
    np.testing.assert_allclose(ntp, expected_ntp, atol=1e-14, rtol=0)
    assert np.max(np.abs(ntp - beliefs @ strata_model(**environment_config()["model"]["kwargs"]).emission_matrix)) > 1e-3


def test_replay_never_conditions_on_rewards_guesses_or_diagnostic_targets():
    normal, rewards = _public_trace(guess=0)
    altered, changed_rewards = _public_trace(guess=1, changed_reward=True)
    assert np.all(changed_rewards == 7)
    assert not np.array_equal(rewards, changed_rewards)
    np.testing.assert_array_equal(normal.observations, altered.observations)
    np.testing.assert_array_equal(normal.beliefs, altered.beliefs)
    altered = replace(
        altered,
        activations=np.full_like(altered.activations, np.nan),
        beliefs=np.full_like(altered.beliefs, np.nan),
        mask=~altered.mask,
    )
    for expected, actual in zip(replay_beliefs(normal), replay_beliefs(altered)):
        np.testing.assert_array_equal(expected, actual)
    assert {field.name for field in fields(GeometryData)} == {
        "activations", "beliefs", "observations", "episode_ids", "episode_steps", "mask"
    }


@pytest.mark.parametrize("suffix_length", [0, 1, 2, 4, 8, 32, 1000, None])
@pytest.mark.parametrize("parameters", [{}, {"alpha": 0.83, "t0": 0.23, "t1": 0.71}])
def test_suffix_replay_matches_scalar_episode_safe_oracle(suffix_length, parameters):
    data, _ = _public_trace()
    actual = replay_beliefs(data, suffix_length=suffix_length, **parameters)
    expected = _reference_replay(data, suffix_length=suffix_length, **parameters)
    for value, target in zip(actual, expected):
        np.testing.assert_allclose(value, target, atol=1e-14, rtol=0)
    if suffix_length is not None:
        full = replay_beliefs(data, **parameters)
        complete = data.episode_steps <= suffix_length
        for value, target in zip(actual, full):
            np.testing.assert_allclose(value[complete], target[complete], atol=1e-14)
    if suffix_length == 0:
        model = strata_model(**(environment_config()["model"]["kwargs"] | parameters))
        np.testing.assert_allclose(
            actual[0], np.tile(model.initial_distribution @ model.transition_matrix, (72, 1))
        )
        np.testing.assert_allclose(
            actual[1], np.tile(model.initial_distribution @ model.emission_matrix, (72, 1))
        )


@pytest.mark.parametrize("suffix_length", [None, 0, 1, 4])
def test_replay_does_not_use_future_visible_tokens(suffix_length):
    data, _ = _public_trace()
    observations = data.observations.copy()
    future = data.episode_steps >= 5
    observations[future] = observations[future, ::-1]
    changed = replace(data, observations=observations)
    for original, altered in zip(
        replay_beliefs(data, suffix_length=suffix_length),
        replay_beliefs(changed, suffix_length=suffix_length),
    ):
        np.testing.assert_array_equal(original[~future], altered[~future])


@pytest.mark.parametrize("suffix_length", [None, 0, 2])
@pytest.mark.parametrize("fault", ["missing_reset", "gap", "duplicate", "out_of_order"])
def test_replay_rejects_incomplete_or_nonchronological_histories(fault, suffix_length):
    data, _ = _public_trace()
    steps = data.episode_steps.copy()
    members = np.flatnonzero(data.episode_ids == data.episode_ids[0])
    if fault == "missing_reset":
        steps[members] += 1
    elif fault == "gap":
        steps[members[3:]] += 1
    elif fault == "duplicate":
        steps[members[3]] = steps[members[2]]
    else:
        steps[members[2:4]] = steps[members[2:4]][::-1]
    with pytest.raises(ValueError, match="start at step 0.*chronologically"):
        replay_beliefs(replace(data, episode_steps=steps), suffix_length=suffix_length)


@pytest.mark.parametrize("observation", [[0, 0], [1, 1], [0.5, 0.5], [np.nan, 1]])
def test_replay_rejects_invalid_visible_observations(observation):
    data, _ = _public_trace()
    observations = data.observations.copy()
    observations[np.flatnonzero(data.episode_steps > 0)[0]] = observation
    with pytest.raises(ValueError, match="visible-token one-hots"):
        replay_beliefs(replace(data, observations=observations))


def test_replay_rejects_nonblank_reset():
    data, _ = _public_trace()
    observations = data.observations.copy()
    observations[0] = [1, 0]
    with pytest.raises(ValueError, match="reset observations must be blank"):
        replay_beliefs(replace(data, observations=observations))


@pytest.mark.parametrize("suffix_length", [None, 1, 4])
def test_replay_raises_on_impossible_observation(suffix_length):
    data, _ = _public_trace()
    assert np.any(data.observations[:, 0] == 1)
    with pytest.raises(ValueError, match="impossible observation"):
        replay_beliefs(data, t0=0, t1=0, suffix_length=suffix_length)


@pytest.mark.parametrize("suffix_length", [-1, 1.5, True, np.bool_(False), "2"])
def test_replay_rejects_invalid_suffix_lengths(suffix_length):
    data, _ = _public_trace()
    with pytest.raises(ValueError, match="suffix_length"):
        replay_beliefs(data, suffix_length=suffix_length)
