"""Focused, network-free tests for the cycle 4/5 belief-symmetry probes."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.probes import fit_affine_probe, predictive_belief_update, probe_predict
from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.analysis import (
    RIDGE,
    _coarse_spec,
    _coarse_targets,
    _install_checkpoint_import_aliases,
    decompose_belief,
    reconstruct_belief,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    _candidate_bases,
    _final_checkpoint_name,
    _prepare_results_history,
    _select_source_base,
)


def _environment(cycle: int, variant: int) -> HMMEnv:
    shared = importlib.import_module(
        f"experiments.mess3_reward_state_action_symmetry_cycle_{cycle}.shared"
    )
    return HMMEnv(shared.environment_config(variant))


def _full_coarsened_emission(environment: HMMEnv) -> np.ndarray:
    emission = np.asarray(environment.model.emission_matrix, dtype=np.float64)
    return np.column_stack((emission[:, :2].sum(axis=1), emission[:, 2]))


def _filter_history(
    initial: np.ndarray,
    emissions: np.ndarray,
    transitions: dict[int, np.ndarray],
    tokens: list[int],
    actions: list[int],
) -> np.ndarray:
    belief = np.asarray(initial, dtype=np.float64)
    values = []
    for step, token in enumerate(tokens):
        measurement = np.diag(emissions[:, token])
        operator = measurement if step == 0 else transitions[actions[step - 1]] @ measurement
        belief = predictive_belief_update(belief, operator)
        values.append(belief.copy())
    return np.asarray(values)


def test_symmetric_antisymmetric_decomposition_is_invertible():
    beliefs = np.asarray(
        [[0.2, 0.3, 0.5], [0.05, 0.9, 0.05], [1 / 3, 1 / 3, 1 / 3]],
        dtype=np.float64,
    )
    symmetric, antisymmetric = decompose_belief(beliefs)

    assert symmetric.shape == (3, 1)
    assert antisymmetric.shape == (3, 1)
    np.testing.assert_allclose(
        reconstruct_belief(symmetric, antisymmetric), beliefs, atol=1e-15
    )


@pytest.mark.parametrize("cycle", (4, 5))
@pytest.mark.parametrize("variant", (1, 2))
def test_coarse_specs_are_strongly_lumpable_and_stochastic(cycle, variant):
    environment = _environment(cycle, variant)
    try:
        initial, emission, transitions = _coarse_spec(environment)
        np.testing.assert_allclose(emission, [[0.925, 0.075], [0.15, 0.85]])
        np.testing.assert_allclose(initial.sum(), 1.0)
        for action, coarse_transition in transitions.items():
            full = environment.task.transition_matrix_for_action(action)
            destination_lumps = np.column_stack(
                (full[:, :2].sum(axis=1), full[:, 2])
            )
            np.testing.assert_allclose(destination_lumps[0], destination_lumps[1])
            np.testing.assert_allclose(coarse_transition[0], destination_lumps[0])
            np.testing.assert_allclose(coarse_transition[1], destination_lumps[2])
            np.testing.assert_allclose(coarse_transition.sum(axis=1), 1.0)
            assert coarse_transition.shape == (2, 2)
    finally:
        environment.close()


@pytest.mark.parametrize("variant", (1, 2))
def test_coarse_filter_matches_three_state_projection_for_coarsened_emissions(variant):
    environment = _environment(4, variant)
    try:
        coarse_initial, coarse_emission, coarse_transitions = _coarse_spec(environment)
        full_initial = np.asarray(environment.model.initial_distribution)
        full_emission = _full_coarsened_emission(environment)
        full_transitions = {
            action: np.asarray(environment.task.transition_matrix_for_action(action))
            for action in range(environment.action_space.n)
        }
        tokens = [0, 1, 0, 0, 1, 1]
        actions = [1, 2, 0, 1, 0, 2]
        full = _filter_history(
            full_initial, full_emission, full_transitions, tokens, actions
        )
        coarse = _filter_history(
            coarse_initial, coarse_emission, coarse_transitions, tokens, actions
        )
        np.testing.assert_allclose(full[:, 2], coarse[:, 1], atol=1e-12)
    finally:
        environment.close()


def test_coarse_target_differs_from_full_token_projection():
    environment = _environment(4, 2)
    try:
        coarse_initial, coarse_emission, coarse_transitions = _coarse_spec(environment)
        original_emission = np.asarray(environment.model.emission_matrix)
        full_initial = np.asarray(environment.model.initial_distribution)
        full_transitions = {
            action: np.asarray(environment.task.transition_matrix_for_action(action))
            for action in range(environment.action_space.n)
        }
        # Original token 0 distinguishes states 0 and 1. The coarse filter sees
        # only "not token 2", so its posterior is intentionally different.
        original_tokens = [0, 0, 2]
        coarse_tokens = [0, 0, 1]
        actions = [1, 1, 0]
        full = _filter_history(
            full_initial,
            original_emission,
            full_transitions,
            original_tokens,
            actions,
        )
        coarse = _filter_history(
            coarse_initial,
            coarse_emission,
            coarse_transitions,
            coarse_tokens,
            actions,
        )
        assert np.max(np.abs(full[:, 2] - coarse[:, 1])) > 1e-3
    finally:
        environment.close()


def test_coarse_reconstruction_uses_previous_action_for_current_observation():
    environment = _environment(4, 2)
    try:
        initial, emission, transitions = _coarse_spec(environment)
        data = SimpleNamespace(
            episode_steps=np.asarray([0, 1, 2]),
            env_indices=np.asarray([0, 0, 0]),
            tokens=np.asarray([0, 2, 0]),
            # Action 2 produces the transition into row 1; action 1 produces
            # the transition into row 2. Action 0 has not executed yet.
            actions=np.asarray([[2], [1], [0]]),
        )
        reconstructed = _coarse_targets(
            data,
            initial=initial,
            emission=emission,
            transitions=transitions,
        )
        expected = _filter_history(
            initial,
            emission,
            transitions,
            tokens=[0, 1, 0],
            actions=[2, 1, 0],
        )
        np.testing.assert_allclose(reconstructed[:, 0], expected[:, 1], atol=1e-12)
    finally:
        environment.close()


def test_scalar_targets_support_exact_affine_fit():
    rng = np.random.default_rng(7)
    beliefs = rng.dirichlet(np.ones(3), size=128)
    symmetric, antisymmetric = decompose_belief(beliefs)
    features = np.column_stack((symmetric, antisymmetric, rng.normal(size=128)))

    for target in (symmetric, antisymmetric):
        weight, bias = fit_affine_probe(features, target, ridge=RIDGE)
        prediction = probe_predict(weight, bias, features)
        assert target.shape == (128, 1)
        assert prediction.shape == target.shape
        np.testing.assert_allclose(prediction, target, atol=1e-7)


def test_cycle_five_checkpoint_import_aliases_resolve_renamed_task():
    _install_checkpoint_import_aliases(5)
    old_task = importlib.import_module(
        "experiments.mess3_reward_state_action_asymmetry_cycle_5.task"
    )
    new_task = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_5.task"
    )
    assert old_task is new_task


def test_final_checkpoint_name_uses_highest_iteration():
    summary = {
        "trials": [
            {
                "best": "/tmp/checkpoint_000008",
                "last": {"path": "/tmp/checkpoint_000021"},
            }
        ]
    }
    assert _final_checkpoint_name(summary) == "checkpoint_000021"
    with pytest.raises(ValueError, match="no checkpoint"):
        _final_checkpoint_name({"trials": []})


class _FakeS3:
    def __init__(self, existing: set[str]):
        self.existing = existing
        self.requested: list[str] = []

    def head_object(self, *, Bucket: str, Key: str):
        del Bucket
        self.requested.append(Key)
        if Key not in self.existing:
            raise RuntimeError("not found")
        return {}


def test_b2_base_selection_falls_back_to_unprefixed_historical_root():
    bases = _candidate_bases(
        configured_prefix="current-prefix",
        study="mess3_reward_state_action_symmetry_cycle_4",
        variant=2,
        source_run_id="mess3-rsa-c4-v2-seed43",
    )
    expected = (
        "experiments/mess3_reward_state_action_symmetry_cycle_4/"
        "variant_2/mess3-rsa-c4-v2-seed43"
    )
    tune_key = f"{expected}/compact-results/tune_summary.json"
    client = _FakeS3({tune_key})

    base, selected_key = _select_source_base(client, "bucket", bases)

    assert base == expected
    assert selected_key == tune_key
    assert client.requested == [
        f"{bases[0]}/compact-results/tune_summary.json",
        tune_key,
    ]


def test_b2_candidate_bases_are_deduplicated_without_configured_prefix():
    bases = _candidate_bases(
        configured_prefix="",
        study="study",
        variant=1,
        source_run_id="run",
    )
    assert bases == ["experiments/study/variant_1/run"]


def test_result_history_join_preserves_source_and_existing_results(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }

    def git(*args, cwd=None):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    git("init", "--bare", str(remote))
    git("clone", str(remote), str(source))
    (source / "base.txt").write_text("base\n")
    git("add", "base.txt", cwd=source)
    git("commit", "-m", "base", cwd=source)
    base = git("rev-parse", "HEAD", cwd=source).stdout.strip()
    git("switch", "--orphan", "results", cwd=source)
    (source / "base.txt").unlink(missing_ok=True)
    old_result = source / "experiments" / "old" / "results" / "result.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text("{}\n")
    shared = source / "shared.txt"
    shared.write_text("results version\n")
    git("add", "experiments", cwd=source)
    git("add", "shared.txt", cwd=source)
    git("commit", "-m", "old result", cwd=source)
    git("push", "origin", "results", cwd=source)
    git("switch", "-c", "feature", base, cwd=source)
    shared.write_text("feature version\n")
    new_source = source / "experiments" / "study" / "probe.py"
    new_source.parent.mkdir(parents=True)
    new_source.write_text("TARGET = 'symmetric'\n")
    git("add", "experiments", "shared.txt", cwd=source)
    git("commit", "-m", "probe source", cwd=source)
    monkeypatch.setenv("VAST_EXPERIMENT_DIR", str(source))

    assert _prepare_results_history("results")
    assert old_result.read_text() == "{}\n"
    assert new_source.read_text() == "TARGET = 'symmetric'\n"
    assert shared.read_text() == "feature version\n"
    git("merge-base", "--is-ancestor", "origin/results", "HEAD", cwd=source)


@pytest.mark.parametrize("cycle", (4, 5))
@pytest.mark.parametrize("variant", (1, 2, 3))
def test_all_probe_leaves_import_and_encode_cycle_variant(cycle, variant):
    module = importlib.import_module(
        f"experiments.mess3_reward_state_action_symmetry_cycle_{cycle}."
        f"belief_symmetry_probes.variant_{variant}.experiment"
    )
    assert module.CYCLE == cycle
    assert module.VARIANT == variant
    assert callable(module.run)
    assert Path(module.__file__).name == "experiment.py"
