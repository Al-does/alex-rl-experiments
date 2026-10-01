from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.pusher_b_reward_state_action_symmetry_cycle_1 import (
    probe_analysis,
    report_probes,
)


def _rollout_operators(preset, variant, reward_state, *, n_steps, seed):
    """Replay the adapter's operator bookkeeping against a random policy."""

    env = probe_analysis.make_environment_factory(preset, variant, reward_state)()
    rng = np.random.default_rng(seed)
    n_states = len(env.model.initial_distribution)
    reset_edges = np.asarray(env.model.edge_transition_matrices, dtype=np.float64)
    reset_transition = np.asarray(env.model.transition_matrix, dtype=np.float64)
    source, executed, diagnostic, groups, steps = [], [], [], [], []
    episode_id = -1
    try:
        _, info = env.reset(seed=seed)
        episode_step = 0
        pending = reset_edges
        for _ in range(n_steps):
            if episode_step == 0:
                episode_id += 1
                source.append(np.eye(n_states))
                executed.append(reset_transition)
                pending = reset_edges
            else:
                source.append(pending[int(info["visible_token_current"])])
                executed.append(np.asarray(info["executed_transition_matrix"]))
                pending = np.asarray(info["executed_edge_transition_matrices"])
            diagnostic.append(np.asarray(info["belief_current"]))
            groups.append(episode_id)
            steps.append(episode_step)
            action = int(rng.integers(env.action_space.n))
            _, _, terminated, truncated, info = env.step(action)
            episode_step += 1
            if terminated or truncated:
                _, info = env.reset()
                episode_step = 0
    finally:
        env.close()
    return (
        np.asarray(env.model.initial_distribution, dtype=np.float64),
        np.stack(source),
        np.stack(executed),
        np.stack(diagnostic),
        np.asarray(groups),
        np.asarray(steps),
    )


@pytest.mark.parametrize("preset,reward_state", [("b90", "A"), ("b10", "B")])
@pytest.mark.parametrize("variant", [2, 3])
def test_transducer_target_matches_environment_belief(preset, variant, reward_state):
    initial, source, executed, diagnostic, groups, steps = _rollout_operators(
        preset, variant, reward_state, n_steps=600, seed=7
    )
    assert groups.max() >= 3, "rollout must cross several episode boundaries"
    beliefs = probe_analysis.transducer_targets(
        initial, source, executed, groups, steps
    )
    np.testing.assert_allclose(beliefs.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(beliefs, diagnostic, atol=1e-10)


def test_transducer_target_depends_on_executed_actions():
    initial, source, executed, _, groups, steps = _rollout_operators(
        "b90", 3, "A", n_steps=300, seed=3
    )
    beliefs = probe_analysis.transducer_targets(
        initial, source, executed, groups, steps
    )
    # Replace every executed operator by the uncontrolled reset kernel: the
    # action-conditioned target must change on non-reset rows.
    env = probe_analysis.make_environment_factory("b90", 3, "A")()
    try:
        uncontrolled_edges = np.asarray(env.model.edge_transition_matrices)
        uncontrolled_transition = np.asarray(env.model.transition_matrix)
    finally:
        env.close()
    passive_source = source.copy()
    passive_executed = executed.copy()
    for row in np.flatnonzero(steps > 0):
        token = int(np.argmax([np.allclose(source[row], k) for k in uncontrolled_edges]))
        passive_source[row] = uncontrolled_edges[token]
        passive_executed[row] = uncontrolled_transition
    passive = probe_analysis.transducer_targets(
        initial, passive_source, passive_executed, groups, steps
    )
    assert np.max(np.abs(passive - beliefs)) > 1e-3


def test_reset_rows_restart_from_initial_distribution():
    initial, source, executed, _, groups, steps = _rollout_operators(
        "b10", 2, "B", n_steps=400, seed=11
    )
    beliefs = probe_analysis.transducer_targets(
        initial, source, executed, groups, steps
    )
    reset_rows = np.flatnonzero(steps == 0)
    assert len(reset_rows) > 1
    expected = initial @ executed[reset_rows[0]]
    for row in reset_rows:
        np.testing.assert_allclose(beliefs[row], expected / expected.sum())


def test_report_loads_points_and_plots(tmp_path):
    study = tmp_path / "study"
    probes = study / "b90_reward_a" / "variant_2" / "results" / "run" / "transducer_probes"
    probes.mkdir(parents=True)
    for name, steps, r2 in [("final", 50_000, 0.9), ("initial", 0, 0.3)]:
        (probes / f"{name}.json").write_text(
            json.dumps(
                {
                    "checkpoint_name": name,
                    "env_steps": steps,
                    "r_squared": r2,
                    "one_minus_r_squared": 1 - r2,
                    "n_fit": 1,
                    "n_test": 1,
                    "checkpoint": "s3://bucket/x",
                }
            )
        )
    arms = report_probes.load_points(study)
    assert list(arms) == ["b90_reward_a.variant_2"]
    assert [p["env_steps"] for p in arms["b90_reward_a.variant_2"]] == [0, 50_000]
    report_probes.plot(arms, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
