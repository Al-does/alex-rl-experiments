from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from envs.hmm import HMMEnv, condition_edge

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5 import (
    simplex_scores,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.alpha095.process import (
    COMPONENT_PARAMETERS,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.process import (
    environment_config,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.shared import (
    MODEL_CONFIG,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.simplex_data import (
    HORIZON,
    collect_histories,
    extract_layers,
    geometry_metrics,
    representation_sites,
)


@pytest.fixture
def module():
    torch.manual_seed(130)
    return FactoredReproductionActorCritic(
        observation_space=gym.spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
        action_space=gym.spaces.Discrete(3),
        model_config={**MODEL_CONFIG, "d_model": 16, "d_mlp": 32},
    ).eval()


@pytest.mark.parametrize("variant", [2, 3])
def test_complete_histories_match_manual_action_conditioned_filter(module, variant):
    config = environment_config(variant, component_parameters=COMPONENT_PARAMETERS)
    histories = collect_histories(module, config, episodes=2, seed=82)
    assert histories.observations.shape == (2, HORIZON, 6)
    assert (histories.tokens[:, 0] == -1).all()
    assert (histories.actions[:, -1] == -1).all()
    assert (histories.actions[:, :-1] >= 0).all()
    assert histories.diagnostic_error < 1e-12
    assert np.all(histories.observations[:, 0] == 0)
    np.testing.assert_array_equal(
        histories.observations[:, 1:, 3:].argmax(axis=2),
        histories.actions[:, :-1],
    )
    diagnostic_config = {
        **config,
        "diagnostics": {"belief": True, "state": True, "tokens": True, "transitions": True},
    }
    env = HMMEnv(diagnostic_config)
    try:
        seed = int(np.random.SeedSequence(82).spawn(2)[0].generate_state(1)[0])
        observation, _ = env.reset(seed=seed)
        source = env.model.initial_distribution.copy()
        edges = env.model.edge_transition_matrices
        np.testing.assert_allclose(histories.beliefs[0, 0], source @ env.model.transition_matrix)
        for step in range(1, HORIZON):
            observation, _, terminated, truncated, info = env.step(int(histories.actions[0, step - 1]))
            np.testing.assert_array_equal(observation, histories.observations[0, step])
            source = condition_edge(source, edges, int(histories.tokens[0, step]))
            edges = np.asarray(info["executed_edge_transition_matrices"])
            np.testing.assert_allclose(
                histories.beliefs[0, step], source @ info["executed_transition_matrix"],
                atol=1e-13,
            )
            assert bool(terminated or truncated) == (step == HORIZON - 1)
    finally:
        env.close()


def test_full_episode_extraction_is_causal_and_matches_policy_encoding(module):
    config = environment_config(3, component_parameters=COMPONENT_PARAMETERS)
    histories = collect_histories(module, config, episodes=2, seed=91)
    values = extract_layers(module, histories.observations)
    assert values.shape == (2, HORIZON, len(module.encoder.blocks) + 1, 16)
    with torch.inference_mode():
        for step in (0, 1, 25, 126, 127):
            prefix = torch.from_numpy(histories.observations[:, :step + 1])
            encoded = module.encoder.forward_complete_episode(prefix)
            np.testing.assert_allclose(
                values[:, step, -1], encoded[:, -1].numpy(), atol=2e-6,
            )
        state = {
            key: torch.from_numpy(np.repeat(value[None], 2, axis=0))
            for key, value in module.get_initial_state().items()
        }
        for step in range(HORIZON):
            encoded, state = module.encode_step(
                torch.from_numpy(histories.observations[:, step]), state,
            )
            np.testing.assert_allclose(
                values[:, step, -1], encoded.numpy(), atol=2e-6,
            )
            if step < HORIZON - 1:
                np.testing.assert_array_equal(
                    module.action_distribution_inputs(encoded).argmax(dim=-1).numpy(),
                    histories.actions[:, step],
                )
    altered = histories.observations.copy()
    altered[:, 80:] = 0
    np.testing.assert_allclose(
        extract_layers(module, altered)[:, :80], values[:, :80], atol=1e-6,
    )


def test_independent_seed_streams_and_batch_size_invariance(module):
    config = environment_config(2, component_parameters=COMPONENT_PARAMETERS)
    first = collect_histories(module, config, episodes=2, seed=66, batch_size=1)
    same = collect_histories(module, config, episodes=2, seed=66, batch_size=2)
    different = collect_histories(module, config, episodes=2, seed=67)
    np.testing.assert_allclose(first.beliefs, same.beliefs)
    np.testing.assert_array_equal(first.observations, same.observations)
    assert not np.array_equal(first.tokens, different.tokens)


def test_raw_overshoots_are_scored_without_projection():
    target = np.array([[0.8, 0.1, 0.1, 0, 0, 0], [0, 0, 0, 0.8, 0.1, 0.1]])
    prediction = target.copy()
    prediction[0, :2] = [1.1, -0.2]
    before = prediction.copy()
    metrics = geometry_metrics(prediction, target)
    assert metrics["outside_simplex_fraction"] == 0.5
    assert metrics["coordinate_min"] == -0.2
    assert metrics["coordinate_max"] == 1.1
    assert metrics["mse"] == pytest.approx(0.015)
    assert metrics["component_posterior"]["mse"] == pytest.approx(0)
    np.testing.assert_array_equal(prediction, before)


@pytest.mark.parametrize("sites", [("post_final_norm",), ("layer_4",), ("layer_4", "layer_2")])
def test_selected_sites_match_full_extraction(module, sites):
    observations = np.zeros((2, 8, 6), dtype=np.float32)
    all_sites = representation_sites(module)
    full = extract_layers(module, observations)
    selected = extract_layers(module, observations, sites=sites)
    assert selected.shape == (2, 8, len(sites), 16)
    np.testing.assert_array_equal(selected, full[:, :, [all_sites.index(site) for site in sites]])
    assert all(not block._forward_hooks for block in module.encoder.blocks)


@pytest.mark.parametrize("sites", [(), ("missing",), ("layer_4", "layer_4")])
def test_invalid_representation_selection(module, sites):
    with pytest.raises(ValueError, match="nonempty distinct subset"):
        representation_sites(module, sites)


def test_refresh_reuses_scores_without_model_loading_or_recalculation(tmp_path, monkeypatch):
    monkeypatch.setattr(simplex_scores, "ROOT", tmp_path)
    monkeypatch.setattr(simplex_scores, "REPO", tmp_path)
    monkeypatch.setattr(simplex_scores, "RUNS", {"variant_fixture": ("leaf", "run")})
    for step in (0, 20):
        path = tmp_path / "leaf" / "results" / "run" / "checkpoint_probes" / f"steps_{step:09d}"
        path.mkdir(parents=True)
        report = {
            "agent_steps": step, "n_test": 8,
            "metadata": {"policy_mode": "greedy", "seed": 42, "warmup_per_episode": 0},
            "layers": {"layer_4": {"probe_fits": {
                "pending_token_distribution": {"r_squared": 0.25, "mse": 0.01},
            }}},
            "policy": {"mean_reward_state_occupancy": 0.0},
        }
        (path / "probe_battery.json").write_text(json.dumps(report))
    source = tmp_path / "source"
    source.mkdir()
    report_path = source / "report_0.json"
    report_path.write_text(json.dumps({"provenance": {"agent_steps": 20.0}}))
    original = {
        "name": "Variant Fixture", "checkpoints": ["Initialization", "Final"],
        "sites": ["post_final_norm"], "predictions": {"preserved": [-0.5, 1.5]},
        "metrics": {"preserved": 0.12},
    }
    (source / "data.js").write_text("window.SIMPLEX_DATA=" + json.dumps({
        "description": "Fixture", "runs": [original],
        "reports": [{"name": "variant_fixture full report", "path": "report_0.json"}],
    }) + ";\n")
    supplement = tmp_path / "supplement.json"
    supplement.write_text(json.dumps({"runs": {"variant_fixture": {
        "source_report_sha256": simplex_scores.digest(report_path), "site": "layer_4",
        "protocol": {"log_probability_floor": 1e-12},
        "checkpoints": {"Final": {"metrics": {"n_evaluated": 8, "r_squared": 0.5, "mse": 0.1}}},
    }}}))

    def forbidden(*args, **kwargs):
        pytest.fail("cached refresh must not load a model, collect histories or fit probes")

    for name in ("fit_missing_log_ntp", "load_model", "collect_histories", "fit_grouped_affine"):
        monkeypatch.setattr(simplex_scores, name, forbidden)
    exported = []
    monkeypatch.setattr(simplex_scores, "write_nonergodic_belief_explorer", lambda output, runs, **kwargs: exported.extend(runs))
    simplex_scores.refresh(source, tmp_path / "output", supplement_path=supplement)
    run = exported[0]
    assert run["metrics"] == original["metrics"]
    assert run["predictions"] == original["predictions"]
    assert run["sites"] == original["sites"]
    assert run["layer_scores"]["Final"]["layer_4"]["Activation → NTP R² (archived)"]["value"] == 0.25
    assert run["layer_scores"]["Final"]["layer_4"]["Activation → log NTP R²"]["value"] == 0.5
    assert run["task_scores"]["Final reward occupancy (archived)"]["value"] == 0.0
    assert not any("Bayes" in name for name in run["task_scores"])
    report_path.write_text(json.dumps({"provenance": {"agent_steps": 20.0}, "changed": True}))
    with pytest.raises(ValueError, match="different source report"):
        simplex_scores.refresh(source, tmp_path / "mismatch", supplement_path=supplement)
