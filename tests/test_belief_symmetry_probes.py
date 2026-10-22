"""Focused, network-free tests for the cycle 4/5 belief-symmetry probes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.probes import fit_affine_probe, predictive_belief_update, probe_predict
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.analysis import (
    RIDGE,
    _checkpoint_requests,
    _coarse_spec,
    _coarse_targets,
    _install_checkpoint_import_aliases,
    decompose_belief,
    reconstruct_belief,
    run_probe_condition,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.campaign_analysis import (
    _run_paths,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    SEEDS,
    TARGET_VARIANTS,
    TRAJECTORY_SUFFIX,
    _candidate_bases,
    _final_checkpoint_name,
    _select_source_base,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.trajectory_campaign import (
    _bootstrap_mean_ci,
    aggregate as aggregate_trajectories,
    write_campaign,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.token_swap_diagnostic.analysis import (
    _validate_intervention_environment,
    evaluate_token_swap,
    paired_token_swap_activations,
    swap_state_0_1_tokens,
)
from learners.models import TransformerModel
from harness.context import RunContext


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


def test_token_swap_exchanges_only_first_two_observation_channels():
    observations = np.asarray(
        [[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]],
        dtype=np.float32,
    )

    swapped = swap_state_0_1_tokens(observations)

    np.testing.assert_array_equal(
        swapped,
        [[0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]],
    )
    np.testing.assert_array_equal(
        observations,
        [[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]],
    )


def test_equivariant_decoding_preserves_token_swap_mse():
    targets = np.asarray(
        [[0.7, 0.2, 0.1], [0.1, 0.4, 0.5], [0.25, 0.6, 0.15]],
        dtype=np.float64,
    )
    residual = np.asarray([0.02, -0.01, 0.03])
    factual = targets + residual
    swapped = factual[:, [1, 0, 2]]

    metrics = evaluate_token_swap(
        factual_activations=factual,
        swapped_activations=swapped,
        factual_targets=targets,
        weight=np.eye(3),
        bias=np.zeros(3),
    )

    assert metrics["counterfactual_minus_factual_mse"] == pytest.approx(0.0)
    assert metrics["counterfactual_over_factual_mse"] == pytest.approx(1.0)
    assert metrics["equivariance_mse"] == pytest.approx(0.0)
    assert metrics["state_2_invariance_rmse"] == pytest.approx(0.0)
    assert metrics["antisymmetric_sign_reversal_rmse"] == pytest.approx(0.0)


@pytest.mark.parametrize("cycle", (4, 5))
def test_variant_2_environment_supports_exact_token_swap_counterfactual(cycle):
    environment = _environment(cycle, 2)
    try:
        _validate_intervention_environment(environment, cycle=cycle)
    finally:
        environment.close()


def test_paired_token_swap_replay_reconstructs_rollout_activations():
    config = {
        **importlib.import_module(
            "experiments.mess3_reward_state_action_symmetry_cycle_4.shared"
        ).environment_config(2),
        "episode_length": 128,
        "randomize_first_episode_length": False,
        "diagnostics": {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        },
    }

    def make_environment():
        return HMMEnv(config)

    environment = make_environment()
    try:
        target = make_transducer_target(environment)
        module = TransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config={
                "context_len": 2,
                "d_model": 24,
                "n_layers": 1,
                "n_heads": 3,
                "max_seq_len": 8,
            },
        )
    finally:
        environment.close()
    data = collect_probe_data(
        module,
        make_environment,
        n_steps=40,
        seed=42,
        policy_mode="random",
        n_envs=2,
        warmup=64,
        store_observations=True,
        initial_belief=target[0],
        action_outcome_operator=target[1],
        initial_outcome_operator=target[2],
    )

    factual, swapped, indices, error = paired_token_swap_activations(
        module,
        data,
        device="cpu",
    )

    assert len(indices) > 0
    assert error < 2e-5
    np.testing.assert_allclose(factual, data.activations[indices], atol=2e-5)
    assert np.sqrt(np.mean(np.square(swapped - factual))) > 0.0


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

    def list_objects_v2(self, *, Bucket: str, Prefix: str, MaxKeys: int = 1):
        del Bucket, MaxKeys
        self.requested.append(Prefix)
        contents = [{"Key": key} for key in self.existing if key.startswith(Prefix)]
        return {"Contents": contents[:1]}


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


def test_campaign_run_paths_prefer_suffix_and_accept_legacy_results(tmp_path):
    current, legacy = _run_paths(tmp_path, cycle=4, variant=2, seed=43)
    assert current.name == "condition_summary.json"
    assert current.parent.name == "mess3-rsa-c4-belief-symmetry-probe-0035-v2-seed43"
    assert legacy.parent.name == "mess3-rsa-c4-belief-symmetry-probe-v2-seed43"


def test_subset_preserves_optional_observations_none():
    data = SimpleNamespace(
        activations=np.arange(6, dtype=np.float64).reshape(3, 2),
        beliefs=np.zeros((3, 3)),
        diagnostic_beliefs=np.zeros((3, 3)),
        tokens=np.zeros(3, dtype=np.int64),
        previous_tokens=np.zeros(3, dtype=np.int64),
        env_indices=np.zeros(3, dtype=np.int64),
        episode_steps=np.arange(3),
        states=np.zeros(3, dtype=np.int64),
        actions=np.zeros((3, 1), dtype=np.int64),
        rewards=np.zeros(3, dtype=np.float64),
        observations=None,
    )
    from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes import (
        analysis as probe_analysis,
    )

    subset = probe_analysis._subset(data, np.asarray([0, 2]))
    assert subset.observations is None
    assert subset.activations.shape == (2, 2)


def test_checkpoint_manifest_includes_init_and_every_saved_checkpoint(tmp_path):
    manifest = {
        "checkpoints": [
            {
                "label": "initial",
                "training_iteration": 0,
                "path": "initial_checkpoint",
            },
            {
                "label": "checkpoint_000000",
                "training_iteration": 1,
                "path": "checkpoints/checkpoint_000000",
            },
            {
                "label": "checkpoint_000001",
                "training_iteration": 2,
                "path": "checkpoints/checkpoint_000001",
            },
        ]
    }
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(manifest))

    assert _checkpoint_requests(tmp_path) == manifest["checkpoints"]


def test_requested_target_runs_at_init_and_every_manifest_checkpoint(
    tmp_path, monkeypatch
):
    bundle = tmp_path / "bundle"
    schedule = [
        {
            "label": "initial",
            "training_iteration": 0,
            "path": "initial_checkpoint",
        },
        {
            "label": "checkpoint_000000",
            "training_iteration": 1,
            "path": "checkpoints/checkpoint_000000",
        },
        {
            "label": "checkpoint_000001",
            "training_iteration": 2,
            "path": "checkpoints/checkpoint_000001",
        },
    ]
    for request in schedule:
        checkpoint = bundle / request["path"]
        checkpoint.mkdir(parents=True)
        (checkpoint / "rllib_checkpoint.json").write_text("{}")
    (bundle / "checkpoint_manifest.json").write_text(
        json.dumps({"checkpoints": schedule})
    )
    (bundle / "source_provenance.json").write_text(
        json.dumps({"requested_target": "antisymmetric_b0_minus_b1"})
    )
    calls = []

    def fake_probe(context, checkpoint, *, cycle, variant, label, target_names):
        calls.append(
            {
                "checkpoint": checkpoint,
                "cycle": cycle,
                "variant": variant,
                "label": label,
                "target_names": target_names,
            }
        )
        return {
            "checkpoint": label,
            "targets": {"antisymmetric_b0_minus_b1": {"global_mse_ratio": 0.5}},
        }

    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_4."
        "belief_symmetry_probes.analysis"
    )
    monkeypatch.setattr(module, "probe_checkpoint", fake_probe)
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        run_id="trajectory-test",
        resume_from=bundle,
    )

    summary = run_probe_condition(context, cycle=5, variant=3)

    assert list(summary["checkpoints"]) == [
        "initial",
        "checkpoint_000000",
        "checkpoint_000001",
    ]
    assert [call["target_names"] for call in calls] == [
        ("antisymmetric_b0_minus_b1",)
    ] * 3
    assert json.loads(
        (context.results_dir / "condition_summary.json").read_text()
    ) == summary


def test_full_belief_trajectory_skips_imported_training_checkpoints(
    tmp_path, monkeypatch
):
    bundle = tmp_path / "bundle"
    schedule = [
        {
            "label": "initial",
            "training_iteration": 0,
            "path": "initial_checkpoint",
        },
        {
            "label": "checkpoint_000000",
            "training_iteration": 1,
            "path": "checkpoints/checkpoint_000000",
        },
        {
            "label": "checkpoint_000001",
            "training_iteration": 2,
            "path": "checkpoints/checkpoint_000001",
        },
    ]
    for request in schedule:
        checkpoint = bundle / request["path"]
        checkpoint.mkdir(parents=True)
        (checkpoint / "rllib_checkpoint.json").write_text("{}")
    (bundle / "checkpoint_manifest.json").write_text(
        json.dumps({"checkpoints": schedule})
    )
    (bundle / "source_provenance.json").write_text(
        json.dumps({"requested_target": "full_belief"})
    )
    calls = []

    def fake_probe(context, checkpoint, *, cycle, variant, label, target_names):
        calls.append(label)
        return {
            "checkpoint": label,
            "targets": {"full_belief": {"mse": 0.002}},
        }

    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_4."
        "belief_symmetry_probes.analysis"
    )
    monkeypatch.setattr(module, "probe_checkpoint", fake_probe)
    monkeypatch.setattr(
        module,
        "_training_full_belief_mse_by_label",
        lambda **kwargs: {"initial": 0.01, "checkpoint_000000": 0.005},
    )
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        run_id="trajectory-test",
        resume_from=bundle,
    )

    summary = run_probe_condition(context, cycle=5, variant=2)

    assert calls == ["checkpoint_000001"]
    assert summary["checkpoints"]["initial"]["imported_from"] == (
        "training_checkpoint_probe_curve"
    )
    assert summary["checkpoints"]["initial"]["targets"]["full_belief"]["mse"] == 0.01
    assert summary["checkpoints"]["checkpoint_000001"]["targets"]["full_belief"]["mse"] == 0.002


def test_target_campaign_assigns_requested_variants_and_all_seeds():
    assert TARGET_VARIANTS["symmetric_b2"] == (1, 2, 3)
    assert TARGET_VARIANTS["antisymmetric_b0_minus_b1"] == (1, 2, 3)
    assert TARGET_VARIANTS["coarse_b2"] == (2,)
    assert TARGET_VARIANTS["full_belief"] == (1, 2, 3)
    assert SEEDS == (42, 43, 44, 45, 46)


def test_bootstrap_mean_ci_brackets_empirical_mean():
    values = np.asarray(
        [
            [1.0, 2.0],
            [1.2, 2.2],
            [0.8, 1.8],
            [1.1, 2.1],
            [0.9, 1.9],
        ],
        dtype=np.float64,
    )
    mean, ci_low, ci_high = _bootstrap_mean_ci(values, n_resamples=5000, seed=42)
    assert mean.tolist() == pytest.approx(values.mean(axis=0).tolist())
    assert np.all(ci_low <= mean)
    assert np.all(ci_high >= mean)


def test_trajectory_campaign_aggregates_and_plots_every_checkpoint(tmp_path):
    target = "symmetric_b2"
    schedule = [
        {"label": "initial", "training_iteration": 0},
        {"label": "checkpoint_000000", "training_iteration": 1},
        {"label": "checkpoint_000001", "training_iteration": 2},
    ]
    for variant in TARGET_VARIANTS[target]:
        for seed in SEEDS:
            run_id = (
                f"mess3-rsa-c5-belief-trajectory-{TRAJECTORY_SUFFIX}-"
                f"symmetric-b2-v{variant}-seed{seed}"
            )
            run_dir = tmp_path / f"variant_{variant}" / "results" / run_id
            run_dir.mkdir(parents=True)
            checkpoints = {}
            for index, point in enumerate(schedule):
                mse = 0.01 / (index + 1) + variant * 0.001 + seed / 10_000
                checkpoints[point["label"]] = {
                    "targets": {
                        target: {
                            "mse": mse,
                            "global_mse_ratio": variant + seed / 100 + index / 10,
                        }
                    }
                }
            (run_dir / "condition_summary.json").write_text(
                json.dumps(
                    {
                        "requested_target": target,
                        "checkpoint_schedule": schedule,
                        "checkpoints": checkpoints,
                    }
                )
            )

    summary = aggregate_trajectories(tmp_path, cycle=5, target=target)
    assert summary["metric"] == "held-out affine probe MSE"
    assert summary["uncertainty_band"]["n_resamples"] == 10_000
    assert "750,000 environment steps" in summary["checkpoint_scope"]
    assert set(summary["variants"]) == {"variant_1", "variant_2", "variant_3"}
    assert all(
        curve["training_iterations"] == [0, 1, 2]
        for curve in summary["variants"].values()
    )
    assert all(
        curve["agent_steps"] == [0, 33_000, 66_000]
        for curve in summary["variants"].values()
    )
    for curve in summary["variants"].values():
        assert len(curve["ci_95_low"]) == len(curve["mean"])
        assert len(curve["ci_95_high"]) == len(curve["mean"])
        assert all(
            low <= mean <= high
            for low, mean, high in zip(
                curve["ci_95_low"], curve["mean"], curve["ci_95_high"], strict=True
            )
        )

    png = write_campaign(tmp_path, cycle=5, target=target)
    assert png.is_file()
    assert not png.with_suffix(".pdf").exists()


def test_coarse_campaign_aggregates_full_belief_from_training_curves(tmp_path):
    target = "coarse_b2"
    schedule = [
        {"label": "initial", "training_iteration": 0},
        {"label": "checkpoint_000000", "training_iteration": 1},
    ]
    probes_root = tmp_path / "study" / "belief_symmetry_probes"
    study_root = tmp_path / "study"
    for seed in SEEDS:
        run_id = (
            f"mess3-rsa-c5-belief-trajectory-{TRAJECTORY_SUFFIX}-"
            f"coarse-b2-v2-seed{seed}"
        )
        run_dir = probes_root / "variant_2" / "results" / run_id
        run_dir.mkdir(parents=True)
        checkpoints = {}
        for index, point in enumerate(schedule):
            checkpoints[point["label"]] = {
                "targets": {
                    target: {
                        "mse": 0.01 / (index + 1) + seed / 10_000,
                    }
                }
            }
        (run_dir / "condition_summary.json").write_text(
            json.dumps(
                {
                    "requested_target": target,
                    "checkpoint_schedule": schedule,
                    "checkpoints": checkpoints,
                }
            )
        )
        training_points = []
        for index, point in enumerate(schedule):
            step = index * 33_000
            training_points.append(
                {
                    "agent_steps": step,
                    **(
                        {}
                        if step == 0
                        else {"checkpoint_name": point["label"]}
                    ),
                    "training_iteration": point["training_iteration"],
                    "mse": 0.02 / (index + 1) + seed / 10_000,
                    "probe": {"target": "exact_predictive_bayesian_belief"},
                }
            )
        training_dir = study_root / "variant_2" / "results" / f"mess3-rsa-c5-v2-seed{seed}"
        training_dir.mkdir(parents=True)
        (training_dir / "checkpoint_probe_curve.json").write_text(
            json.dumps({"checkpoints": training_points})
        )

    summary = aggregate_trajectories(probes_root, cycle=5, target=target)
    assert summary["comparison"]["target"] == "full_belief"
    assert summary["comparison"]["source"] == "training_checkpoint_probe_curve"
    assert summary["comparison"]["variant_2"]["agent_steps"] == [0, 33_000]

    png = write_campaign(probes_root, cycle=5, target=target)
    assert png.is_file()


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


@pytest.mark.parametrize("cycle", (4, 5))
def test_token_swap_diagnostic_leaves_import_and_encode_cycle_variant(cycle):
    module = importlib.import_module(
        f"experiments.mess3_reward_state_action_symmetry_cycle_{cycle}."
        "token_swap_diagnostic.experiment"
    )
    assert module.CYCLE == cycle
    assert module.VARIANT == 2
    assert callable(module.run)
