"""Scientific, wiring, and recipe tests for factored MESS3 cycle 1."""

from __future__ import annotations

from dataclasses import replace
import importlib

import numpy as np
import pytest

from envs.hmm import HMMEnv
from experiments.mess3_factored_cycle_1 import dynamics
from experiments.mess3_factored_cycle_1.analysis import (
    geometry_report,
    nested_function_features,
)
from experiments.mess3_factored_cycle_1.observation import (
    FactoredObservationHMMEnv,
)
from experiments.mess3_factored_cycle_1.prediction import (
    _causal_prediction_examples,
    _joint_token_targets,
    _sequence_chunks,
)
from experiments.mess3_factored_cycle_1.reference import (
    aware_filter_update,
    coarse_e2_transition,
    e2_lumpability_audit,
    factor_targets,
    posterior_from_symbol,
    structural_audit_report,
    value_invariance_audit,
)
from experiments.mess3_factored_cycle_1.reference_campaign import (
    CampaignProtocol,
    ConditionSpec,
    run_reference_campaign,
    simulate_condition,
)
from experiments.mess3_factored_cycle_1.shared import (
    BASE_MODEL_CONFIG,
    Condition,
    _pretraining_audits,
    build_config,
    environment_config,
    make_environment,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models.transformer import TransformerModel


LEAVES = (
    "e1_f2_diagonal_factored",
    "e1_f1_diagonal_factored",
    "e1_f2_product_factored",
    "e2_lambda_0p0_factored",
    "e2_lambda_0p5_factored",
    "e2_lambda_1p0_factored",
    "e2_lambda_1p5_factored",
    "e2_lambda_2p0_factored",
    "e3a_additive_product_factored",
    "e3b_conjunctive_product_factored",
    "e3c_additive_diagonal_factored",
    "e4_gauge_a050_factored",
    "e4_gauge_a085_reactive_null",
    "e1_product_factored_asymmetric",
    "e1_product_joint_asymmetric",
    "e2_lambda_1p0_joint",
    "e3a_product_factored_asymmetric",
    "e3a_product_joint_asymmetric",
    "e4_gauge_a050_joint",
)


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_shift_orientation_and_product_action_indexing():
    base = dynamics.BASE_TRANSITION
    for state in range(3):
        shift = (2 - state) % 3
        shifted = dynamics.shifted_transition(shift)
        assert shifted[state, 2] == pytest.approx(base[state].max())
    product = dynamics.product_shift_kernels()
    diagonal = dynamics.diagonal_shift_kernels()
    assert product.shape == (9, 9, 9)
    assert diagonal.shape == (3, 9, 9)
    np.testing.assert_allclose(
        product[3 * 1 + 2],
        np.kron(
            dynamics.shifted_transition(1),
            dynamics.shifted_transition(2),
        ),
    )
    np.testing.assert_allclose(diagonal.sum(axis=-1), 1.0)


def test_e2_tilt_table_and_within_block_coupling_are_exact():
    menu = dynamics.e2_action_transitions()
    np.testing.assert_allclose(
        menu[:, 0, 2],
        [0.100, 0.33242786, 0.67931897],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        menu[:, 2, 2],
        [0.400, 0.74923471, 0.19695031],
        atol=1e-7,
    )
    modulated = dynamics.modulate_within_non_goal(
        menu[1],
        context_state=0,
        coupling_lambda=1.25,
    )
    np.testing.assert_allclose(modulated[:, 2], menu[1, :, 2])
    np.testing.assert_allclose(
        modulated[:, 0] / modulated[:, 1],
        np.exp(1.25) * menu[1, :, 0] / menu[1, :, 1],
    )
    audit = e2_lumpability_audit()
    assert audit["passed"]
    assert audit["worst_deviation"] <= 1e-12


def test_e4_gauge_only_relabels_the_reachable_action_menu():
    kernels = dynamics.gauge_kernels()
    for source_f1 in range(3):
        source = 3 * source_f1
        observed_rows = {
            tuple(np.round(kernel[source].reshape(3, 3).sum(axis=0), 12))
            for kernel in kernels
        }
        expected_rows = {
            tuple(np.round(dynamics.shifted_transition(shift)[0], 12))
            for shift in range(3)
        }
        assert observed_rows == expected_rows


def test_structural_audits_pass_and_e3_dissociation_is_present():
    value_audit = value_invariance_audit()
    assert value_audit["passed"]
    report = structural_audit_report()
    assert report["status"] == "passed"
    e3 = report["audits"]["E3_function_coupling"]
    assert e3["e3a"]["value_nonadditive_residual"] < 1e-12
    assert e3["e3a"]["policy_factorizes"]
    assert e3["e3b"]["value_nonadditive_residual"] > 0.25
    assert e3["e3b"]["policy_factorizes"]
    assert not e3["e3c"]["policy_is_single_factor_rule"]
    assert report["monte_carlo_audits"]["status"] == "not_run"


def test_pr35_factored_model_and_policy_presentations_share_simulation():
    condition = Condition(
        name="test",
        experiment="E3a",
        action_kind="product",
        reward_kind="additive",
        alpha1=0.60,
        alpha2=0.55,
    )
    factored = make_environment(condition)
    joint = HMMEnv(
        environment_config(
            replace(
                condition,
                token_encoding="joint",
                action_encoding="joint",
            )
        )
    )
    try:
        assert factored.model.n_states == 9
        assert factored.model.n_tokens == 9
        np.testing.assert_allclose(
            factored.model.transition_matrix,
            np.kron(dynamics.BASE_TRANSITION, dynamics.BASE_TRANSITION),
        )
        assert factored.observation_space.shape == (12,)
        assert joint.observation_space.shape == (18,)
        factored_observation, factored_info = factored.reset(seed=7)
        joint_observation, joint_info = joint.reset(seed=7)
        assert factored_info["decision_step"] == joint_info["decision_step"]
        joint_symbol = int(joint_observation[:9].argmax())
        assert factored_observation[joint_symbol // 3] == 1.0
        assert factored_observation[3 + joint_symbol % 3] == 1.0
        for action in (0, 5, 8):
            factored_step = factored.step(action)
            joint_step = joint.step(action)
            assert factored_step[1:4] == joint_step[1:4]
    finally:
        factored.close()
        joint.close()


def test_task_action_encodings_and_reward_components():
    product = make_environment(
        Condition(
            name="product",
            experiment="E1",
            action_kind="product",
            reward_kind="f2_goal",
            alpha1=0.55,
            alpha2=0.55,
        )
    )
    diagonal = make_environment(
        Condition(
            name="diagonal",
            experiment="E1",
            action_kind="diagonal",
            reward_kind="f2_goal",
            alpha1=0.55,
            alpha2=0.55,
        )
    )
    try:
        np.testing.assert_allclose(
            product.task.encode_action(5),
            [0, 1, 0, 0, 0, 1],
        )
        np.testing.assert_allclose(diagonal.task.encode_action(2), [0, 0, 1])
        assert product.action_space.n == 9
        assert diagonal.action_space.n == 3
    finally:
        product.close()
        diagonal.close()


def test_exact_filter_matches_environment_belief_diagnostics():
    condition = Condition(
        name="filter",
        experiment="E4",
        action_kind="e4_gauge",
        reward_kind="f2_goal",
        alpha1=0.50,
        alpha2=0.85,
    )
    config = environment_config(condition)
    config["diagnostics"] = {
        "belief": True,
        "tokens": True,
        "transitions": True,
    }
    environment = FactoredObservationHMMEnv(config)
    try:
        _, info = environment.reset(seed=3)
        belief = info["belief_current"]
        for action in (1, 2, 0, 1):
            _, _, _, _, info = environment.step(action)
            symbol = info["raw_token_current"]
            belief = aware_filter_update(
                belief,
                action,
                symbol,
                kernels=environment.task.kernels,
                emission=environment.model.emission_matrix,
            )
            np.testing.assert_allclose(belief, info["belief_current"], atol=1e-12)
    finally:
        environment.close()


def test_factor_targets_and_coarse_filter_shapes():
    rng = np.random.default_rng(4)
    beliefs = rng.dirichlet(np.ones(9), size=20)
    targets = factor_targets(beliefs)
    assert targets["f1"].shape == targets["f2"].shape == (20, 3)
    assert targets["f2_goal_block"].shape == (20, 1)
    np.testing.assert_allclose(targets["product"].sum(axis=1), 1.0)
    assert coarse_e2_transition().shape == (3, 2, 2)
    posterior = posterior_from_symbol(
        np.array([0.5, 0.5]),
        1,
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )
    np.testing.assert_allclose(posterior, [2 / 11, 9 / 11])


def test_pr35_geometry_report_recovers_direct_sum_fixture():
    rng = np.random.default_rng(5)
    first = rng.dirichlet(np.ones(3), size=500)
    second = rng.dirichlet(np.ones(3), size=500)
    joint = (first[:, :, None] * second[:, None, :]).reshape(-1, 9)
    first_contrast = first[:, :2] - first[:, 2:3]
    second_contrast = second[:, :2] - second[:, 2:3]
    activations = np.concatenate(
        [first_contrast, second_contrast, np.zeros((len(first), 4))],
        axis=1,
    )
    report = geometry_report(
        activations,
        joint,
        expected_quotient_dimension=4,
    )
    assert report["dimension_predictions"] == {"factored": 4, "joint": 8}
    assert report["activation_geometry"]["rank"] == 4
    assert report["joint_product_mse"] < 1e-30
    assert (
        report["factor_geometry"]["pairwise_subspace_overlap"]["f1_vs_f2"]
        < 1e-8
    )
    features = nested_function_features(first, second)
    assert features["factor_only"].shape == (500, 5)
    assert features["with_joint_interactions"].shape == (500, 9)


def test_prediction_targets_cover_factored_and_joint_presentations():
    joint = np.zeros((1, 3, 12), dtype=np.float32)
    factored = np.zeros((1, 3, 9), dtype=np.float32)
    symbols = (1, 5, 8)
    for step, symbol in enumerate(symbols):
        joint[0, step, symbol] = 1.0
        factored[0, step, symbol // 3] = 1.0
        factored[0, step, 3 + symbol % 3] = 1.0
    factored[0, 1, 7] = 1.0
    factored[0, 2, 8] = 1.0
    np.testing.assert_array_equal(
        _joint_token_targets(joint, token_encoding="joint"),
        [symbols],
    )
    np.testing.assert_array_equal(
        _joint_token_targets(factored, token_encoding="factored"),
        [symbols],
    )
    inputs, labels = _causal_prediction_examples(
        factored,
        np.asarray([symbols]),
        token_width=6,
    )
    # The action suffix from t+1 is moved beside x_t, without leaking x_{t+1}.
    assert inputs[0, 0, 7] == 1.0
    assert inputs[0, 0, 3:6].argmax() == 1 % 3
    chunks = _sequence_chunks(
        inputs,
        labels,
        lookback=2,
        chunk_length=2,
    )
    contexts, lengths, observations, labels, masks = chunks
    assert contexts.shape == (1, 2, 9)
    np.testing.assert_array_equal(lengths, [0])
    np.testing.assert_array_equal(labels[0], [5, 8])
    np.testing.assert_array_equal(masks[0], [True, True])
    assert observations.shape == (1, 2, 9)


def test_every_leaf_builds_a_fresh_cycle5_scale_smoke_recipe(tmp_path):
    context = _context(tmp_path)
    assert BASE_MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 10,
        "max_seq_len": 32,
    }
    names = set()
    for leaf in LEAVES:
        module = importlib.import_module(
            f"experiments.mess3_factored_cycle_1.{leaf}.experiment"
        )
        names.add(module.CONDITION.name)
        first = module.build_config(context)
        second = module.build_config(context)
        assert first is not second
        assert first.gamma == 0.99
        assert first.lambda_ == 0.95
        assert first.num_env_runners == 0
        assert first.train_batch_size_per_learner == 1_024
        assert first.minibatch_size == 256
        assert first.rl_module_spec.module_class is TransformerModel
        environment = first.env(first.env_config)
        try:
            assert environment.model.n_states == 9
            assert environment.model.n_tokens == 9
            assert environment.config.episode_length == 1024
        finally:
            environment.close()
    assert names == set(LEAVES)


def test_condition_validation_rejects_unknown_science():
    with pytest.raises(ValueError, match="unknown action"):
        Condition(
            name="bad",
            experiment="E1",
            action_kind="magic",
            reward_kind="f2_goal",
            alpha1=0.55,
            alpha2=0.55,
        )


def test_reference_simulator_is_deterministic_and_reports_chain_uncertainty():
    protocol = CampaignProtocol(
        n_chains=32,
        n_steps=96,
        burn_in=16,
        seed=11,
    )
    spec = ConditionSpec(
        "E1_test",
        "diagonal",
        "f2_goal",
        0.55,
        0.55,
    )
    first = simulate_condition(
        spec,
        protocol,
        policy_kinds=("aware", "reactive", "greedy"),
    )
    second = simulate_condition(
        spec,
        protocol,
        policy_kinds=("aware", "reactive", "greedy"),
    )
    assert first["policies"] == second["policies"]
    assert first["policies"]["aware"]["standard_error"] >= 0.0
    assert first["best_constant"]["estimate"] > 0.0
    assert first["_chain_values"]["aware"].shape == (32,)


def test_reduced_reference_campaign_has_complete_audit_schema():
    report = run_reference_campaign(
        CampaignProtocol(
            n_chains=24,
            n_steps=80,
            burn_in=16,
            seed=7,
        )
    )
    assert report["schema_version"] == 1
    assert set(report["audits"]) == {"A1", "A2", "A3", "A4", "A5", "A6"}
    assert set(report["audits"]["A3"]["conditions"]) == {
        "E1",
        "E2",
        "E3b",
        "E3c",
        "E4",
    }
    assert len(report["references"]["e2_lambda_sweep"]) == 5
    assert report["protocol"]["reward_timing"] == (
        "decision_state_before_transition"
    )


def test_passing_canonical_campaign_authorizes_non_smoke_training(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )
    report = _pretraining_audits(context)
    assert report["training_authorization"] == "registered_A1_A6_passed"
    assert report["reference_campaign"]["status"] == "passed"
    assert report["reference_campaign"]["max_standard_error"] <= 5e-4
