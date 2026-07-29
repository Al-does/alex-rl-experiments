"""Focused tests for the sticky-state action-symmetry cycle."""

from __future__ import annotations

import importlib
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from envs.hmm import HMMEnv
from envs.mess3.model import (
    STICKY_CONTROL_TRANSITION_MATRIX,
    sticky_control_model,
)
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.design import (
    EXPECTED_ORACLE_POLICIES,
    analytic_design_summary,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.shared import (
    TOTAL_ENV_STEPS,
    _log_spaced_records,
    environment_config,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.task import (
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
    ActionSymmetryTask,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models import TransformerModel


EXPECTED_DIRECTIONS = {
    1: {
        POSITIVE_ACTION: (1, 1, 1),
        NEGATIVE_ACTION: (-1, -1, -1),
    },
    2: {
        POSITIVE_ACTION: (1, 1, -1),
        NEGATIVE_ACTION: (-1, -1, -1),
    },
    3: {
        POSITIVE_ACTION: (1, -1, -1),
        NEGATIVE_ACTION: (-1, 1, -1),
    },
}
EXPECTED_ORACLE_OCCUPANCIES = {
    1: 0.5700133141432316,
    2: 0.35651858484973303,
    3: 0.35651858484973303,
}
EXPECTED_ORACLE_GAPS = {
    1: 0.1296046565229799,
    2: 0.08017165223389872,
    3: 0.08017165223389872,
}


def _stationary_state_two(transition: np.ndarray) -> float:
    system = transition.T - np.eye(3)
    system[-1] = 1.0
    return float(np.linalg.solve(system, np.array([0.0, 0.0, 1.0]))[2])


def test_cycle_4_selects_the_sticky_baseline_and_preserves_cycle_2_controls():
    config = environment_config(2)

    assert config["model"]["factory"] == "envs.mess3.model:sticky_control_model"
    assert config["model"]["kwargs"] == {"alpha": 0.85}
    assert config["delay"] == 0
    assert config["episode_length"] == 1024
    assert config["task"]["kwargs"] == {"variant": 2, "effect_size": 1.5}
    assert TOTAL_ENV_STEPS == 700_000
    np.testing.assert_array_equal(
        sticky_control_model().transition_matrix,
        STICKY_CONTROL_TRANSITION_MATRIX,
    )


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_actions_apply_requested_effects_to_sticky_baseline(variant):
    model = sticky_control_model()
    task = ActionSymmetryTask(model=model, variant=variant)
    base = model.transition_matrix

    np.testing.assert_array_equal(
        task.transition_matrix_for_action(NOOP_ACTION),
        base,
    )
    for action, directions in EXPECTED_DIRECTIONS[variant].items():
        controlled = task.transition_matrix_for_action(action)
        np.testing.assert_allclose(controlled.sum(axis=1), 1.0)
        np.testing.assert_allclose(
            controlled[:, 0] / controlled[:, 1],
            base[:, 0] / base[:, 1],
        )
        for state, direction in enumerate(directions):
            assert np.sign(controlled[state, 2] - base[state, 2]) == direction
            base_odds = base[state, 2] / (1.0 - base[state, 2])
            controlled_odds = (
                controlled[state, 2] / (1.0 - controlled[state, 2])
            )
            assert np.log(controlled_odds / base_odds) == pytest.approx(
                direction * task.effect_size
            )


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_oracle_policy_and_gap_match_cycle_4_design(variant):
    task = ActionSymmetryTask(model=sticky_control_model(), variant=variant)
    occupancies = {}
    for policy in itertools.product(range(3), repeat=3):
        transition = np.stack(
            [
                task.transition_matrix_for_action(policy[state])[state]
                for state in range(3)
            ]
        )
        occupancies[policy] = _stationary_state_two(transition)

    ranked = sorted(occupancies, key=occupancies.get, reverse=True)
    oracle = ranked[0]

    assert oracle == EXPECTED_ORACLE_POLICIES[variant]
    assert occupancies[oracle] == pytest.approx(
        EXPECTED_ORACLE_OCCUPANCIES[variant]
    )
    assert occupancies[oracle] - occupancies[ranked[1]] == pytest.approx(
        EXPECTED_ORACLE_GAPS[variant]
    )


def test_analytic_design_records_reachable_noop_boundary_and_larger_gap():
    design = analytic_design_summary()
    variant_2 = design["fully_observed"]["variant_2"]

    assert design["baseline_transition_matrix"][2] == [0.3, 0.3, 0.4]
    assert design["variant_2_one_step_noop_belief_threshold"] == pytest.approx(
        0.4621414003564968
    )
    assert variant_2["oracle_gap"] == pytest.approx(
        EXPECTED_ORACLE_GAPS[2]
    )
    assert variant_2["oracle_gap"] > 5.0 * 0.01563
    assert (
        variant_2["oracle_stationary_state_2"]
        - variant_2["always_positive_stationary_state_2"]
        == pytest.approx(EXPECTED_ORACLE_GAPS[2])
    )


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_ppo_variant_recipes_build_fresh_discrete_configs(tmp_path, variant):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_4."
        f"variant_{variant}.experiment"
    )

    first = module.build_config(context)
    second = module.build_config(context)

    assert first is not second
    assert first.gamma == 0.99
    assert first.env_config["task"]["kwargs"]["variant"] == variant
    assert first.train_batch_size_per_learner == 2_048
    assert first.minibatch_size == 256
    environment = HMMEnv(first.env_config)
    try:
        assert environment.action_space.n == 3
        assert environment.observation_space.shape == (6,)
        np.testing.assert_array_equal(
            environment.model.transition_matrix,
            STICKY_CONTROL_TRANSITION_MATRIX,
        )
    finally:
        environment.close()


def test_probe_target_matches_sticky_environment_diagnostics():
    config = {
        **environment_config(3),
        "episode_length": 3,
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
                "context_len": 4,
                "d_model": 24,
                "n_layers": 1,
                "n_heads": 3,
                "max_seq_len": 3,
            },
        )
    finally:
        environment.close()

    data = collect_probe_data(
        module,
        make_environment,
        n_steps=11,
        seed=42,
        policy_mode="random",
        n_envs=2,
        warmup=1,
        initial_belief=target[0],
        action_outcome_operator=target[1],
        initial_outcome_operator=target[2],
    )

    np.testing.assert_allclose(
        data.beliefs,
        data.diagnostic_beliefs,
        atol=1e-12,
    )


def test_probe_schedule_selects_init_powers_of_two_and_final():
    records = [
        {
            "checkpoint": SimpleNamespace(path=f"/tmp/checkpoint_{iteration}"),
            "checkpoint_name": f"checkpoint_{iteration}",
            "training_iteration": iteration,
            "agent_steps": iteration * 2_048,
        }
        for iteration in range(1, 7)
    ]

    selected = _log_spaced_records(records)

    assert [record["training_iteration"] for record in selected] == [1, 2, 4, 6]
