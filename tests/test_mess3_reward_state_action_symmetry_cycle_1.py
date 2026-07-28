"""Focused tests for discrete MESS3 reward-state action symmetry."""

from __future__ import annotations

import importlib
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from envs.hmm import HMMEnv, TransitionEvent
from envs.mess3.model import control_model
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_1.shared import (
    _log_spaced_records,
    environment_config,
)
from experiments.mess3_reward_state_action_symmetry_cycle_1.task import (
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
EXPECTED_ORACLE_POLICIES = {
    1: (POSITIVE_ACTION, POSITIVE_ACTION, POSITIVE_ACTION),
    2: (POSITIVE_ACTION, POSITIVE_ACTION, NOOP_ACTION),
    3: (POSITIVE_ACTION, NEGATIVE_ACTION, NOOP_ACTION),
}


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_actions_apply_the_requested_state_dependent_effects(variant):
    model = control_model()
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


def test_task_rewards_pre_transition_state_two_and_encodes_actions():
    task = ActionSymmetryTask(model=control_model(), variant=3)
    decision = task.resolve_action(POSITIVE_ACTION, 0, control_model())

    for state, expected in ((0, 0.0), (1, 0.0), (2, 1.0)):
        reward, components = task.reward(
            TransitionEvent(
                step=0,
                state_before=state,
                state_after=0,
                raw_token_before=0,
                raw_token_after=0,
            ),
            decision,
        )
        assert reward == expected
        assert components == {"occupancy_reward": expected}

    np.testing.assert_array_equal(
        task.encode_action(NEGATIVE_ACTION),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="outside the action space"):
        task.transition_matrix_for_action(3)


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_described_oracle_policy_uniquely_maximizes_state_two_occupancy(variant):
    task = ActionSymmetryTask(model=control_model(), variant=variant)
    occupancies = {}
    for policy in itertools.product(range(3), repeat=3):
        transition = np.stack(
            [
                task.transition_matrix_for_action(policy[state])[state]
                for state in range(3)
            ]
        )
        system = transition.T - np.eye(3)
        system[-1] = 1.0
        stationary = np.linalg.solve(
            system,
            np.array([0.0, 0.0, 1.0]),
        )
        occupancies[policy] = stationary[2]

    ranked = sorted(occupancies, key=occupancies.get, reverse=True)

    assert ranked[0] == EXPECTED_ORACLE_POLICIES[variant]
    assert occupancies[ranked[0]] > occupancies[ranked[1]]


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
        "experiments.mess3_reward_state_action_symmetry_cycle_1."
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
    finally:
        environment.close()


def test_discrete_action_probe_target_matches_environment_diagnostics():
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

    assert data.actions.shape == (11, 1)
    assert set(data.actions.astype(int).reshape(-1)) <= {0, 1, 2}
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
