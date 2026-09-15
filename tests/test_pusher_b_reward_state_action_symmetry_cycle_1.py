from __future__ import annotations

from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.pusher_b_reward_state_action_symmetry_cycle_1 import (
    shared as pusher_shared,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.learning import (
    NEXT_TOKEN_AUX_COEFFICIENT,
    ActorCriticWithNextTokenAux,
    PPOWithNextTokenAux,
    next_token_targets,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.design import (
    MAX_CONTROLLED_TRANSITION_PROBABILITY,
    analytic_design_summary,
    condition_design_summary,
    controlled_kernels,
    expected_oracle_policy,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.process import (
    EFFECT_SIZE,
    EPISODE_LENGTH,
    PRESETS,
    REWARD_STATES,
    TOKEN_COUNT,
    environment_config,
    pusher_b_model,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.shared import (
    MINIBATCH_SIZE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
    TRAIN_BATCH_SIZE,
    build_config,
    resolved_recipe,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.task import (
    N_ACTIONS,
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
    ActionSymmetryTask,
    direction_matrix,
)
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
from harness.hardware import HardwareProfile, PROFILES


CONDITIONS = [
    (preset, reward_state, variant)
    for preset in PRESETS
    for reward_state in REWARD_STATES
    for variant in (1, 2, 3)
]


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def _diagnostic_config(
    preset: str,
    variant: int,
    reward_state: str,
) -> dict[str, object]:
    config = environment_config(preset, variant, reward_state)
    config["diagnostics"] = {
        "belief": True,
        "state": True,
        "tokens": True,
        "transitions": True,
    }
    return config


@pytest.mark.parametrize(("preset", "reward_state", "variant"), CONDITIONS)
def test_action_conditioned_edges_preserve_pusher_token_conditionals(
    preset,
    reward_state,
    variant,
):
    model = pusher_b_model(preset)
    task = ActionSymmetryTask(
        model=model,
        variant=variant,
        reward_state=reward_state,
    )
    np.testing.assert_allclose(
        task.transition_matrix_for_action(NOOP_ACTION),
        model.transition_matrix,
    )
    np.testing.assert_allclose(
        task.edge_transition_matrices_for_action(NOOP_ACTION),
        model.edge_transition_matrices,
    )
    for action in range(N_ACTIONS):
        transition = task.transition_matrix_for_action(action)
        edges = task.edge_transition_matrices_for_action(action)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)
        np.testing.assert_allclose(edges.sum(axis=0), transition, atol=1e-14)
        assert np.all(transition > 0.0)
        reference_conditionals = (
            model.edge_transition_matrices
            / model.transition_matrix[None, :, :]
        )
        controlled_conditionals = edges / transition[None, :, :]
        np.testing.assert_allclose(
            controlled_conditionals,
            reference_conditionals,
            atol=1e-14,
        )


@pytest.mark.parametrize(("preset", "reward_state", "variant"), CONDITIONS)
def test_full_state_oracle_has_requested_action_symmetry(
    preset,
    reward_state,
    variant,
):
    model = pusher_b_model(preset)
    reward_state_index = model.state_labels.index(reward_state)
    summary = condition_design_summary(preset, variant, reward_state)
    expected = expected_oracle_policy(variant, reward_state_index)
    assert tuple(summary["oracle_policy"]) == expected
    assert summary["oracle_gap"] > 0.008
    assert summary["maximum_controlled_transition_probability"] < (
        MAX_CONTROLLED_TRANSITION_PROBABILITY
    )
    assert summary["minimum_controlled_transition_probability"] > 0.0
    assert summary["oracle_stationary_reward_occupancy"] > (
        summary["baseline_stationary_reward_occupancy"]
    )


@pytest.mark.parametrize("reward_state", REWARD_STATES)
def test_direction_tables_encode_one_two_and_three_optimal_actions(
    reward_state,
):
    model = pusher_b_model("b90")
    reward_state_index = model.state_labels.index(reward_state)
    assert expected_oracle_policy(1, reward_state_index) == (
        POSITIVE_ACTION,
        POSITIVE_ACTION,
        POSITIVE_ACTION,
    )
    assert len(set(expected_oracle_policy(2, reward_state_index))) == 2
    assert len(set(expected_oracle_policy(3, reward_state_index))) == 3
    for variant in (1, 2, 3):
        directions = direction_matrix(variant, reward_state_index)
        assert directions.shape == (3, 3)
        np.testing.assert_array_equal(
            directions[:, NOOP_ACTION],
            np.zeros(3),
        )


def test_effect_size_is_strong_but_does_not_collapse_a_transition():
    summary = analytic_design_summary()
    assert EFFECT_SIZE == 1.5
    assert len(summary["conditions"]) == 12
    assert max(
        condition["maximum_controlled_transition_probability"]
        for condition in summary["conditions"]
    ) == pytest.approx(0.9517924931140221)
    assert min(
        condition["oracle_gap"] for condition in summary["conditions"]
    ) > 0.008


def test_bos_and_delayed_token_action_observation():
    env = HMMEnv(_diagnostic_config("b90", 3, "B"))
    try:
        observation, info = env.reset(seed=23)
        np.testing.assert_array_equal(
            observation,
            np.zeros(2 + N_ACTIONS, dtype=np.float32),
        )
        pending_token = int(info["raw_token_current"])
        observation, _, _, _, info = env.step(NEGATIVE_ACTION)
        np.testing.assert_array_equal(
            observation[:2],
            np.eye(2, dtype=np.float32)[pending_token],
        )
        np.testing.assert_array_equal(
            observation[2:],
            np.eye(N_ACTIONS, dtype=np.float32)[NEGATIVE_ACTION],
        )
        assert info["visible_source_token"] == pending_token
    finally:
        env.close()


@pytest.mark.parametrize("reward_state", REWARD_STATES)
def test_reward_uses_pre_transition_designated_state(reward_state):
    model = pusher_b_model("b10")
    reward_state_index = model.state_labels.index(reward_state)
    env = HMMEnv(_diagnostic_config("b10", 2, reward_state))
    try:
        env.reset(seed=31)
        for action in [0, 1, 2] * 20:
            _, reward, _, _, info = env.step(action)
            assert reward == float(info["state_before"] == reward_state_index)
    finally:
        env.close()


def test_complete_episode_ppo_config_and_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(pusher_shared, "available_cpus", lambda: 64)
    context = _context(tmp_path)
    config = build_config(
        context,
        preset="b90",
        variant=3,
        reward_state="B",
    )
    assert config is not build_config(
        context,
        preset="b90",
        variant=3,
        reward_state="B",
    )
    assert config.seed == 42
    assert config.gamma == 0.99
    assert config.lambda_ == 0.95
    assert config.clip_param == 0.2
    assert config.use_critic and config.use_gae
    assert not config.use_kl_loss
    assert config.vf_loss_coeff == 0.5
    assert config.entropy_coeff == 0.01
    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert config.num_epochs == 6
    assert config.num_env_runners == 0
    assert config.env_config == environment_config("b90", 3, "B")
    assert config.env_config["delay"] == 1
    assert config.env_config["episode_length"] == EPISODE_LENGTH == 127
    assert config.env_config["randomize_first_episode_length"] is False
    assert config.env_runner_cls is FreshEpisodeSingleAgentEnvRunner
    assert config.rollout_fragment_length == "auto"
    assert config.batch_mode == "complete_episodes"
    assert config.rl_module_spec.model_config == MODEL_CONFIG
    assert MODEL_CONFIG["context_length"] == 128
    assert MODEL_CONFIG["max_seq_len"] == 128
    assert MODEL_CONFIG["training_sequence_mode"] == "complete_episode"

    recipe = resolved_recipe(
        context,
        preset="b90",
        variant=3,
        reward_state="B",
    )
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 1_024
    assert recipe["effect_size"] == EFFECT_SIZE
    assert recipe["previous_action_in_observation"] is True
    assert recipe["environment"] == environment_config("b90", 3, "B")
    full_context = replace(context, smoke=False)
    full_config = build_config(
        full_context,
        preset="b90",
        variant=3,
        reward_state="B",
    )
    assert full_config.train_batch_size_per_learner == TRAIN_BATCH_SIZE
    assert full_config.minibatch_size == MINIBATCH_SIZE
    assert full_config.num_env_runners == 52
    assert full_config.num_envs_per_env_runner == 40
    full_recipe = resolved_recipe(
        full_context,
        preset="b90",
        variant=3,
        reward_state="B",
    )
    assert full_recipe["sampling_layout"]["preferred_num_env_runners"] == 52
    assert full_recipe["sampling_layout"]["cpu_limit_policy"] == (
        "at most available CPUs minus one"
    )
    assert (
        full_recipe["sampling_layout"]["episodes_per_sampling_round"] == 2_080
    )
    assert resolved_recipe(
        full_context,
        preset="b90",
        variant=3,
        reward_state="B",
    )["total_env_steps"] == TOTAL_ENV_STEPS == 50_000_000


@pytest.mark.parametrize(
    ("available_cpus", "profile_runners", "expected"),
    [
        (64, None, (52, 40)),
        (33, None, (32, 65)),
        (64, 16, (16, 130)),
    ],
)
def test_sampling_layout_respects_cpu_and_profile_limits(
    tmp_path,
    monkeypatch,
    available_cpus,
    profile_runners,
    expected,
):
    monkeypatch.setattr(
        pusher_shared,
        "available_cpus",
        lambda: available_cpus,
    )
    profile = HardwareProfile(
        "test",
        "cpu",
        profile_runners,
        24,
        None,
    )
    context = replace(_context(tmp_path), smoke=False, hardware=profile)
    assert pusher_shared.pusher_sampling_layout(context, profile) == expected
    assert expected[0] * expected[1] >= 2_065


@pytest.mark.parametrize(("preset", "reward_state", "variant"), CONDITIONS)
def test_all_twelve_leaf_modules_are_importable_and_bound_correctly(
    tmp_path,
    preset,
    reward_state,
    variant,
):
    module_name = (
        "experiments.pusher_b_reward_state_action_symmetry_cycle_1."
        f"{preset}_reward_{reward_state.lower()}.variant_{variant}.experiment"
    )
    module = importlib.import_module(module_name)
    assert callable(module.run)
    config = module.build_config(_context(tmp_path))
    assert config.env_config == environment_config(
        preset,
        variant,
        reward_state,
    )


def test_controlled_kernel_shapes():
    transitions, edges = controlled_kernels("b10", 3, "A")
    assert transitions.shape == (N_ACTIONS, 3, 3)
    assert edges.shape == (N_ACTIONS, 2, 3, 3)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("unknown", 1, "A"), "unknown Pusher-B preset"),
        (("b90", 4, "A"), "variant"),
        (("b90", 1, "C"), "reward_state"),
    ],
)
def test_environment_config_rejects_unknown_conditions(args, match):
    with pytest.raises(ValueError, match=match):
        environment_config(*args)


def test_next_token_targets_aligns_logits_with_following_observation():
    # Observation layout: 2-column delayed-token one-hot then 3-column
    # executed-action one-hot; only the token slice supervises.
    observations = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
            ],
        ]
    )
    logits = torch.randn(2, 4, TOKEN_COUNT)
    aux_logits, targets, valid = next_token_targets(
        {Columns.OBS: observations},
        logits,
    )
    torch.testing.assert_close(aux_logits, logits[:, :-1, :])
    assert targets.tolist() == [[0, 1, 0], [1, 0, 1]]
    # Position t is supervised on the token revealed at t+1; unpopulated
    # (all-zero) next observations are dropped.
    assert valid.tolist() == [[True, True, True], [True, False, True]]


def test_next_token_targets_respects_loss_mask():
    observations = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        ]
    )
    loss_mask = torch.tensor([[True, True, False, True]])
    logits = torch.randn(1, 4, TOKEN_COUNT)
    _, _, valid = next_token_targets(
        {Columns.OBS: observations, Columns.LOSS_MASK: loss_mask},
        logits,
    )
    assert valid.tolist() == [[True, False, False]]


def test_aux_ce_ppo_config_wiring(tmp_path, monkeypatch):
    monkeypatch.setattr(pusher_shared, "available_cpus", lambda: 64)
    context = _context(tmp_path)
    config = build_config(
        context,
        preset="b10",
        variant=3,
        reward_state="A",
        next_token_aux=True,
    )
    assert (
        config.rl_module_spec.module_class is ActorCriticWithNextTokenAux
    )
    assert config.rl_module_spec.model_config["next_token_aux"] == {
        "num_classes": TOKEN_COUNT
    }
    assert config.learner_class is PPOWithNextTokenAux
    learner_config = config.learner_config_dict
    assert learner_config["next_token_aux/lambda"] == NEXT_TOKEN_AUX_COEFFICIENT
    assert (
        learner_config["next_token_aux/target_extractor"]
        is next_token_targets
    )

    recipe = resolved_recipe(
        context,
        preset="b10",
        variant=3,
        reward_state="A",
        next_token_aux=True,
    )
    assert recipe["next_token_aux"] is True
    assert recipe["next_token_aux_coefficient"] == NEXT_TOKEN_AUX_COEFFICIENT


def test_aux_ce_leaf_module_is_importable_and_bound(tmp_path):
    module_name = (
        "experiments.pusher_b_reward_state_action_symmetry_cycle_1."
        "b10_reward_a.variant_3_aux_ce.experiment"
    )
    module = importlib.import_module(module_name)
    assert callable(module.run)
    config = module.build_config(_context(tmp_path))
    assert config.env_config == environment_config("b10", 3, "A")
    assert (
        config.rl_module_spec.module_class is ActorCriticWithNextTokenAux
    )
    assert config.learner_class is PPOWithNextTokenAux
