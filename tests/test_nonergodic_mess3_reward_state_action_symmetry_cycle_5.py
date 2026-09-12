from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.connectors.common import AddTimeDimToBatchAndZeroPad
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from envs.hmm import HMMEnv, condition_edge
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.alpha095.process import (
    COMPONENT_PARAMETERS as ALPHA095_COMPONENT_PARAMETERS,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5 import (
    analysis,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.design import (
    bayes_observer_reward_audit,
    constant_action_expected_return,
    controlled_kernels,
    expected_next_reward_scores,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.process import (
    COMPONENT_COUNT,
    CONTEXT_LENGTH,
    EFFECT_SIZE,
    EPISODE_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    TOKEN_COUNT,
    environment_config,
    nonergodic_mess3_model,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.shared import (
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
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.task import (
    DIRECTIONS,
    N_ACTIONS,
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
    REWARD_STATE,
    ActionSymmetryTask,
)
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
from harness.hardware import PROFILES


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def _module() -> FactoredReproductionActorCritic:
    return FactoredReproductionActorCritic(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(TOKEN_COUNT + N_ACTIONS,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(N_ACTIONS),
        model_config=dict(MODEL_CONFIG),
    ).eval()


def _diagnostic_config(variant: int) -> dict[str, object]:
    config = environment_config(variant)
    config["diagnostics"] = {
        "belief": True,
        "state": True,
        "tokens": True,
        "transitions": True,
    }
    return config


def test_dual_model_is_block_diagonal_and_component_is_fixed_per_episode():
    model = nonergodic_mess3_model()
    assert model.n_states == STATE_COUNT == 6
    assert model.n_tokens == TOKEN_COUNT == 3
    np.testing.assert_allclose(
        model.initial_distribution.reshape(
            COMPONENT_COUNT,
            STATES_PER_COMPONENT,
        ).sum(axis=1),
        [0.5, 0.5],
    )
    np.testing.assert_array_equal(
        model.transition_matrix[:3, 3:],
        np.zeros((3, 3)),
    )
    np.testing.assert_array_equal(
        model.transition_matrix[3:, :3],
        np.zeros((3, 3)),
    )

    env = HMMEnv(_diagnostic_config(3))
    try:
        _, info = env.reset(seed=19)
        component = info["state_current"] // STATES_PER_COMPONENT
        for step in range(EPISODE_LENGTH):
            _, _, _, truncated, info = env.step(step % N_ACTIONS)
            assert info["state_current"] // STATES_PER_COMPONENT == component
        assert truncated
        components = []
        for _ in range(64):
            _, info = env.reset()
            components.append(info["state_current"] // STATES_PER_COMPONENT)
        assert set(components) == {0, 1}
    finally:
        env.close()


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_bos_and_delayed_token_action_observation(variant):
    env = HMMEnv(_diagnostic_config(variant))
    try:
        observation, info = env.reset(seed=23)
        np.testing.assert_array_equal(
            observation,
            np.zeros(TOKEN_COUNT + N_ACTIONS, dtype=np.float32),
        )
        pending_token = int(info["raw_token_current"])
        observation, _, _, _, info = env.step(POSITIVE_ACTION)
        np.testing.assert_array_equal(
            observation[:TOKEN_COUNT],
            np.eye(TOKEN_COUNT, dtype=np.float32)[pending_token],
        )
        np.testing.assert_array_equal(
            observation[TOKEN_COUNT:],
            np.eye(N_ACTIONS, dtype=np.float32)[POSITIVE_ACTION],
        )
        assert info["visible_source_token"] == pending_token
    finally:
        env.close()


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_action_conditioned_edges_match_controlled_transitions(variant):
    model = nonergodic_mess3_model()
    task = ActionSymmetryTask(model=model, variant=variant)
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
        np.testing.assert_allclose(edges.sum(axis=0), transition, atol=1e-14)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            transition[:3, 3:],
            np.zeros((3, 3)),
        )
        np.testing.assert_array_equal(
            transition[3:, :3],
            np.zeros((3, 3)),
        )
        support = model.edge_transition_matrices > 0.0
        reference_conditionals = np.divide(
            model.edge_transition_matrices,
            model.transition_matrix[None, :, :],
            out=np.zeros_like(model.edge_transition_matrices),
            where=model.transition_matrix[None, :, :] > 0.0,
        )
        controlled_conditionals = np.divide(
            edges,
            transition[None, :, :],
            out=np.zeros_like(edges),
            where=transition[None, :, :] > 0.0,
        )
        np.testing.assert_allclose(
            controlled_conditionals[support],
            reference_conditionals[support],
            atol=1e-14,
        )


def test_variants_apply_the_cycle_5_direction_tables_in_each_component():
    model = nonergodic_mess3_model()
    for variant, directions in DIRECTIONS.items():
        task = ActionSymmetryTask(model=model, variant=variant)
        for component in range(COMPONENT_COUNT):
            start = component * STATES_PER_COMPONENT
            block = slice(start, start + STATES_PER_COMPONENT)
            noop = task.transition_matrix_for_action(NOOP_ACTION)[
                block,
                start + REWARD_STATE,
            ]
            positive = task.transition_matrix_for_action(POSITIVE_ACTION)[
                block,
                start + REWARD_STATE,
            ]
            negative = task.transition_matrix_for_action(NEGATIVE_ACTION)[
                block,
                start + REWARD_STATE,
            ]
            for state in range(STATES_PER_COMPONENT):
                if noop[state] == 0.0:
                    assert positive[state] == negative[state] == 0.0
                    continue
                expected_positive = np.sign(
                    directions[state, POSITIVE_ACTION]
                )
                expected_negative = np.sign(
                    directions[state, NEGATIVE_ACTION]
                )
                assert np.sign(positive[state] - noop[state]) == (
                    expected_positive
                )
                assert np.sign(negative[state] - noop[state]) == (
                    expected_negative
                )


def test_reward_uses_pre_transition_local_state():
    env = HMMEnv(_diagnostic_config(2))
    try:
        env.reset(seed=31)
        for action in [0, 1, 2] * 20:
            _, reward, _, _, info = env.step(action)
            expected = float(
                info["state_before"] % STATES_PER_COMPONENT == REWARD_STATE
            )
            assert reward == expected
    finally:
        env.close()


def test_expected_next_reward_scores_use_belief_not_hidden_state():
    transitions, _ = controlled_kernels(3)
    belief = np.asarray([0.08, 0.12, 0.30, 0.20, 0.10, 0.20])
    scores = expected_next_reward_scores(belief, transitions)
    reward = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        scores,
        [belief @ transition @ reward for transition in transitions],
    )
    assert scores.shape == (N_ACTIONS,)


def test_constant_action_returns_are_exact_finite_horizon_expectations():
    transitions, _ = controlled_kernels(2)
    belief = (
        nonergodic_mess3_model().initial_distribution
        @ transitions[NOOP_ACTION]
    )
    reward = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    expected = 0.0
    for _ in range(EPISODE_LENGTH):
        expected += belief @ reward
        belief = belief @ transitions[POSITIVE_ACTION]
    assert constant_action_expected_return(2, POSITIVE_ACTION) == (
        pytest.approx(expected)
    )


def test_bayes_observer_reward_audit_certifies_the_control_ladder():
    audits = {
        variant: bayes_observer_reward_audit(
            variant,
            n_episodes=4_096,
            seed=20_260_920 + variant,
        )
        for variant in (1, 2, 3)
    }
    assert EFFECT_SIZE == 3.0
    assert audits[1]["action_fractions"] == [0.0, 1.0, 0.0]
    assert not audits[1]["nonconstant_optimum_certificate"]

    variant_2_fractions = np.asarray(audits[2]["action_fractions"])
    assert variant_2_fractions[NOOP_ACTION] > 0.20
    assert variant_2_fractions[POSITIVE_ACTION] > 0.65
    assert variant_2_fractions[NEGATIVE_ACTION] == 0.0
    assert audits[2]["advantage_normal_95_interval"][0] > 1.5
    assert audits[2]["nonconstant_optimum_certificate"]

    variant_3_fractions = np.asarray(audits[3]["action_fractions"])
    assert np.all(variant_3_fractions > 0.25)
    assert audits[3]["advantage_normal_95_interval"][0] > 6.5
    assert audits[3]["nonconstant_optimum_certificate"]


def test_original_effect_collapses_variant_2_to_constant_positive_action():
    audit = bayes_observer_reward_audit(
        2,
        effect_size=1.5,
        n_episodes=2_048,
        seed=20_260_930,
    )
    assert audit["action_fractions"] == [0.0, 1.0, 0.0]
    assert not audit["nonconstant_optimum_certificate"]


def test_delay_one_transducer_uses_preceding_edges_and_current_action():
    env = HMMEnv(_diagnostic_config(3))
    target = analysis.ActionConditionedTransducerTarget(env.model, 1)
    try:
        observation, info = env.reset(seed=37)
        reset_targets = target(
            observation[None],
            [info],
            np.asarray([0]),
        )
        np.testing.assert_allclose(
            reset_targets["weighted_belief"][0],
            info["belief_current"],
            atol=1e-12,
        )
        source = env.model.initial_distribution.copy()
        pending_edges = env.model.edge_transition_matrices
        for episode_step, action in enumerate([1, 2, 0, 1], start=1):
            observation, _, _, _, info = env.step(action)
            visible_token = int(info["visible_token_current"])
            source = condition_edge(source, pending_edges, visible_token)
            expected = source @ info["executed_transition_matrix"]
            pending_edges = info["executed_edge_transition_matrices"]
            info["requested_action"] = (action + 1) % N_ACTIONS
            targets = target(
                observation[None],
                [info],
                np.asarray([episode_step]),
            )
            np.testing.assert_allclose(
                targets["weighted_belief"][0],
                expected,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                targets["weighted_belief"][0],
                info["belief_current"],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                targets["pending_token_distribution"][0],
                np.einsum("i,yij->y", source, pending_edges),
                atol=1e-12,
            )
    finally:
        env.close()


def test_complete_episode_ppo_config_and_recipe(tmp_path):
    context = _context(tmp_path)
    config = build_config(context, 3)
    assert config is not build_config(context, 3)
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
    assert config.env_config == environment_config(3)
    assert config.env_config["delay"] == 1
    assert config.env_config["episode_length"] == EPISODE_LENGTH == 127
    assert config.env_config["randomize_first_episode_length"] is False
    assert config.env_runner_cls is FreshEpisodeSingleAgentEnvRunner
    assert config.rollout_fragment_length == "auto"
    assert config.batch_mode == "complete_episodes"
    assert config.rl_module_spec.module_class is FactoredReproductionActorCritic
    assert config.rl_module_spec.model_config == MODEL_CONFIG
    assert MODEL_CONFIG["context_length"] == CONTEXT_LENGTH == 128
    assert MODEL_CONFIG["max_seq_len"] == CONTEXT_LENGTH
    assert MODEL_CONFIG["training_sequence_mode"] == "complete_episode"

    recipe = resolved_recipe(context, 3)
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 1_024
    assert recipe["effect_size"] == EFFECT_SIZE == 3.0
    assert recipe["previous_action_in_observation"] is True
    assert recipe["environment"] == environment_config(3)
    assert recipe["environment"]["task"]["kwargs"]["effect_size"] == (
        EFFECT_SIZE
    )
    assert recipe["variant_directions"] == DIRECTIONS[3].tolist()
    full_context = replace(context, smoke=False)
    full_config = build_config(full_context, 3)
    assert full_config.train_batch_size_per_learner == TRAIN_BATCH_SIZE
    assert full_config.minibatch_size == MINIBATCH_SIZE
    assert resolved_recipe(full_context, 3)["total_env_steps"] == (
        TOTAL_ENV_STEPS
    )


def test_model_dimensions_and_complete_episode_sequence():
    torch.manual_seed(41)
    module = _module()
    state = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in module.get_initial_state().items()
    }
    batch = {
        Columns.OBS: torch.zeros(
            (1, 2, TOKEN_COUNT + N_ACTIONS),
        ),
        Columns.STATE_IN: state,
    }
    outputs = module.forward_train(batch)
    assert outputs[Columns.ACTION_DIST_INPUTS].shape == (1, 2, N_ACTIONS)
    assert module.compute_values(batch).shape == (1, 2)

    connector = AddTimeDimToBatchAndZeroPad(as_learner_connector=True)
    observations = [
        np.zeros(TOKEN_COUNT + N_ACTIONS, dtype=np.float32)
        for _ in range(EPISODE_LENGTH + 1)
    ]
    episode = SingleAgentEpisode(
        observations=observations,
        actions=[0] * EPISODE_LENGTH,
        rewards=[0.0] * EPISODE_LENGTH,
        len_lookback_buffer=0,
    )
    key = (episode.id_,)
    output = connector(
        rl_module=module,
        batch={Columns.OBS: {key: observations[:-1]}},
        episodes=[episode],
        shared_data={},
    )
    assert np.asarray(output[Columns.OBS][key]).shape == (
        1,
        CONTEXT_LENGTH,
        TOKEN_COUNT + N_ACTIONS,
    )
    assert output[Columns.SEQ_LENS][key] == [EPISODE_LENGTH]
    assert int(np.asarray(output[Columns.LOSS_MASK][key]).sum()) == (
        EPISODE_LENGTH
    )


def test_probe_targets_match_environment_diagnostics():
    data = analysis.collect_probe_data(
        _module(),
        variant=3,
        n_steps=24,
        seed=np.random.SeedSequence(43),
        device=torch.device("cpu"),
        policy_mode="random",
    )
    assert data.activations.shape == (24, 4, 128)
    assert data.weighted_beliefs.shape == (24, STATE_COUNT)
    assert data.component_posteriors.shape == (24, COMPONENT_COUNT)
    assert data.reward_state_beliefs.shape == (24, 1)
    assert data.antisymmetric_beliefs.shape == (24, 1)
    assert data.pending_token_distributions.shape == (24, TOKEN_COUNT)
    np.testing.assert_allclose(
        data.weighted_beliefs,
        data.diagnostic_beliefs,
        atol=1e-12,
    )
    np.testing.assert_allclose(data.weighted_beliefs.sum(axis=1), 1.0)
    np.testing.assert_allclose(data.component_posteriors.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        data.pending_token_distributions.sum(axis=1),
        1.0,
    )


@pytest.mark.parametrize("action", [-1, 3, 1.5, True, "1"])
def test_task_rejects_invalid_actions(action):
    env = HMMEnv(environment_config(1))
    try:
        env.reset(seed=47)
        with pytest.raises(ValueError, match="action"):
            env.step(action)
    finally:
        env.close()


def test_component_parameters_override_reaches_model_and_recipe(tmp_path):
    context = _context(tmp_path)
    config = build_config(
        context,
        3,
        component_parameters=ALPHA095_COMPONENT_PARAMETERS,
    )
    expected_kwargs = {
        "component_parameters": [
            dict(parameters)
            for parameters in ALPHA095_COMPONENT_PARAMETERS
        ]
    }
    assert config.env_config["model"]["kwargs"] == expected_kwargs

    env = HMMEnv(config.env_config)
    try:
        reference = nonergodic_mess3_model()
        expected = nonergodic_mess3_model(
            component_parameters=ALPHA095_COMPONENT_PARAMETERS
        )
        np.testing.assert_allclose(
            env.model.transition_matrix,
            expected.transition_matrix,
        )
        assert not np.allclose(
            env.model.transition_matrix,
            reference.transition_matrix,
        )
    finally:
        env.close()

    recipe = resolved_recipe(
        context,
        3,
        component_parameters=ALPHA095_COMPONENT_PARAMETERS,
    )
    assert recipe["components"] == expected_kwargs["component_parameters"]
    assert recipe["environment"]["model"]["kwargs"] == expected_kwargs


def test_alpha095_components_validate_and_stay_within_bounds():
    model = nonergodic_mess3_model(
        component_parameters=ALPHA095_COMPONENT_PARAMETERS
    )
    np.testing.assert_allclose(
        model.transition_matrix.sum(axis=1),
        np.ones(STATE_COUNT),
    )
    np.testing.assert_allclose(
        model.initial_distribution.sum(),
        1.0,
    )
