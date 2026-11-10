import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import env as hmm
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as ladder
from experiments.reward_factors_history_jax_lumped_2026_11 import process as factors


@pytest.fixture
def updated_source(monkeypatch):
    from experiments.belief_quotient_history_REINFORCE_2026_11 import design
    from experiments.reward_factors_history_REINFORCE_2026_11 import task

    monkeypatch.setattr(design, "BASELINE_TRANSITION_MATRIX", ladder.BASELINE_TRANSITION_MATRIX)
    monkeypatch.setattr(task, "BASELINE_TRANSITION_MATRIX", ladder.BASELINE_TRANSITION_MATRIX)


def _source_env(config, diagnostics=False):
    from envs.hmm.env import HMMEnv

    config = dict(config)
    if diagnostics:
        config["diagnostics"] = {"belief": True, "state": True, "tokens": True}
    return HMMEnv(config)


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_ladder_process_matches_source_environment(variant, updated_source):
    from experiments.belief_quotient_history_REINFORCE_2026_11.shared import environment_config

    source = _source_env(environment_config(variant))
    process = ladder.ladder_process(variant)
    np.testing.assert_allclose(process.initial, source.model.initial_distribution, atol=1e-7)
    np.testing.assert_allclose(process.emission, source.model.emission_matrix, atol=1e-7)
    for action in range(3):
        expected = source.task.resolve_action(action, 0, source.model).transition_matrix
        np.testing.assert_allclose(process.transitions[action], expected, atol=1e-6)


@pytest.mark.parametrize("condition", factors.CONDITIONS)
def test_factor_process_matches_source_environment(condition, updated_source):
    from envs.hmm import TransitionEvent
    from experiments.reward_factors_history_REINFORCE_2026_11.process import environment_config

    source = _source_env(environment_config(condition))
    process = factors.factor_process(condition)
    np.testing.assert_allclose(process.emission, source.model.emission_matrix, atol=1e-7)
    np.testing.assert_allclose(process.initial, source.model.initial_distribution, atol=1e-7)
    for action in range(9):
        decision = source.task.resolve_action(action, 0, source.model)
        np.testing.assert_allclose(process.transitions[action], decision.transition_matrix, atol=1e-6)
        for state in range(9):
            event = TransitionEvent(step=0, state_before=state, state_after=0, raw_token_before=0,
                                    raw_token_after=0, belief_before=None, belief_after=None)
            reward, parts = source.task.reward(event, decision)
            assert process.reward[state] == pytest.approx(reward)
            assert list(np.asarray(process.reward_components[:, state])) == pytest.approx(
                [parts["factor_1_occupancy_reward"], parts["factor_2_occupancy_reward"]])


def test_filter_and_observation_match_source_diagnostics(updated_source):
    from experiments.reward_factors_history_REINFORCE_2026_11.process import environment_config

    source = _source_env(environment_config("reward_both"), diagnostics=True)
    process = factors.factor_process("reward_both")
    obs, info = source.reset(seed=3)
    state = hmm.EnvState(jnp.int32(info["state_current"]), jnp.int32(info["raw_token_current"]),
                         jnp.int32(-1))
    np.testing.assert_allclose(hmm.observe(process, state), obs)
    belief = hmm.belief_reset(process, state.token)
    rng = np.random.default_rng(0)
    for _ in range(200):
        np.testing.assert_allclose(belief, info["belief_current"], atol=1e-5)
        action = int(rng.integers(9))
        obs, _, _, truncated, info = source.step(action)
        if truncated:
            break
        state = hmm.EnvState(jnp.int32(info["state_current"]),
                             jnp.int32(info["raw_token_current"]), jnp.int32(action))
        np.testing.assert_allclose(hmm.observe(process, state), obs)
        belief = hmm.belief_update(process, belief, action, state.token)
