import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import env as hmm
from experiments.belief_quotient_history_jax_2026_11 import models, reinforce
from experiments.belief_quotient_history_jax_2026_11 import shared as ladder
from experiments.reward_factors_history_jax_2026_11 import process as factors
from experiments.reward_factors_history_jax_2026_11 import shared as factor_shared


def _source_env(config, diagnostics=False):
    from envs.hmm.env import HMMEnv

    config = dict(config)
    if diagnostics:
        config["diagnostics"] = {"belief": True, "state": True, "tokens": True}
    return HMMEnv(config)


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_ladder_process_matches_source_environment(variant):
    from experiments.belief_quotient_history_REINFORCE_2026_11.shared import environment_config

    source = _source_env(environment_config(variant))
    process = ladder.ladder_process(variant)
    np.testing.assert_allclose(process.initial, source.model.initial_distribution, atol=1e-7)
    np.testing.assert_allclose(process.emission, source.model.emission_matrix, atol=1e-7)
    for action in range(3):
        expected = source.task.resolve_action(action, 0, source.model).transition_matrix
        np.testing.assert_allclose(process.transitions[action], expected, atol=1e-6)


@pytest.mark.parametrize("condition", factors.CONDITIONS)
def test_factor_process_matches_source_environment(condition):
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


def test_filter_and_observation_match_source_diagnostics():
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


def test_jax_env_transition_frequencies():
    process = ladder.ladder_process(3)
    start = hmm.EnvState(jnp.int32(2), jnp.int32(0), jnp.int32(-1))
    keys = jax.random.split(jax.random.PRNGKey(0), 40_000)
    next_states = jax.vmap(lambda k: hmm.step(process, k, start, jnp.int32(1))[0].state)(keys)
    frequencies = np.bincount(np.asarray(next_states), minlength=3) / len(keys)
    np.testing.assert_allclose(frequencies, process.transitions[1, 2], atol=0.01)


@pytest.mark.parametrize("architecture,context,length", [
    ("transformer", 32, 96), ("transformer", 10, 64), ("memoryless", 32, 40),
    ("gru128", 32, 64), ("gru92", 32, 64),
])
def test_cached_rollout_matches_training_forward(architecture, context, length):
    process = factors.factor_process("reward_both")
    spec = models.architecture_spec(architecture, obs_dim=process.obs_dim, n_actions=9,
                                    n_tokens=9, context_len=context)
    if spec.kind == "gru":
        spec = models.ModelSpec(**{**spec.__dict__, "chunk_len": 16})
    params = models.init_params(spec, jax.random.PRNGKey(1))
    with jax.default_matmul_precision("highest"):
        records = reinforce.rollout(spec, process, params, jax.random.PRNGKey(2), 3, length)
        logits, _, _ = models.sequence(spec, params, records["obs"], records.get("chunk_starts"))
    expected = jnp.take_along_axis(jax.nn.log_softmax(logits), records["action"][..., None], -1)
    np.testing.assert_allclose(expected[..., 0], records["logp"], atol=2e-5)


def test_transformer_history_is_bounded_by_layers_times_context():
    spec = models.ModelSpec("transformer", 6, 3, 16, 2, 1, 4)
    params = models.init_params(spec, jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (1, 20, 6))
    changed = obs.at[0, 5].add(1.0)
    delta = jnp.abs(models.sequence(spec, params, obs)[0] - models.sequence(spec, params, changed)[0])
    delta = np.asarray(delta.max(-1)[0])
    assert np.all(delta[:5] == 0) and delta[13] > 0 and np.all(delta[14:] == 0)


def test_parameter_counts_match_pytorch_encoders_without_value_head():
    def count(architecture, obs_dim, n_actions, n_tokens, context):
        spec = models.architecture_spec(architecture, obs_dim=obs_dim, n_actions=n_actions,
                                        n_tokens=n_tokens, context_len=context)
        return models.parameter_count(models.init_params(spec, jax.random.PRNGKey(0)))

    h = 128
    gru = 6 * h + 2 * 2 * (3 * h * h + 3 * h) + 2 * h + h * 3 + 3 + h
    assert count("gru128", 6, 3, 3, 32) == gru
    assert count("memoryless", 6, 3, 3, 32) == 3 * 64 + 64 + 6 * (128 + 64 * 256 + 256 + 256 * 64 + 64) + 128 + 64 * 3 + 3


def test_returns_advantages_and_entropy_schedule():
    rewards = jnp.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    returns = reinforce.discounted_returns(rewards, 0.5)
    np.testing.assert_allclose(returns, [[1.25, 0.5, 1.0], [0.0, 0.0, 0.0]])
    advantages = reinforce.standardized_advantages(rewards, 0.5)
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(advantages.std()) == pytest.approx(1.0, abs=1e-5)
    for config in (ladder.train_config(False), factor_shared.train_config(False)):
        assert config.num_minibatches == 32 and config.num_envs % 32 == 0
        for steps in (0, 5_000_000, 30_000_000):
            assert reinforce.entropy_coefficient(config.entropy_schedule, steps) == pytest.approx(0.015)


def test_training_iteration_improves_surrogate_and_is_seeded():
    process = ladder.ladder_process(2)
    spec = ladder.model_spec("gru92", process)
    config = reinforce.TrainConfig(total_env_steps=4 * 64, num_envs=2, episode_length=64)
    first = reinforce.train(spec, process, config, seed=0)
    second = reinforce.train(spec, process, config, seed=0)
    np.testing.assert_array_equal(
        jax.tree_util.tree_leaves(first[0])[0], jax.tree_util.tree_leaves(second[0])[0])
    assert first[2]["iterations"] == 2 and np.isfinite(first[1][-1]["policy_loss"])


def test_minibatches_take_one_adam_step_each():
    process = ladder.ladder_process(3)
    spec = ladder.model_spec("transformer", process)
    single = reinforce.TrainConfig(total_env_steps=4 * 64, num_envs=4, episode_length=64)
    split = reinforce.TrainConfig(total_env_steps=4 * 64, num_envs=4, episode_length=64,
                                  num_minibatches=4)
    one = reinforce.train(spec, process, single, seed=0)[0]
    four = reinforce.train(spec, process, split, seed=0)[0]
    init = models.init_params(spec, jax.random.fold_in(jax.random.PRNGKey(0), 0))
    leaf = lambda tree: np.asarray(jax.tree_util.tree_leaves(tree)[-1])
    step_one = np.abs(leaf(one) - leaf(init)).max()
    step_four = np.abs(leaf(four) - leaf(init)).max()
    assert step_four > 2 * step_one
    with pytest.raises(ValueError):
        reinforce.TrainConfig(total_env_steps=1, num_envs=6, num_minibatches=4)
