import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.training import _save
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11 import steering as s

NOOP, X, Y = 0, 1, 2


def _spec():
    process = s.source_process("reward_both")
    return models.architecture_spec("transformer", obs_dim=process.obs_dim,
                                    n_actions=process.n_actions, n_tokens=process.n_tokens,
                                    context_len=32)


def test_state_known_action_maps():
    assert s.myopic_table(2).argmax(1).tolist() == [X, Y, NOOP]
    assert s.discounted_table(2).argmax(1).tolist() == [X, Y, NOOP]
    assert s.myopic_table(0).argmax(1).tolist() == [Y, X, X]
    assert s.discounted_table(0).argmax(1).tolist() == [Y, Y, X]
    np.testing.assert_allclose(s.myopic_table(0), s.FACTOR_TRANSITIONS[:, :, 0].T)


def test_moved_process_changes_only_factor_2_reward():
    source, moved = s.source_process("reward_both"), s.moved_process()
    for name in ("initial", "transitions", "emission"):
        np.testing.assert_array_equal(getattr(source, name), getattr(moved, name))
    first, second = np.divmod(np.arange(9), 3)
    np.testing.assert_array_equal(moved.reward_components, np.stack([first == 2, second == 0]))
    np.testing.assert_array_equal(source.reward_components[0], moved.reward_components[0])


def test_forward_matches_model_step_and_params_round_trip(tmp_path):
    spec = _spec()
    params = models.init_params(spec, jax.random.PRNGKey(3))
    _save(tmp_path / "p.npz", params)
    restored = s.load_params(tmp_path / "p.npz", spec)
    carry_a = carry_b = models.init_carry(spec, 4)
    obs = jax.nn.one_hot(jnp.arange(4) % 9, spec.obs_dim)
    for t in map(jnp.int32, range(3)):
        logits_a, _, embedding, carry_a = models.step(spec, params, carry_a, obs, t)
        logits_b, sites, carry_b = s.forward(spec, restored, carry_b, obs, t)
        np.testing.assert_allclose(logits_a, logits_b, atol=1e-6)
        np.testing.assert_allclose(embedding, sites[:, -1], atol=1e-6)


def test_joint_target_preserves_factor_1_and_anchors_factor_2():
    rng = np.random.default_rng(0)
    joint = rng.dirichlet(np.ones(9), size=5)
    decoded = jnp.asarray(joint[:, :8], jnp.float32)
    anchor = jnp.asarray([2, 0, 1])
    table = jnp.asarray(s.myopic_table(0), jnp.float32)
    target = s.belief_target(decoded, "joint", table, anchor)
    factors = np.asarray(s.decoded_factor_beliefs(target, "joint"))
    np.testing.assert_allclose(factors[:, 0], joint.reshape(-1, 3, 3).sum(2), atol=1e-6)
    desired = np.asarray(s.best_factor_action(joint.reshape(-1, 3, 3).sum(1), table))
    np.testing.assert_allclose(factors[:, 1], np.eye(3)[np.asarray(anchor)[desired]], atol=1e-6)


def test_marginal_target_changes_only_factor_2():
    decoded = jnp.asarray([[0.2, 0.3, 0.6, 0.1]], jnp.float32)
    target = s.belief_target(decoded, "marginal", jnp.asarray(s.myopic_table(0)), jnp.asarray([2, 0, 1]))
    np.testing.assert_allclose(target[:, :2], decoded[:, :2])
    assert set(np.asarray(target[0, 2:]).tolist()) <= {0.0, 1.0}


def test_null_direction_is_decoder_orthogonal():
    decoder = np.random.default_rng(1).standard_normal((64, 8))
    direction = s.null_direction(decoder, 7)
    np.testing.assert_allclose(direction @ decoder, 0, atol=1e-10)
    assert np.isclose(np.linalg.norm(direction), 1)
