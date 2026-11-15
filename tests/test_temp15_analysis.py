import jax
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.within_class_belief_probes_2026_11.temp15_analyze import perturb, replay


@pytest.mark.parametrize("kind", ["gru", "memoryless", "transformer"])
def test_matched_replay_matches_sequential_steps_and_resets_episodes(kind):
    spec = models.ModelSpec(kind, 6, 3, 4, 2, context_len=4,
                            feature_count=3 if kind == "memoryless" else None)
    params = models.init_params(spec, jax.random.PRNGKey(4))
    obs = np.random.default_rng(0).normal(size=(2, 8, 6)).astype(np.float32)
    actual = np.asarray(replay(spec, params, obs))
    carry = models.init_carry(spec, 2)
    expected = []
    for t in range(8):
        _, _, h, carry = models.step(spec, params, carry, obs[:, t], jax.numpy.asarray(t))
        expected.append(np.asarray(h))
    np.testing.assert_allclose(actual, np.stack(expected, axis=1), atol=2e-5)
    first = np.asarray(replay(spec, params, obs[:1]))
    np.testing.assert_allclose(first, actual[:1], atol=2e-5)


def test_history_controls_keep_complete_pairs_and_reset_pools():
    tokens = np.eye(3)[np.array([[0, 1, 2, 0, 2], [1, 2, 0, 1, 1]])]
    actions = np.eye(3)[np.array([[0, 2, 1, 2, 0], [0, 1, 2, 1, 0]])]
    actions[:, 0] = 0
    obs = np.concatenate([tokens, actions], axis=2).astype(np.float32)
    shuffled = perturb(obs, "history_shuffle", 0)
    np.testing.assert_array_equal(shuffled[:, 0], obs[:, 0])
    for before, after in zip(obs[:, 1:], shuffled[:, 1:]):
        assert sorted(map(tuple, before)) == sorted(map(tuple, after))
    random = perturb(obs, "random_input", 0)
    for pair in random[:, 0]:
        assert tuple(pair) in set(map(tuple, obs[:, 0]))
    for pair in random[:, 1:].reshape(-1, 6):
        assert tuple(pair) in set(map(tuple, obs[:, 1:].reshape(-1, 6)))
    np.testing.assert_array_equal(perturb(obs, "random_input", 0), random)
    with pytest.raises(ValueError):
        perturb(obs, "unknown", 0)
