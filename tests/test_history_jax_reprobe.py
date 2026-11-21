import jax
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import env, models, probe
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared
from experiments.belief_quotient_history_jax_lumped_temp15_2026_11 import reprobe


@pytest.mark.parametrize("variant", [1, 2])
def test_coarse_filter_distribution_and_distinct_information(variant):
    process = shared.ladder_process(variant)
    tokens = np.array([[0, 1, 2, 0, 2, 1], [2, 0, 0, 1, 2, 2]])
    actions = np.array([[1, 2, 0, 1, 2, 0], [2, 1, 0, 2, 1, 0]])
    coarse = reprobe.coarse_filter(process, tokens, actions)
    np.testing.assert_allclose(coarse.sum(-1), 1)
    assert np.all((coarse >= 0) & (coarse <= 1))
    fine = np.asarray([env.belief_reset(process, y) for y in tokens[:, 0]])
    np.testing.assert_allclose(coarse[:, 0, 1], fine[:, 2], atol=1e-7)
    for t in range(1, tokens.shape[1]):
        fine = np.asarray([env.belief_update(process, b, a, y)
                           for b, a, y in zip(fine, actions[:, t-1], tokens[:, t])])
        np.testing.assert_allclose(coarse[:, t, 1], fine[:, 2], atol=1e-7)
    asymmetric = process._replace(initial=jax.numpy.array([.6, .1, .3]))
    coarse_asymmetric = reprobe.coarse_filter(asymmetric, tokens, actions)
    fine_asymmetric = np.asarray([env.belief_reset(asymmetric, y) for y in tokens[:, 0]])
    np.testing.assert_allclose(coarse_asymmetric.sum(-1), 1)
    assert np.max(abs(coarse_asymmetric[:, 0, 1] - fine_asymmetric[:, 2])) > .01
    # Equal identity probabilities within not-C make coarsening lossless.
    emission = np.asarray(process.emission).copy()
    emission[:, :2] = emission[:, :2].sum(-1, keepdims=True) / 2
    process = process._replace(emission=jax.numpy.asarray(emission))
    coarse = reprobe.coarse_filter(process, tokens, actions)
    fine = np.asarray([env.belief_reset(process, y) for y in tokens[:, 0]])
    np.testing.assert_allclose(coarse[:, 0, 1], fine[:, 2], atol=1e-7)
    for t in range(1, tokens.shape[1]):
        fine = np.asarray([env.belief_update(process, b, a, y)
                           for b, a, y in zip(fine, actions[:, t-1], tokens[:, t])])
        np.testing.assert_allclose(coarse[:, t, 1], fine[:, 2], atol=1e-7)


def test_reject_nonlumpable_and_empty_inputs():
    with pytest.raises(ValueError, match="not exactly lumpable"):
        reprobe.quotient_matrices(shared.ladder_process(3))
    with pytest.raises(ValueError, match="nonempty"):
        reprobe.coarse_filter(shared.ladder_process(2), np.empty((1, 0)), np.empty((1, 0)))


def test_reprobe_matches_archived_probe_and_warmup_alignment():
    process = shared.ladder_process(2)
    spec = shared.model_spec("memoryless", process)
    params = models.init_params(spec, jax.random.PRNGKey(1))
    key = jax.random.fold_in(jax.random.PRNGKey(42), 2)
    original = probe.probe_checkpoint(spec, process, params, key, smoke=True,
                                      episode_length=32, representation="embedding", targets=shared.belief_targets)
    result = reprobe.reprobe_checkpoint(spec, process, params, key, smoke=True, episode_length=32)
    assert result["targets"]["belief"] == original["targets"]["belief"]
    assert result["reward_per_step"] == original["policy"]["reward_per_step"]
    assert set(result["targets"]) == set(reprobe.TARGET_DEFINITIONS)
    without = reprobe.reprobe_checkpoint(spec, process, params, key, smoke=True, episode_length=32, coarse=False)
    assert "coarse_reward_class" not in without["targets"]
