import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.two_factor_factor2_suppression_steering_eps02_jax_2026_11 import steering as s
from experiments.two_factor_factor2_suppression_steering_eps02_jax_2026_11.summarize import summarize


def test_targets_preserve_factor_one_and_suppress_factor_two():
    belief = np.array([.2, .3, .5])[:, None] * np.array([.6, .1, .3])[None, :]
    for target, decoded in (("factored", [[.2, .3, .6, .1]]), ("joint", [belief.ravel()[:8]])):
        out = s.suppression_target(jnp.asarray(decoded), target)
        factors = s.base.decoded_factor_beliefs(out, "marginal" if target == "factored" else target)
        np.testing.assert_allclose(factors[0, 0], [.2, .3, .5], atol=1e-6)
        np.testing.assert_allclose(factors[0, 1], [0, 1, 0], atol=1e-6)


def test_selection_and_empty_summary():
    sweep = {str(x): {"factor_reward": [0.5, .6 - .1 * x]} for x in s.base.STRENGTHS}
    assert s.select_strength(sweep, .5) == {"strength": 1., "reached_random_baseline": True}
    assert s.select_strength(sweep, 0.) == {"strength": 3., "reached_random_baseline": False}
    with pytest.raises(ValueError, match="no reports"):
        summarize([])


def test_summary_seed_standard_error_and_duplicate_rejection():
    reports = [{"source": {"seed": seed}, "confirmation": {"intact": {
        "factor_reward": [value, value], "total_reward": 2 * value}},
        "strength_tuning": {"selected": {t: {"reached_random_baseline": seed == 42}
                                         for t in s.TARGETS}}}
        for seed, value in ((42, .2), (43, .4))]
    result = summarize(reports)
    np.testing.assert_allclose(result["across_agent"]["intact"]["F1"]["mean"], .3)
    np.testing.assert_allclose(result["across_agent"]["intact"]["F1"]["seed_se"], .1)
    assert result["reached_random_baseline"] == {"factored": 1, "joint": 1}
    assert summarize(reports[:1])["across_agent"]["intact"]["F1"]["seed_se"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize([reports[0], reports[0]])


def test_zero_steering_null_norm_and_complete_episode_rewards():
    spec = s.base.models.ModelSpec(kind="transformer", obs_dim=18, n_actions=9, width=16,
                                   n_layers=1, context_len=4, chunk_len=4)
    params = s.base.models.init_params(spec, jax.random.PRNGKey(0))
    rng = np.random.default_rng(1)
    decoder = rng.normal(size=(16, 4))
    steer = {"decoder": jnp.asarray(decoder), "bias": jnp.zeros(4),
             "embedding": jnp.asarray(rng.normal(size=(4, 16))),
             "null_direction": jnp.asarray(s.base.null_direction(decoder, 2)),
             "strength": jnp.float32(.8), "use_null": jnp.bool_(False)}
    x = jnp.asarray(rng.normal(size=(3, 16)))
    shift = s.steering_shift(x, steer, "factored")
    null = s.steering_shift(x, {**steer, "use_null": jnp.bool_(True)}, "factored")
    np.testing.assert_allclose(np.linalg.norm(shift, axis=1), np.linalg.norm(null, axis=1), rtol=1e-5)
    np.testing.assert_allclose(null @ decoder, 0, atol=1e-5)
    process = s.source.source_process()
    kwargs = dict(episodes=2, seed=2042, site=0, target="factored")
    intact = s.evaluate(spec, params, process, **kwargs)
    zero = s.evaluate(spec, params, process, **kwargs, policy="steered",
                      steer={**steer, "strength": jnp.float32(0)})
    assert intact == zero
    assert intact["factor_action_change_fraction"] == [0, 0]
    assert intact["total_reward"] == sum(intact["factor_reward"])
    with pytest.raises(ValueError, match="at least two"):
        s.evaluate(spec, params, process, episodes=1, seed=0)
