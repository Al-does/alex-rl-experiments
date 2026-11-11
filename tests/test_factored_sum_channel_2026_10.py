from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from experiments.factored_sum_channel_n3_2026_10 import heldout
from experiments.factored_sum_channel_n3_2026_10.process import SumChannel, mess3_edges
from experiments.pusher_b_jax_2026_10 import model as tm


def small_spec(channel: SumChannel) -> tm.ModelSpec:
    return tm.ModelSpec(
        vocab=channel.vocab_size, d_model=32, n_layers=1, n_heads=2, d_mlp=64
    )


def test_mess3_edges_are_stochastic():
    np.testing.assert_allclose(mess3_edges().sum(axis=(0, 2)), 1.0)


@pytest.mark.parametrize("n", [3, 4])
def test_sum_channel_vocab_and_mass(n):
    channel = SumChannel(n_factors=n, epsilon=0.5)
    edges = channel.edges
    assert edges.shape == (3**n + 3, 3**n, 3**n)
    assert channel.vocab_size == 3**n + 4
    np.testing.assert_allclose(edges.sum(axis=(0, 2)), 1.0)
    np.testing.assert_allclose(edges[3**n :].sum(axis=(0, 2)), 0.5)
    control = SumChannel(n_factors=n, epsilon=0.0)
    assert control.edges[3**n :].sum() == 0.0


def test_exact_filter_matches_jax_predictive():
    channel = SumChannel(n_factors=3)
    raw = np.asarray(channel.sample_tokens(jax.random.key(0), 8))
    beliefs, nll = channel.run_filter(raw)
    np.testing.assert_allclose(beliefs[:, 0], np.broadcast_to(channel.initial, (8, 27)))
    predictive = np.asarray(channel.make_env().predictive_distributions(raw))
    expected = -np.log(np.take_along_axis(predictive, raw[..., None], -1)[..., 0])
    np.testing.assert_allclose(nll, expected, atol=1e-4)


def test_factored_filter_is_exact_only_without_coupling():
    control = SumChannel(n_factors=3, epsilon=0.0)
    raw = np.asarray(control.sample_tokens(jax.random.key(1), 16))
    np.testing.assert_allclose(
        control.run_filter(raw, project=True)[1], control.run_filter(raw)[1], atol=1e-10
    )
    coupled = SumChannel(n_factors=3, epsilon=0.5)
    raw = np.asarray(coupled.sample_tokens(jax.random.key(1), 64))
    beliefs, exact = coupled.run_filter(raw)
    assert coupled.run_filter(raw, project=True)[1].mean() > exact.mean() + 1e-3
    residual = beliefs - coupled.product_of_marginals(beliefs)
    assert np.abs(residual).max() > 1e-2


def test_ladder_schedule_and_splits():
    config = heldout.HeldoutConfig()
    assert config.total_updates == 6_152
    assert len(config.eval_steps) == 36
    for fraction in heldout.HELDOUT_LADDER:
        replace(config, heldout_fraction=fraction).validate()
    assert replace(config, heldout_fraction=0.80).train_sequences == 26_214


def test_kelly_arm_shares_trunk_init():
    channel = SumChannel(n_factors=3)
    spec = small_spec(channel)
    key = jax.random.key(0)
    ce = heldout.init_params(spec, channel, key, kelly=False)
    kelly = heldout.init_params(spec, channel, key, kelly=True)
    assert kelly["kelly"]["w"].shape == (32, channel.token_count)
    for a, b in zip(jax.tree.leaves(ce), jax.tree.leaves({k: kelly[k] for k in ce})):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("n", "eps", "kelly"), [(3, 0.5, 0.0), (3, 0.5, 1.0), (4, 0.5, 1.0), (3, 0.0, 1.0)]
)
def test_heldout_smoke_runs(n, eps, kelly):
    channel = SumChannel(n_factors=n, epsilon=eps)
    config = heldout.HeldoutConfig.smoke(heldout_fraction=0.80, kelly_weight=kelly)
    saved = []
    result = heldout.train(
        channel,
        seed=0,
        config=config,
        spec=small_spec(channel),
        log=lambda _: None,
        on_checkpoint=lambda step, params: saved.append(step),
    )
    assert saved == list(config.eval_steps)
    refs = result["references"]
    assert refs["heldout_factored_gap_nats"] >= -1e-9
    for split in ("train", "heldout"):
        assert np.isfinite(result[f"{split}_probe_joint_1_minus_r2"])
        assert np.isfinite(result[f"{split}_probe_marginals_1_minus_r2"])
        assert np.isfinite(result[f"{split}_probe_residual_1_minus_r2"]) == (eps > 0)
        assert 0.0 < result[f"{split}_guess_hit_rate"] < 1.0
        assert result[f"{split}_effective_dimension"] >= 1
        assert (f"{split}_kelly_wager_mean" in result) == (kelly > 0)
