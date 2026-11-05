import itertools
import json

import numpy as np
import pytest
from harness.context import RunContext

from experiments.mrrxor_grokking_jax_2026_11 import process
from experiments.mrrxor_grokking_jax_2026_11.analysis import analyze, fit_pca
from experiments.mrrxor_grokking_jax_2026_11.ppo import PPOConfig, checkpoint_updates, train
from experiments.pusher_b_jax_2026_10.model import ModelSpec


def test_edges():
    emit = np.zeros((2, 5, 5))
    for x, s, t, p in [(0, 0, 1, .5), (1, 0, 2, .5), (0, 1, 4, .5),
                        (1, 1, 3, .5), (0, 2, 3, .5), (1, 2, 4, .5),
                        (1, 3, 0, 1), (0, 4, 0, 1)]:
        emit[x, s, t] = p
    jump = np.full((5, 5), .05 / 4)
    np.fill_diagonal(jump, .95)
    edges = process.transition_tensor()
    np.testing.assert_allclose(edges, np.einsum("su,xut->xst", jump, emit))
    np.testing.assert_allclose(edges.sum((0, 2)), 1)
    assert np.all(edges >= 0)
    pi = process.stationary_distribution(edges)
    np.testing.assert_allclose(pi @ edges.sum(0), pi)


def test_filter_brute_force():
    edges = process.transition_tensor()
    pi = process.stationary_distribution(edges)
    for tokens in itertools.product(range(2), repeat=3):
        actual = process.beliefs(np.array([tokens]))[0]
        for n in range(4):
            mass = np.zeros(5)
            for path in itertools.product(range(5), repeat=n + 1):
                p = pi[path[0]]
                for i in range(n):
                    p *= edges[tokens[i], path[i], path[i+1]]
                mass[path[-1]] += p
            np.testing.assert_allclose(actual[n], mass / mass.sum(), atol=1e-14)
    assert process.beliefs(np.empty((2, 0), dtype=int)).shape == (2, 1, 5)
    with pytest.raises(ValueError):
        process.beliefs(np.array([[2]]))


def test_pca():
    pca = fit_pca()
    assert pca["ntp_pc_predictive_r2"] == pytest.approx(1, abs=1e-10)
    b = process.beliefs(process.sample(np.random.default_rng(17), 128, 32))[:, :-1].reshape(-1, 5)
    pc = ((b - pca["mean"]) @ np.asarray(pca["components"]))[:, pca["ntp_pc_index"] - 1]
    design = np.stack([pc, np.ones_like(pc)], -1)
    target = process.predictive(b)[:, 1]
    pred = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(pred, target, atol=1e-12)


def test_reward_pools_and_schedule():
    np.testing.assert_array_equal(process.reward(np.array([0, 1, 0]), np.array([0, 0, 0])), [1, 0, 1])
    assert process.reward(np.array([], dtype=int), np.array([], dtype=int)).size == 0
    pools = process.fixed_pools(42, 16, 32)
    again = process.fixed_pools(42, 16, 32)
    np.testing.assert_array_equal(pools["train"], again["train"])
    assert not ({tuple(x) for x in pools["train"]} & {tuple(x) for x in pools["heldout"]})
    assert checkpoint_updates(20, 7) == [0, 1, 2, 5, 7, 10, 14, 20]
    with pytest.raises(ValueError):
        process.fixed_pools(42, 0, 10)
    with pytest.raises(ValueError):
        PPOConfig(gamma=.99)


def test_smoke_and_analysis(tmp_path):
    context = RunContext(experiment_dir=tmp_path, results_dir=tmp_path / "results",
                         artifacts_dir=tmp_path / "artifacts", smoke=True, upload_artifacts=False)
    config = PPOConfig.smoke(total_env_steps=2048)
    spec = ModelSpec(vocab=3, context_length=32, n_layers=1, d_model=16, n_heads=2, d_mlp=32)
    summary = train(context, config, spec)
    assert summary["env_steps"] == 2048
    assert summary["checkpoint_count"] == 3
    report = analyze(context.artifacts_dir, tmp_path / "report.json")
    assert (tmp_path / "report.png").is_file()
    assert len(report["checkpoints"]) == 3
    assert report["checkpoints"][0]["env_steps"] == 0
    entries = json.loads((context.artifacts_dir / "checkpoint_index.json").read_text())
    with np.load(context.artifacts_dir / entries[0]["path"]) as first, np.load(context.artifacts_dir / entries[-1]["path"]) as last:
        assert any(not np.array_equal(first[k], last[k]) for k in first.files)
    assert report["checkpoints"][-1]["metrics"]["heldout"]["random_bit_count"] > 0
