import itertools
import json

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from harness.context import RunContext

from experiments.mrrxor_grokking_jax_2026_11 import process
from experiments.mrrxor_grokking_jax_2026_11 import sweep
from experiments.mrrxor_grokking_jax_2026_11.analysis import analyze, fit_pca
from experiments.mrrxor_grokking_jax_2026_11.ppo import PPOConfig, checkpoint_updates, make_update, train
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


def test_sweep_grid_and_validation():
    recipe = sweep.grid_recipe()
    assert len(recipe.arms) == 4
    assert [a.name for a in recipe.arms] == ["n256", "n1024", "n4096", "n16384"]
    assert {(a.ppo.train_sequences, a.ppo.weight_decay, a.ppo.lr_schedule[0][1]) for a in recipe.arms} == {
        (n, 1., 3e-4) for n in (256, 1024, 4096, 16384)}
    assert PPOConfig().weight_decay == 1.
    assert PPOConfig().lr_schedule == ((0, 3e-4),)
    with pytest.raises(ValueError, match="at least one"):
        sweep.SweepRecipe(())
    with pytest.raises(ValueError, match="duplicate"):
        sweep.SweepRecipe((recipe.arms[0], recipe.arms[0]))
    with pytest.raises(ValueError, match="directory"):
        sweep.SweepRecipe((sweep.Arm("../bad", PPOConfig()),))


def test_sweep_smoke(tmp_path):
    context = RunContext(experiment_dir=tmp_path, results_dir=tmp_path,
                         artifacts_dir=tmp_path / "artifacts", smoke=True, upload_artifacts=False, seed=17)
    hits_before = make_update.cache_info().hits
    summary = sweep.run(context, sweep.smoke_recipe())
    assert len(summary["arms"]) == 2
    assert make_update.cache_info().hits > hits_before
    assert summary["arms"] == sorted(summary["arms"], key=lambda r: (
        r["heldout_random_bit_accuracy"], r["heldout_reward"]), reverse=True)
    for row in summary["arms"]:
        assert row["seed"] == 17
        assert row["env_steps"] == 2048
        assert np.isfinite(row["ntp_pc_probe_1_minus_r2"])
        report = json.loads((context.results_dir / row["arm"] / "analysis.json").read_text())
        assert len(report["checkpoints"]) == 3
        assert (context.results_dir / row["arm"] / "analysis.png").is_file()
        assert (context.results_dir / row["arm"] / "curve.json").is_file()
        assert (context.results_dir / row["arm"] / "artifacts" / "checkpoint_index.json").is_file()
    assert json.loads((context.results_dir / "summary.json").read_text())["arms"] == summary["arms"]
    assert "NTP-PC 1−R²" in (context.results_dir / "summary.md").read_text()


def test_sweep_run_paths_and_launcher(monkeypatch, tmp_path):
    from harness import cli

    root, artifacts = sweep.run_paths("layout-check")
    assert root == sweep.STUDY_DIR / "runs" / "layout-check"
    assert artifacts == root / "artifacts"
    assert sweep.run_paths("layout-check", smoke=True)[0] == sweep.STUDY_DIR / ".smoke" / "layout-check"
    with pytest.raises(ValueError):
        sweep.run_paths("../escape")
    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 0)
    assert sweep.main(["--run-id", "layout-check", "--no-upload-artifacts"]) == 0
    args = cli.build_parser().parse_args(calls[0])
    assert args.results_dir == root
    assert args.artifacts_dir == artifacts
    with pytest.raises(SystemExit):
        sweep.main(["--results-dir", str(tmp_path)])


def test_dynamic_adamw_matches_optax():
    params = {"w": jnp.array([1., -2.])}
    optimizer = optax.scale_by_adam()
    reference = optax.adamw(learning_rate=3e-4, weight_decay=.1)
    state, reference_state = optimizer.init(params), reference.init(params)
    reference_params = params
    for _ in range(3):
        grads = {"w": jnp.array([.2, -.4])}
        updates, state = optimizer.update(grads, state, params)
        updates = {"w": -3e-4 * (updates["w"] + .1 * params["w"])}
        expected, reference_state = reference.update(grads, reference_state, reference_params)
        params = optax.apply_updates(params, updates)
        reference_params = optax.apply_updates(reference_params, expected)
        np.testing.assert_allclose(params["w"], reference_params["w"], atol=1e-7)
