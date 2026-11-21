import json

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_encoder_nonlinearity_2026_11 import encoders
from experiments.reward_factors_encoder_nonlinearity_2026_11 import report, run


def test_encoder_architecture_validation_selection_and_displacement():
    rng = np.random.default_rng(42)
    x, xv, xt = (rng.uniform(-2, 2, (256, 1)) for _ in range(3))
    target = lambda a: np.c_[a, a ** 2, np.sin(2 * a)]
    h, hv, ht = map(target, (x, xv, xt))
    affine = encoders.fit(x, h, xv, hv, "affine", rcond=1e-6)
    mlp = encoders.fit(x, h, xv, hv, "mlp", max_epochs=300, patience=20, batch_size=32)
    assert mlp.diagnostics["hidden_width"] == 2 * h.shape[1]
    assert len(mlp.params) == 4
    assert encoders.r_squared(mlp(jnp.asarray(xt)), ht) > encoders.r_squared(affine(jnp.asarray(xt)), ht) + .1
    assert mlp.diagnostics["best_epoch"] <= mlp.diagnostics["stopped_epoch"] <= 300
    normalized_error = np.mean(((np.asarray(mlp(jnp.asarray(xv))) - hv) / h.std(0)) ** 2)
    np.testing.assert_allclose(normalized_error, mlp.diagnostics["best_validation_loss"], rtol=1e-5, atol=1e-7)
    for model in (affine, mlp):
        np.testing.assert_array_equal(model.displacement(jnp.asarray(xv), jnp.asarray(xv)), np.zeros_like(hv))
        np.testing.assert_allclose(model.displacement(jnp.asarray(xv), jnp.asarray(xt)), model(jnp.asarray(xt)) - model(jnp.asarray(xv)), atol=1e-6)
    np.testing.assert_allclose(affine.displacement(jnp.asarray(xv), jnp.asarray(xt)), (xt-xv) @ affine.params[0], atol=1e-6)


@pytest.mark.parametrize("bad", ["empty", "nan", "alignment", "dimensions", "kind", "budget"])
def test_encoder_input_errors(bad):
    x, h = np.ones((5, 1)), np.ones((5, 3))
    if bad == "empty":
        x = x[:0]
    if bad == "nan":
        x[0] = np.nan
    if bad == "alignment":
        h = h[:4]
    if bad == "dimensions":
        x = x[:, 0]
    with pytest.raises(ValueError):
        encoders.fit(x, h, np.ones((5, 1)), np.ones((5, 3)),
                     "unknown" if bad == "kind" else "mlp", max_epochs=0 if bad == "budget" else 1)
    assert encoders.r_squared(np.ones((5, 3)), np.ones((5, 3))) is None


def test_historical_lambda_selection():
    scores = {"0.0": .4, "0.5": .09, "1.0": .06}
    assert run.choose_strength((0., .5, 1.), scores, True, .1) == .5
    assert run.choose_strength((0., .5, 1.), scores, True, .01) == 1.
    assert run.choose_strength((0., .5, 1.), scores, False) == 0.
    assert len(run.ARMS) == 4


def test_report_complete_matrix_and_protocol_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.collect(tmp_path)
    first = None
    for test in run.TESTS:
        for seed in report.SEEDS:
            for rep in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
                for kind, source in run.ARMS:
                    path = report.paths(tmp_path, test, seed, rep, kind, source)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = dict(test=test, seed=seed, representation=rep, encoder=kind, source=source,
                               pilot=False, historical_encoder_source="exact", write_current_source="probe_decoded_unclipped",
                               tuning_episodes=16, confirmation_episodes=32, horizon=1024, held_episodes=12,
                               site=3, train_rows=12000 if test.startswith("ladder") else 60000,
                               code_hashes={"run.py": "a", "encoders.py": "b"})
                    path.write_text(json.dumps(row))
                    if first is None:
                        first = path
    rows, manifest = report.collect(tmp_path)
    assert len(rows) == len(manifest) == 96
    original = json.loads(first.read_text())
    first.write_text(json.dumps(dict(original, write_current_source="exact")))
    with pytest.raises(ValueError, match="write_current_source"):
        report.collect(tmp_path)
    first.write_text(json.dumps(dict(original, code_hashes={"run.py": "different"})))
    with pytest.raises(ValueError, match="mixed experiment code"):
        report.collect(tmp_path)


def test_aggregation_requires_exact_unchanged_runner_recheck_for_platform_drift(tmp_path):
    for test in run.TESTS:
        for seed in report.SEEDS:
            checkpoint = {"sha256": f"{test}-{seed}"}
            modes = ("2", "1") if test.startswith("ladder") else ("0",)
            metrics = {mode: dict(strength=1., reward=.5, factor1_reward=.2, actions_changed=.1) for mode in modes}
            common = dict(test=test, seed=seed, pilot=False, campaign=True, smoke=False, horizon=1024,
                          checkpoint=checkpoint, tuning_episodes=16, confirmation_episodes=32)
            prefix = "belief_quotient" if test.startswith("ladder") else "reward_factors"
            old_folder = tmp_path / f"experiments/{prefix}_causal_decoding_2026_11/results"
            old_folder.mkdir(parents=True, exist_ok=True)
            cpu_folder = tmp_path / "experiments/reward_factors_encoder_nonlinearity_2026_11/results"
            cpu_folder.mkdir(parents=True, exist_ok=True)
            for method in ("eq4", "eq4_joint", "intact", "map"):
                control = dict(common, method=method, metrics=metrics)
                if test == "moving" and method in ("intact", "map"):
                    control["metrics"] = {"0": dict(metrics["0"], reward=.25 if method == "intact" else .75)}
                    (cpu_folder / f"cpu_recheck_{test}_seed{seed}_{method}.json").write_text(json.dumps(control))
                (old_folder / f"campaign_{test}_seed{seed}_{method}.json").write_text(json.dumps(control))
            for rep in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
                for kind, source in run.ARMS:
                    path = report.paths(tmp_path, test, seed, rep, kind, source)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = dict(common, representation=rep, encoder=kind, source=source,
                               historical_encoder_source="exact", write_current_source="probe_decoded_unclipped",
                               site=3, held_episodes=12, train_rows=12000 if test.startswith("ladder") else 60000,
                               code_hashes={"run.py": "a", "encoders.py": "b"}, diagnostics={"test_r2": .8},
                               metrics=metrics, matched_lambda_metrics=metrics)
                    path.write_text(json.dumps(row))
    old_path = tmp_path / "experiments/reward_factors_causal_decoding_2026_11/results/campaign_moving_seed43_eq4.json"
    reference = json.loads(old_path.read_text())
    changed = json.loads(old_path.read_text())
    changed["metrics"]["0"]["reward"] = .501
    old_path.write_text(json.dumps(changed))
    with pytest.raises(FileNotFoundError):
        report.aggregate(tmp_path)
    cpu_path = cpu_folder / "cpu_recheck_moving_seed43_eq4.json"
    cpu_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="historical CPU recheck mismatch"):
        report.aggregate(tmp_path)
    cpu_path.write_text(json.dumps(reference))
    report.aggregate(tmp_path)
    manifest = json.loads((cpu_folder / "campaign_manifest.json").read_text())
    assert len(manifest["platform_differences"]) == 1
    summary = json.loads((cpu_folder / "campaign_summary.json").read_text())
    assert summary["metrics"]["matched_lambda_metrics/affine/exact/marginal"]["moving_gap_closed"]["per_seed"] == [.5] * 3
