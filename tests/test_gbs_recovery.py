"""Offline recovery selection, partial snapshots, diagnostics, and report outputs."""
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs import recovery, summarize


def make_fit(condition="variant_2", k=1, kind="affine", val=.0001, test=.0002):
    return {"condition": condition, "seed": 42, "control": False, "class": kind, "k": k,
            "selected": {"val_kl": val, "width": 32}, "test_kl": test, "status": "completed",
            "geometry": {target: {"affine_r2": .96, "mlp_r2": .99,
                                  "reverse_affine_r2": .96, "reverse_mlp_r2": .99}
                         for target in ("belief", "quotient", "factor_1", "factor_2")},
            "commutation": {"mse": .001, "constant_decoder_mse": .1},
            "hmm_matrices": None}


def store(root, fits, status="running", prefix="gbs-v3"):
    condition = fits[0]["condition"] if fits else "variant_2"
    directory = root/"results"/f"{prefix}-{condition}-policy"
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"run_manifest.json").write_text(json.dumps({"status": status, "git": {"commit": "test"}}))
    for fit in fits:
        (directory/f"seed{fit['seed']}_{fit['class']}_k{fit['k']}.json").write_text(json.dumps(fit))
    return directory


def group(report, condition="variant_2", seed=42, control=False):
    return next(g for g in report["groups"] if (g["condition"], g["seed"], g["control"]) == (condition, seed, control))


def test_elbow_validation_only_absolute_relative_tolerance_and_ties():
    curve = [{"k": 0, "val_kl": .2}, {"k": 1, "val_kl": .0109}, {"k": 2, "val_kl": .01}]
    assert recovery.elbow(curve)["k"] == 1
    assert recovery.elbow([{"k": 0, "val_kl": .109}, {"k": 1, "val_kl": .1}])["k"] == 0
    assert recovery.elbow([{"k": 0, "val_kl": None}, {"k": 1, "val_kl": float("nan")}]) is None
    assert recovery.elbow([]) is None
    rows = [{"class": "gru", "val_kl": .1, "test_kl": 0.},
            {"class": "affine", "val_kl": .1, "test_kl": 999.}]
    assert recovery.winner(rows)["class"] == "affine"


def test_complete_grid_selection_geometry_and_partial_assessment(tmp_path):
    fits = [make_fit(k=k, kind=kind, val=.5 if k == 0 else .0001, test=.5 if k == 0 else .0002)
            for kind in recovery.CLASSES for k in range(7)]
    store(tmp_path, fits)
    report = recovery.build_report([tmp_path])
    result = group(report)
    assert result["grid_complete"]
    assert result["selected"]["k"] == 1
    assert result["selected"]["class"] == "affine"
    assert result["verdict"] == "recovered"
    assert result["selected"]["commutation"]["mse"] == .001
    assert result["classes"]["hmm"]["selected"]["geometry"]["quotient"]["mlp_r2"] == .99
    assert result["envelope"][0]["classes_available"] == 4
    assert report["condition_verdicts"][1]["verdict"] == "partially recovered"
    assert report["condition_verdicts"][1]["provisional"]
    missing = group(report, seed=44)
    assert missing["selected"] is None
    assert missing["assessment"].startswith("pending")
    assert missing["fits_available"] == 0
    assert len(report["groups"]) == 30


def test_test_kl_truth_and_missing_classes_never_select(tmp_path):
    a = make_fit(k=0, val=.2, test=.2)
    b = make_fit(k=1, val=.0109, test=100.)
    c = make_fit(k=2, val=.01, test=0.)
    b["geometry"] = None
    store(tmp_path, [a, b, c])
    result = group(recovery.build_report([tmp_path]))
    assert result["selected"]["k"] == 1
    assert result["selected"]["test_kl"] == 100.
    assert result["verdict"] == "not recovered"
    assert not result["grid_complete"]
    assert result["best_val_kl"] == .01
    assert result["classes"]["gru"]["selected"] is None
    assert result["envelope"][3]["val_kl"] is None


def test_partial_fit_never_claims_complete_recovery_and_zero_dim_needs_no_geometry(tmp_path):
    fit = make_fit(condition="variant_1", k=0)
    fit["geometry"] = None
    fit["geometry_error"] = "constant"
    store(tmp_path, [fit])
    result = group(recovery.build_report([tmp_path]), condition="variant_1")
    assert result["geometry_passes"]
    assert result["verdict"] == "partially recovered"
    assert result["selected"]["probe_errors"]["geometry_error"] == "constant"
    assert not result["grid_complete"]


def test_partial_improvement_and_controls_use_full_belief_dimension(tmp_path):
    fits = [make_fit(condition="reward_factor_1", k=0, val=1., test=1.),
            make_fit(condition="reward_factor_1", k=4, val=.01, test=.02)]
    for fit in fits:
        fit["control"] = True
    store(tmp_path, fits)
    result = group(recovery.build_report([tmp_path]), condition="reward_factor_1", control=True)
    assert result["true_dimension"] == 4
    assert result["geometry_target"] == "belief"
    assert result["test_kl_reduction"] == .98
    assert result["verdict"] == "partially recovered"


def test_reader_reuses_completed_summary_contract_and_rejects_duplicates(tmp_path):
    fit = make_fit()
    root = tmp_path/"one"
    directory = store(root, [fit], status="completed")
    (directory/"seed42_affine_k2_failed.json").write_text('{"candidates": []}')
    assert len(list(summarize.read_fits([root]))) == 1
    store(tmp_path/"two", [fit], prefix="other")
    assert len(list(summarize.read_fits([tmp_path], run_prefix="gbs-v3"))) == 1
    with pytest.raises(ValueError, match="duplicate fit"):
        list(summarize.read_fits([root, root]))
    (directory/"seed42_affine_k3.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        list(summarize.read_fits([root]))


@pytest.mark.parametrize("quotient", [False, True])
def test_trusted_hmm_matrix_entries_align_recorded_permutation_and_known_quotient(tmp_path, monkeypatch, quotient):
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
    monkeypatch.setattr(recovery.data, "ROOT", tmp_path/"experiments")
    fine_T, fine_E = environment.action_transitions(2), environment.emission_matrix()
    source = tmp_path/"experiments"/"source"
    source.mkdir(parents=True)
    (source/"resolved_recipe.json").write_text(json.dumps({"process": {"transitions": fine_T.tolist(), "emission": fine_E.tolist()}}))
    if quotient:
        true_T = np.stack((fine_T[:, [0, 2], :2].sum(-1), fine_T[:, [0, 2], 2]), -1)
        E = fine_E[[0, 2]]
        true_E = np.stack((E[:, :2].sum(-1), E[:, 2]), -1)
        order = [1, 0]
    else:
        true_T, true_E, E = fine_T, fine_E, fine_E
        order = [2, 0, 1]
    inverse = np.argsort(order)
    fit = make_fit(condition="variant_1" if quotient else "variant_2", k=len(E)-1, kind="hmm")
    fit["checkpoint"] = {"results_path": "experiments/source"}
    fit["hmm_matrices"] = {"applicable": True, "reference": "exact_reward_state_quotient" if quotient else "full",
                           "permutation": order, "permutation_method": "exhaustive", "transition_mse": 0., "emission_mse": 0.}
    directory = store(tmp_path/"snapshot", [fit])
    parameter = directory.parents[1]/"artifacts"/directory.name/f"seed42_hmm_k{fit['k']}.pkl"
    parameter.parent.mkdir(parents=True)
    with parameter.open("wb") as stream:
        pickle.dump({"spec": {"kind": "hmm", "k": fit["k"]}, "params": {
            "T": np.log(true_T[:, inverse][:, :, inverse]), "E": np.log(E[inverse])}}, stream)
    row = {"fit": fit, "path": directory/f"seed42_hmm_k{fit['k']}.json", "k": fit["k"]}
    assert not recovery.matrix_detail(row, False)["matrices_available"]
    result = recovery.matrix_detail(row, True)
    assert result["matrices_available"], result
    np.testing.assert_allclose(result["learned_T_aligned"], true_T, atol=1e-7)
    np.testing.assert_allclose(result["learned_E_aligned"], true_E, atol=1e-7)
    assert result["recorded_errors_agree"]
    assert result["counterfactual_reference"] == quotient
    assert result["parameter_sha256"]


def test_empty_report_and_cli_outputs_are_strict_json_markdown_and_pdf(tmp_path, monkeypatch):
    output = tmp_path/"report"/"results"
    monkeypatch.setattr("sys.argv", ["recovery", "--inputs", str(tmp_path/"missing"), "--output", str(output)])
    recovery.main()
    report = json.loads((output/"recovery.json").read_text())
    assert len(report["groups"]) == 30
    assert all(g["selected"] is None for g in report["groups"])
    assert (output/"kl_vs_k.pdf").read_bytes().startswith(b"%PDF")
    assert "pending" in (output/"recovery.md").read_text()
    with pytest.raises(FileExistsError):
        recovery.main()


def test_figure_uses_validation_not_descriptive_test_minimum(tmp_path):
    store(tmp_path, [make_fit(k=1, val=.1, test=.00001)])
    report = recovery.build_report([tmp_path])
    fig = recovery.make_figure(report)
    ax = fig.axes[1]
    policy = ax.lines[0]
    assert policy.get_ydata()[1] == .1
    assert np.isnan(policy.get_ydata()[0])
    assert ax.lines[-2].get_xdata()[0] == 1
    assert ax.lines[-1].get_xdata()[0] == 2
    recovery.plt.close(fig)
