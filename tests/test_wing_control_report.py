import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics

import pytest

from experiments.wing_two_factor_explore_cycle_1 import report_controls as reporting


def _metrics(r_squared, mse):
    return {"r_squared": r_squared, "mse": mse, "fit": {"seed": 42}}


@pytest.fixture
def reports():
    result = {}
    for condition_index, condition in enumerate(reporting.CONDITIONS):
        candidates = []
        for candidate_index in (10, 2, 1):
            candidates.append({
                "id": f"alternative_{candidate_index}", "eligible": candidate_index != 1,
                "parameters": {"alpha": 0.81 + candidate_index / 100, "x": 0.4,
                               "strength": None if condition == "token_guess" else 1.0},
                "selection_score": candidate_index / 100,
                "selection_next_observation_kl": candidate_index / 1000,
                "test_next_observation_kl": candidate_index / 900,
                "selection_affine_residual_ratio_per_factor": [0.1, 0.2],
                "selection_min_permutation_mse_per_factor": [0.01, 0.02],
            })
        report = {
            "schema_version": 2,
            "metadata": {
                "condition": condition, "seed": 42, "n_null_repeats": 2,
                "n_bootstrap_resamples": 200, "n_fit": 20000, "n_test": 20000,
                "train_episodes": 30, "test_episodes": 28,
                "parameters": {"alpha": 0.94, "x": 0.4, "strength": None if condition == "token_guess" else 1.0},
                "train_collection": {"warmup": 32}, "test_collection": {"warmup": 32},
                "checkpoint": f"experiments/{condition}/artifacts/checkpoint_final",
                "provenance": {"source_sha256": {"experiments/source.py": "0" * 64}},
            },
            "adversarial_search": {"candidates": candidates, "selected_ids": ["alternative_10", "alternative_2"],
                                   "selection_scope": "training_episodes_only; no activations or test histories used", "max_mean_kl_nats": 0.02},
            "factors": {},
        }
        for factor_index in (2, 1):
            baselines = {}
            baseline_names = reporting.BASELINES[:-1] if condition == "token_guess" else reporting.BASELINES
            baseline_names += tuple(f"suffix_{length}_belief" for length in reporting.SUFFIX_LENGTHS)
            for baseline_index, name in enumerate(baseline_names):
                baselines[name] = _metrics(0.13 * condition_index + 0.03 * factor_index + 0.001 * baseline_index,
                                           0.001 + baseline_index / 10000)
            layers = {}
            for layer_index in (2, 10, 1):
                r_squared = 0.19 * condition_index + 0.06 * factor_index - 0.002 * layer_index
                mse = 0.02345678901234567 + layer_index / 10000
                probe = _metrics(r_squared, mse)
                surplus = {}
                for name, baseline in baselines.items():
                    delta = r_squared - baseline["r_squared"]
                    improvement = baseline["mse"] - mse
                    surplus[name] = {
                        "delta_r_squared": delta, "delta_r_squared_ci": [delta - 0.02, delta + 0.03],
                        "mse_improvement": improvement, "mse_improvement_ci": [improvement - 0.0002, improvement + 0.0003],
                        "n_groups": 28, "n_resamples": 200, "ci_reason": None,
                    }
                alternative_probes = {}
                for candidate_index in (10, 2):
                    delta = -candidate_index / 100 - factor_index / 1000
                    alternative_probes[f"alternative_{candidate_index}"] = {
                        **_metrics(r_squared - delta, 0.002 + candidate_index / 10000),
                        "paired_target_comparison": {
                            "r_squared_difference": delta,
                            "r_squared_difference_ci": [delta - 0.001, delta + 0.002],
                            "n_groups": 28, "n_resamples": 200,
                        },
                    }
                layers[f"layer_{layer_index}"] = {
                    "belief_probe": probe, "delta_contrast": {"r_squared": r_squared / 2},
                    "surplus_over_baselines": surplus,
                    "nulls": {str(repeat): {
                        name: _metrics(0.05 * condition_index + factor_index / 100 + layer_index / 1000 + repeat / 10000 + control_index / 100000,
                                       0.02 + repeat / 1000)
                        for control_index, name in enumerate(("shuffled_training_labels", "dirichlet_labels", "gaussian_covariance_matched",
                                                            "visible_suffix_matched_activations", "ntp_bin_matched_activations"))
                    } for repeat in range(2)},
                    "shuffled_history_stress": {
                        "frozen_original_probe": _metrics(r_squared - 0.2, mse + 0.03),
                        "refitted_on_shuffled_histories": _metrics(r_squared - 0.1, mse + 0.01),
                    },
                    "alternative_belief_probes": alternative_probes,
                }
            report["factors"][f"factor_{factor_index}"] = {"baselines": baselines, "layers": layers}
        result[condition] = report
    return result


@pytest.fixture
def success():
    conditions = {}
    for index, condition in enumerate(reporting.CONDITIONS):
        conditions[condition] = {
            "metric": "joint_token_accuracy" if condition == "token_guess" else "mean_rewarded_factor_arrival_occupancy",
            "mean": 0.41 + index * 0.1, "ci95": [0.40 + index * 0.1, 0.43 + index * 0.1],
            "n_episodes": 128, "horizon": 1024, "checkpoint": f"experiments/{condition}/artifacts/checkpoint_final",
            "agent_steps": 50000000 if condition == "reward_both" else 30000000,
            "bayes_reference": {
                "value": 0.78, "kind": "monte_carlo_optimal_accuracy" if condition == "token_guess" else "numerical_upper_bound",
                "label": "Passive optimal accuracy MC estimate" if condition == "token_guess" else "Controlled numerical upper bound",
                "ci95": [0.77, 0.79] if condition == "token_guess" else None,
                "details": {"seed": 2718, "horizon": 1024, "warmup": 0},
            },
        }
    return {"schema_version": 1, "conditions": conditions, "metadata": {"seed": 31415, "warmup": 0}}


def test_mapping_and_numeric_last_layer(reports, success):
    original = copy.deepcopy(reports)
    rows = reporting.rows_from_reports(reports, success)
    assert reports == original
    assert [row["condition"] for row in rows["success"]] == list(reporting.CONDITIONS)
    for row in rows["probe_comparison"]:
        factor = reports[row["condition"]]["factors"][row["factor"]]
        layer = factor["layers"][row["layer"]]
        assert row["probe_mse"] == layer["belief_probe"]["mse"]
        assert row["probe_r_squared"] == layer["belief_probe"]["r_squared"]
        assert row["baseline_r_squared"] == factor["baselines"][row["baseline"]]["r_squared"]
        assert row["is_last_layer"] == (row["layer"] == "layer_10")
        assert row["factor_role"] == ("guessed" if row["condition"] == "token_guess" else
                                      "unrewarded" if row["condition"] == "reward_factor_1" and row["factor"] == "factor_2" else "rewarded")
    for row in rows["suffix_baselines"]:
        expected = reports[row["condition"]]["factors"][row["factor"]]["baselines"][f"suffix_{row['suffix_length']}_belief"]
        assert row["baseline_mse"] == expected["mse"]
        assert row["baseline_r_squared"] == expected["r_squared"]
    assert {row["suffix_length"] for row in rows["suffix_baselines"]} == {0, 1, 2, 4, 8, 16, 32}


def test_percent_conversion_only_success_and_correct_ci_units(reports, success):
    rows = reporting.rows_from_reports(reports, success)
    for row in rows["success"]:
        source = success["conditions"][row["condition"]]
        assert row["mean_percent"] == source["mean"] * 100
        assert row["ci95_low_percent"] == source["ci95"][0] * 100
        assert row["ci95_high_percent"] == source["ci95"][1] * 100
        assert row["bayes_reference_percent"] == source["bayes_reference"]["value"] * 100
    for row in rows["probe_comparison"]:
        source = reports[row["condition"]]["factors"][row["factor"]]["layers"][row["layer"]]["surplus_over_baselines"][row["baseline"]]
        assert [row["delta_r_squared_ci_low"], row["delta_r_squared_ci_high"]] == source["delta_r_squared_ci"]
        assert [row["mse_improvement_ci_low"], row["mse_improvement_ci_high"]] == source["mse_improvement_ci"]
        assert row["delta_r_squared_ci_low"] != row["mse_improvement_ci_low"]
    for row in rows["alternative_models"]:
        source = reports[row["condition"]]["factors"][row["factor"]]["layers"][row["layer"]]["alternative_belief_probes"][row["candidate"]]
        assert row["alternative_r_squared"] == source["r_squared"]
        assert row["alternative_mse"] == source["mse"]
        assert row["delta_r_squared"] == pytest.approx(row["true_r_squared"] - row["alternative_r_squared"])
        assert [row["delta_r_squared_ci_low"], row["delta_r_squared_ci_high"]] == source["paired_target_comparison"]["r_squared_difference_ci"]
        factor_index = int(row["factor"].split("_")[1]) - 1
        candidate = next(item for item in reports[row["condition"]]["adversarial_search"]["candidates"] if item["id"] == row["candidate"])
        assert row["selection_affine_residual_ratio"] == candidate["selection_affine_residual_ratio_per_factor"][factor_index]
        assert row["parameters"] == candidate["parameters"]
        assert row["test_next_observation_kl"] == candidate["test_next_observation_kl"]


def test_null_replicates_and_history_mappings(reports, success):
    rows = reporting.rows_from_reports(reports, success)
    for row in rows["null_controls"]:
        source = reports[row["condition"]]["factors"][row["factor"]]["layers"][row["layer"]]["nulls"]
        values = [source[str(repeat)][row["control"]]["r_squared"] for repeat in range(2)]
        assert row["r_squared_replicates"] == values
        assert row["r_squared_mean"] == statistics.mean(values)
        assert row["r_squared_std"] == statistics.stdev(values)
        assert row["n_repeats"] == 2
        assert row["std_ddof"] == 1
    for row in rows["null_replicates"]:
        assert row["null_seed"] == 1042 + row["repeat"]
    for row in rows["history_shuffle"]:
        source = reports[row["condition"]]["factors"][row["factor"]]["layers"][row["layer"]]
        source = source["belief_probe"] if row["evaluation"] == "original_unshuffled" else source["shuffled_history_stress"][row["evaluation"]]
        assert row["probe_r_squared"] == source["r_squared"]
        assert row["probe_mse"] == source["mse"]


def test_all_actual_validated_reports_match_full_arrays_and_summaries(success):
    reports = {condition: json.loads((reporting.REPO_ROOT / path).read_text()) for condition, path in reporting.DEFAULT_REPORT_PATHS.items()}
    rows = reporting.rows_from_reports(reports, success)
    for condition, path in reporting.DEFAULT_REPORT_PATHS.items():
        assert not path.is_absolute()
        assert path.name == "validated.json"
        summary = json.loads((reporting.REPO_ROOT / path.with_name("validated_summary.json")).read_text())
        for row in (item for item in rows["probe_comparison"] if item["condition"] == condition):
            factor = reports[condition]["factors"][row["factor"]]
            layer = factor["layers"][row["layer"]]
            assert row["probe_r_squared"] == layer["belief_probe"]["r_squared"]
            assert row["probe_mse"] == layer["belief_probe"]["mse"]
            assert row["baseline_r_squared"] == factor["baselines"][row["baseline"]]["r_squared"]
            assert row["delta_r_squared_ci_low"] == layer["surplus_over_baselines"][row["baseline"]]["delta_r_squared_ci"][0]
            if row["is_last_layer"]:
                assert row["layer"] == summary["factors"][row["factor"]]["layer"]
                assert row["probe_r_squared"] == summary["factors"][row["factor"]]["belief_r_squared"]
        for row in (item for item in rows["alternative_models"] if item["condition"] == condition and item["is_last_layer"]):
            source = summary["factors"][row["factor"]]["alternatives"][row["candidate"]]
            assert row["alternative_r_squared"] == source["r_squared"]
            assert row["delta_r_squared"] == source["true_minus_alternative_r_squared"]
            assert [row["delta_r_squared_ci_low"], row["delta_r_squared_ci_high"]] == source["paired_95_percent_interval"]
    unrewarded = next(row for row in rows["probe_comparison"] if row["condition"] == "reward_factor_1" and row["factor"] == "factor_2" and row["is_last_layer"] and row["baseline"] == "ntp")
    assert unrewarded["probe_r_squared"] == pytest.approx(0.22892289959344303)
    assert unrewarded["baseline_r_squared"] == pytest.approx(0.48574761670533306)


@pytest.mark.parametrize("value", [-0.01, 0, 1.1, float("nan"), float("inf"), True])
def test_invalid_bayes_reference(reports, success, value):
    success["conditions"]["token_guess"]["bayes_reference"]["value"] = value
    with pytest.raises(ValueError, match="Bayes|bayes_reference|finite"):
        reporting.rows_from_reports(reports, success)


@pytest.mark.parametrize("field", ["bayes_reference", "ci95", "checkpoint", "n_episodes", "agent_steps"])
def test_missing_success_fields_fail(reports, success, field):
    del success["conditions"]["reward_both"][field]
    with pytest.raises(ValueError):
        reporting.rows_from_reports(reports, success)


@pytest.mark.parametrize("target", ["success", "report"])
def test_missing_schema_fails(reports, success, target):
    del (success if target == "success" else reports["reward_both"])["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        reporting.rows_from_reports(reports, success)


@pytest.mark.parametrize("field,value", [("metric", "r_squared"), ("horizon", 992), ("warmup", 32), ("mean", 50), ("n_episodes", 0), ("ci95", [0.5, 0.4])])
def test_invalid_success_semantics(reports, success, field, value):
    success["conditions"]["reward_factor_1"][field] = value
    with pytest.raises(ValueError):
        reporting.rows_from_reports(reports, success)


def test_wrong_condition_and_reference_kind_fail(reports, success):
    reports["reward_both"]["metadata"]["condition"] = "reward_factor_1"
    with pytest.raises(ValueError, match="condition mismatch"):
        reporting.rows_from_reports(reports, success)
    reports["reward_both"]["metadata"]["condition"] = "reward_both"
    success["conditions"]["reward_both"]["bayes_reference"]["kind"] = "exact_optimal_accuracy"
    with pytest.raises(ValueError, match="kind"):
        reporting.rows_from_reports(reports, success)


def test_incomplete_control_data_and_test_selection_fail(reports, success):
    del reports["token_guess"]["factors"]["factor_1"]["baselines"]["joint_ntp"]
    with pytest.raises(ValueError, match="baselines"):
        reporting.rows_from_reports(reports, success)
    reports["token_guess"]["factors"]["factor_1"]["baselines"]["joint_ntp"] = _metrics(0.1, 0.01)
    reports["token_guess"]["adversarial_search"]["selection_scope"] = "test_episodes"
    with pytest.raises(ValueError, match="training episodes only"):
        reporting.rows_from_reports(reports, success)


def test_rendering_reproducible_complete_and_scientifically_labeled(reports, success, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first_manifest = reporting.write_report(reports, success, first)
    second_manifest = reporting.write_report(reports, success, second)
    assert first_manifest == second_manifest
    assert set(path.name for path in first.iterdir()) == set(reporting.OUTPUT_NAMES)
    for name in reporting.OUTPUT_NAMES:
        content = (first / name).read_bytes()
        assert content
        assert content == (second / name).read_bytes()
        if name != "report_manifest.json":
            assert first_manifest["outputs"][name]["sha256"] == hashlib.sha256(content).hexdigest()
            assert first_manifest["outputs"][name]["bytes"] == len(content)
    text = (first / "tables.md").read_text()
    assert text.index("## Task success") < text.index("## Last-layer belief")
    for phrase in ("task_success.png", "task_success.svg", "numerical upper bounds", "not exact", "Monte Carlo", "unrewarded",
                   "not evidence of causal use", "one trained seed", "no post-selected best layer", "training histories only",
                   "episode-conditional", "not an MSE interval", "no warmup", "sample standard deviation", "joint_ntp"):
        assert phrase in text
    for phrase in ("proves causal", "establishes causal", "exact Bayes maximum", "percent of Bayes maximum"):
        assert phrase not in text
    svg = (first / "task_success.svg").read_text()
    assert all(line == line.rstrip() for line in svg.splitlines())
    assert "<dc:date>" not in svg
    assert "Passive Bayes MC estimate" in svg
    assert "Controlled Bayes numerical upper bound" in svg
    assert "R²" not in svg
    assert (first / "task_success.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with (first / "probe_comparison.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected = reporting.rows_from_reports(reports, success)["probe_comparison"][0]
    assert float(csv_rows[0]["probe_mse"]) == expected["probe_mse"]
    assert len(csv_rows[0]["probe_mse"]) > 10
    with pytest.raises(FileExistsError):
        reporting.write_report(reports, success, first)
    reporting.write_report(reports, success, first, overwrite=True)
    assert (first / "report_manifest.json").read_bytes() == (second / "report_manifest.json").read_bytes()


def test_exact_passive_reference_and_compact_benchmark_details(reports, success, tmp_path):
    exact_value = 0.7219334444444443
    bound = 0.6834409759292766
    success["conditions"]["token_guess"]["bayes_reference"] = {
        "kind": "exact_optimal_accuracy", "value": exact_value, "ci95": None,
        "label": "Bayes maximum (exact dominant-token rule)",
        "details": {
            "mean": 0.72195, "ci95": [0.7218, 0.7221], "n_episodes": 4096,
            "analytic_cross_check": {
                "fraction": exact_value, "joint_guess": [1, 1],
                "justification": "Token 1 dominates in every state; the uniform source prior is stationary.",
                "status": "exact Bayes-optimal value from a statewise-dominant token, evaluated in float64",
                "emission_probabilities_of_dominant_token": [0.97, 0.609, 0.97],
            },
        },
    }
    for condition in ("reward_both", "reward_factor_1"):
        reference = success["conditions"][condition]["bayes_reference"]
        reference["value"] = bound
        reference["details"] = {
            "grid_refinements": {
                "upper_bound_fraction": bound, "selected_subdivisions": 400,
                "refinements": [
                    {"grid": {"subdivisions": divisions}, "upper_bound_fraction": value,
                     "diagnostic_payload": "large_diagnostic_marker" * 500}
                    for divisions, value in ((100, 0.69), (200, 0.685), (400, bound))
                ],
                "convergence": {"last_absolute_fraction_change": 0.685 - bound},
            },
            "attainable_policy": {"mean": 0.67, "ci95": [0.668, 0.672], "n_episodes": 4096},
        }
    rows = reporting.rows_from_reports(reports, success)
    passive = rows["success"][0]
    assert passive["bayes_reference_kind"] == "exact_optimal_accuracy"
    assert passive["bayes_reference_percent"] == exact_value * 100
    assert passive["bayes_ci95_low_percent"] is None
    assert passive["bayes_ci95_high_percent"] is None
    success_path = tmp_path / "task_success_evaluation.json"
    success_path.write_text(json.dumps(success))
    original_input = success_path.read_bytes()
    manifest = reporting.write_report(reports, success, tmp_path, success_path=success_path)
    assert success_path.read_bytes() == original_input
    assert set(path.name for path in tmp_path.iterdir()) == set(reporting.OUTPUT_NAMES) | {success_path.name}
    text = (tmp_path / "tables.md").read_text()
    intro = text.split("## Interpretation and sampling scope")[0]
    for phrase in ("exact Bayes maximum", "Always guess (1, 1)", "numerical upper bounds", "not exact",
                   "[Full source evaluation](task_success_evaluation.json)", "Controlled grid-refinement audit",
                   "Passive optimal-rule MC cross-check", "Attainable full-belief policy (MC)", "not a certified error bar"):
        assert phrase in intro
    assert "large_diagnostic_marker" not in text
    assert len(intro) < 6000
    svg = (tmp_path / "task_success.svg").read_text()
    assert "Passive Bayes maximum (exact)" in svg
    assert "72.19% (dominant-token rule)" in svg
    assert "Controlled Bayes numerical upper bound" in svg
    assert "68.34% (not an exact maximum)" in svg
    assert "exact passive Bayes maximum; controlled numerical upper bounds" in svg
    assert "Passive Bayes MC estimate" not in svg
    rerendered = reporting._render_success(rows["success"])
    for name, content in rerendered.items():
        assert content == (tmp_path / name).read_bytes()
    with (tmp_path / "success.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    for row in csv_rows:
        source = success["conditions"][row["condition"]]["bayes_reference"]["details"]
        assert json.loads(row["bayes_reference_details"]) == source
        assert manifest["success_conditions"][row["condition"]]["bayes_reference"]["details"] == source


def test_plot_limits_include_ci_above_reference(success, monkeypatch):
    import matplotlib.axes
    recorded = []
    original = matplotlib.axes.Axes.set_ylim
    def capture(axis, bottom=None, top=None, **kwargs):
        if bottom == 0 and top is not None:
            recorded.append(top)
        return original(axis, bottom, top, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", capture)
    success["conditions"]["reward_both"]["ci95"] = [0.5, 1.02]
    reporting._render_success(reporting._success_rows(success))
    assert any(limit > 102 for limit in recorded)


def test_refuse_unowned_or_modified_outputs(reports, success, tmp_path):
    existing = tmp_path / "tables.md"
    existing.write_text("not owned")
    with pytest.raises(FileExistsError):
        reporting.write_report(reports, success, tmp_path)
    with pytest.raises(ValueError, match="owned report manifest"):
        reporting.write_report(reports, success, tmp_path, overwrite=True)
    assert existing.read_text() == "not owned"
    owned = tmp_path / "owned"
    reporting.write_report(reports, success, owned)
    (owned / "tables.md").write_text("edited by someone else")
    with pytest.raises(ValueError, match="modified output"):
        reporting.write_report(reports, success, owned, overwrite=True)
    assert (owned / "tables.md").read_text() == "edited by someone else"


def test_manifest_file_hashes_relative_paths_and_seeds(reports, success, tmp_path):
    paths = {}
    for condition, report in reports.items():
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(report))
        paths[condition] = path
    success_path = tmp_path / "input_success.json"
    success_path.write_text(json.dumps(success))
    manifest = reporting.write_report(reports, success, tmp_path / "out", report_paths=paths, success_path=success_path, repo_root=tmp_path)
    for condition, path in {**paths, "success": success_path}.items():
        entry = manifest["inputs"][condition]
        assert entry["path"] == path.name
        assert not Path(entry["path"]).is_absolute()
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["hash_basis"] == "file_bytes"
    assert "[Full source evaluation](../input_success.json)" in (tmp_path / "out" / "tables.md").read_text()
    assert manifest["seed_sources"]["token_guess"]["analysis_seed"] == 42
    assert manifest["success_metadata"]["seed"] == 31415
    assert manifest["control_metadata"]["reward_factor_1"]["parameters"]["strength"] == 1
    assert manifest["rendering"]["svg_date"] is None
    assert manifest["source"]["sha256"] == hashlib.sha256(Path(reporting.__file__).read_bytes()).hexdigest()
    paths["token_guess"].write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        reporting.write_report(reports, success, tmp_path / "bad", report_paths=paths)
    assert not (tmp_path / "bad").exists()


def test_cli_resolves_default_inputs_from_repo_root(reports, success, tmp_path, monkeypatch):
    success_path = tmp_path / "input_success.json"
    success_path.write_text(json.dumps(success))
    captured = {}
    def fake_write(report_by_condition, supplied_success, output_dir, **kwargs):
        captured.update(kwargs)
        assert set(report_by_condition) == set(reporting.CONDITIONS)
        assert supplied_success == success
    monkeypatch.setattr(reporting, "write_report", fake_write)
    monkeypatch.chdir(tmp_path)
    reporting.main(["--success", success_path.name, "--output-dir", str(tmp_path / "out")])
    assert captured["report_paths"] == {condition: reporting.REPO_ROOT / path for condition, path in reporting.DEFAULT_REPORT_PATHS.items()}
    assert captured["success_path"] == success_path
