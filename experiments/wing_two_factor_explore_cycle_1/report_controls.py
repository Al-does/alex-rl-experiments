from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("token_guess", "reward_both", "reward_factor_1")
DEFAULT_REPORT_PATHS = {
    "token_guess": Path("experiments/wing_token_guess_cycle_1/ppo/results/probe_controls_20260908/validated.json"),
    "reward_both": Path("experiments/wing_two_factor_explore_cycle_2/reward_both_state_0/results/probe_controls_20260908/validated.json"),
    "reward_factor_1": Path("experiments/wing_two_factor_explore_cycle_2/reward_factor_1_state_0/results/probe_controls_20260908/validated.json"),
}
TITLES = {
    "token_guess": "Token guess (passive)",
    "reward_both": "Reward both (controlled)",
    "reward_factor_1": "Reward factor 1 (controlled)",
}
BASELINES = (
    "train_mean", "ntp", "log_ntp", "joint_ntp", "log_joint_ntp",
    "policy_probabilities", "centered_log_policy", "expected_immediate_reward_all_actions",
)
SUFFIX_LENGTHS = (0, 1, 2, 4, 8, 16, 32)
TABLE_NAMES = (
    "success", "probe_comparison", "suffix_baselines", "null_controls",
    "null_replicates", "history_shuffle", "alternative_models", "candidate_selection",
)
GENERATOR = "experiments.wing_two_factor_explore_cycle_1.report_controls"
OUTPUT_NAMES = tuple(f"{name}.csv" for name in TABLE_NAMES) + (
    "tables.md", "task_success.png", "task_success.svg", "report_manifest.json",
)


def _number(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _interval(value, name, *, nullable=False):
    if value is None and nullable:
        return None, None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two CI endpoints")
    low, high = (_number(item, name) for item in value)
    if low > high:
        raise ValueError(f"{name} CI endpoints must be ordered")
    return low, high


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _indexed(labels, prefix):
    if not labels or any(re.fullmatch(rf"{prefix}_[1-9][0-9]*", label) is None for label in labels):
        raise ValueError(f"invalid or missing {prefix} labels")
    return sorted(labels, key=lambda label: int(label.rsplit("_", 1)[1]))


def _metrics(value, prefix):
    return {
        f"{prefix}_mse": _number(value["mse"], f"{prefix}.mse", nullable=True),
        f"{prefix}_r_squared": _number(value["r_squared"], f"{prefix}.r_squared", nullable=True),
    }


def _ci_fields(value, key, prefix=None):
    low, high = _interval(value[key], key, nullable=True)
    prefix = key if prefix is None else prefix
    return {f"{prefix}_low": low, f"{prefix}_high": high}


def _factor_role(condition, factor):
    if condition == "token_guess":
        return "guessed"
    return "unrewarded" if condition == "reward_factor_1" and factor == "factor_2" else "rewarded"


def _success_rows(success):
    if success.get("schema_version") != 1:
        raise ValueError("task success requires schema_version 1")
    if not isinstance(success.get("metadata"), dict):
        raise ValueError("task success requires metadata")
    if success["metadata"].get("warmup", 0) != 0:
        raise ValueError("task success must use complete episodes with no warmup")
    if set(success.get("conditions", {})) != set(CONDITIONS):
        raise ValueError("task success requires all three conditions")
    rows = []
    for condition in CONDITIONS:
        data = success["conditions"][condition]
        metric = "joint_token_accuracy" if condition == "token_guess" else "mean_rewarded_factor_arrival_occupancy"
        if data.get("metric") != metric:
            raise ValueError(f"{condition}: expected metric {metric}")
        mean = _number(data["mean"], f"{condition}.mean")
        if not 0 <= mean <= 1:
            raise ValueError(f"{condition}: success mean must be a fraction in [0, 1]")
        low, high = _interval(data["ci95"], f"{condition}.ci95")
        if data.get("horizon") != 1024 or data.get("warmup", 0) != 0:
            raise ValueError("task success must use complete 1024-step episodes with no warmup")
        if not isinstance(data.get("checkpoint"), str) or not data["checkpoint"]:
            raise ValueError(f"{condition}: checkpoint is required")
        reference = data.get("bayes_reference")
        if not isinstance(reference, dict):
            raise ValueError(f"{condition}: a Bayes benchmark reference is required")
        allowed_kinds = ("exact_optimal_accuracy", "monte_carlo_optimal_accuracy") if condition == "token_guess" else ("numerical_upper_bound",)
        kind = reference.get("kind")
        if kind not in allowed_kinds:
            raise ValueError(f"{condition}: Bayes reference kind must be one of {allowed_kinds}")
        value = _number(reference["value"], f"{condition}.bayes_reference.value")
        if not 0 < value <= 1:
            raise ValueError(f"{condition}: Bayes reference must be a positive fraction <= 1")
        ref_low, ref_high = _interval(reference["ci95"], "Bayes reference ci95", nullable=True)
        if not isinstance(reference.get("label"), str) or not reference["label"] or not isinstance(reference.get("details"), dict):
            raise ValueError("Bayes reference requires a label and details")
        rows.append({
            "condition": condition, "metric": metric,
            "mean_fraction": mean, "ci95_low_fraction": low, "ci95_high_fraction": high,
            "mean_percent": mean * 100, "ci95_low_percent": low * 100, "ci95_high_percent": high * 100,
            "n_episodes": _integer(data["n_episodes"], "n_episodes"), "horizon": 1024, "warmup": 0,
            "checkpoint": data["checkpoint"], "agent_steps": _integer(data["agent_steps"], "agent_steps"),
            "bayes_reference_kind": kind, "bayes_reference_label": reference["label"],
            "bayes_reference_fraction": value, "bayes_reference_percent": value * 100,
            "bayes_ci95_low_fraction": ref_low, "bayes_ci95_high_fraction": ref_high,
            "bayes_ci95_low_percent": None if ref_low is None else ref_low * 100,
            "bayes_ci95_high_percent": None if ref_high is None else ref_high * 100,
            "bayes_reference_details": reference["details"],
        })
    return rows


def rows_from_reports(report_by_condition, success):
    try:
        return _rows_from_reports(report_by_condition, success)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"missing or malformed required report schema field: {error}") from error


def _rows_from_reports(report_by_condition, success):
    if set(report_by_condition) != set(CONDITIONS):
        raise ValueError("control reports require all three conditions")
    rows = {name: [] for name in TABLE_NAMES}
    rows["success"] = _success_rows(success)
    for condition in CONDITIONS:
        report = report_by_condition[condition]
        if report.get("schema_version") != 2:
            raise ValueError(f"{condition}: full validated control report requires schema_version 2")
        metadata = report["metadata"]
        if metadata["condition"] != condition:
            raise ValueError(f"control condition mismatch: {condition} != {metadata['condition']}")
        seed = _integer(metadata["seed"], "analysis seed", minimum=0)
        repeats = _integer(metadata["n_null_repeats"], "n_null_repeats")
        if set(report["factors"]) != {"factor_1", "factor_2"}:
            raise ValueError(f"{condition}: both factor_1 and factor_2 must be present")
        search = report["adversarial_search"]
        if not search["selection_scope"].startswith("training_episodes_only"):
            raise ValueError("alternative candidates must be selected using training episodes only")
        candidates = {item["id"]: item for item in search["candidates"]}
        if len(candidates) != len(search["candidates"]) or not set(search["selected_ids"]) <= set(candidates):
            raise ValueError("invalid alternative candidate IDs")
        selected = set(search["selected_ids"])
        for candidate in sorted(candidates.values(), key=lambda item: int(item["id"].rsplit("_", 1)[1])):
            if candidate["id"] in selected and not candidate["eligible"]:
                raise ValueError("selected alternative must be eligible")
            rows["candidate_selection"].append({
                "condition": condition, "candidate": candidate["id"],
                "selected": candidate["id"] in selected, "eligible": candidate["eligible"],
                "parameters": candidate["parameters"],
                "selection_score": candidate["selection_score"],
                "selection_next_observation_kl": candidate["selection_next_observation_kl"],
                "test_next_observation_kl": candidate.get("test_next_observation_kl"),
                "selection_affine_residual_ratio_per_factor": candidate["selection_affine_residual_ratio_per_factor"],
                "selection_min_permutation_mse_per_factor": candidate["selection_min_permutation_mse_per_factor"],
                "max_mean_kl_nats": search["max_mean_kl_nats"], "selection_scope": search["selection_scope"],
            })
        for factor in ("factor_1", "factor_2"):
            data = report["factors"][factor]
            common = {"condition": condition, "factor": factor, "factor_role": _factor_role(condition, factor)}
            baselines = data["baselines"]
            required = set(BASELINES[:-1] if condition == "token_guess" else BASELINES)
            required.update(f"suffix_{length}_belief" for length in SUFFIX_LENGTHS)
            if not required <= set(baselines):
                raise ValueError(f"{condition}/{factor}: missing required baselines: {sorted(required - set(baselines))}")
            for length in SUFFIX_LENGTHS:
                rows["suffix_baselines"].append({
                    **common, "suffix_length": length,
                    **_metrics(baselines[f"suffix_{length}_belief"], "baseline"),
                })
            layer_names = _indexed(data["layers"], "layer")
            for layer in layer_names:
                values = data["layers"][layer]
                identity = {**common, "layer": layer, "is_last_layer": layer == layer_names[-1]}
                probe = _metrics(values["belief_probe"], "probe")
                for baseline in BASELINES + tuple(f"suffix_{length}_belief" for length in SUFFIX_LENGTHS):
                    if baseline not in baselines:
                        continue
                    surplus = values["surplus_over_baselines"][baseline]
                    rows["probe_comparison"].append({
                        **identity, **probe, "delta_contrast_r_squared": values["delta_contrast"]["r_squared"],
                        "baseline": baseline, **_metrics(baselines[baseline], "baseline"),
                        "delta_r_squared": _number(surplus["delta_r_squared"], "delta_r_squared", nullable=True),
                        **_ci_fields(surplus, "delta_r_squared_ci"),
                        "mse_improvement": _number(surplus["mse_improvement"], "mse_improvement", nullable=True),
                        **_ci_fields(surplus, "mse_improvement_ci"),
                        "ci_reason": surplus.get("ci_reason"),
                        "delta_r_squared_ci_reason": surplus.get("delta_r_squared_ci_reason"),
                        "n_test_episodes": surplus["n_groups"], "n_bootstrap_resamples": surplus["n_resamples"],
                    })
                nulls = values["nulls"]
                if len(nulls) != repeats or set(nulls) != {str(index) for index in range(repeats)}:
                    raise ValueError("null repetitions disagree with metadata")
                null_names = sorted(nulls["0"])
                if not null_names or any(set(null) != set(null_names) for null in nulls.values()):
                    raise ValueError("null controls must match across repetitions")
                for name in null_names:
                    replicate_rows = []
                    for repeat in range(repeats):
                        null = nulls[str(repeat)][name]
                        replicate_rows.append({
                            **identity, "control": name, "repeat": repeat, "null_seed": seed + 1000 + repeat,
                            "fit_seed": null.get("fit", {}).get("seed"), **_metrics(null, "null"),
                            "sampling": null.get("sampling"),
                        })
                    rows["null_replicates"].extend(replicate_rows)
                    summary = {**identity, "control": name, "n_repeats": repeats, "std_ddof": 1}
                    for metric in ("mse", "r_squared"):
                        samples = [row[f"null_{metric}"] for row in replicate_rows]
                        valid = [value for value in samples if value is not None]
                        summary.update({
                            f"{metric}_mean": statistics.mean(valid) if valid else None,
                            f"{metric}_std": statistics.stdev(valid) if len(valid) > 1 else None,
                            f"{metric}_n_valid": len(valid), f"{metric}_replicates": samples,
                        })
                    rows["null_controls"].append(summary)
                shifted = values["shuffled_history_stress"]
                for name, metrics in (
                    ("original_unshuffled", values["belief_probe"]),
                    ("frozen_original_probe", shifted["frozen_original_probe"]),
                    ("refitted_on_shuffled_histories", shifted["refitted_on_shuffled_histories"]),
                ):
                    rows["history_shuffle"].append({**identity, "evaluation": name, **_metrics(metrics, "probe")})
                alternatives = values["alternative_belief_probes"]
                if set(alternatives) != selected:
                    raise ValueError("alternative probe IDs disagree with training-only selection")
                for name in sorted(selected, key=lambda label: int(label.rsplit("_", 1)[1])):
                    candidate, alternative = candidates[name], alternatives[name]
                    comparison = alternative["paired_target_comparison"]
                    true_r2, alt_r2 = values["belief_probe"]["r_squared"], alternative["r_squared"]
                    delta = _number(comparison["r_squared_difference"], "alternative r_squared_difference", nullable=True)
                    if true_r2 is not None and alt_r2 is not None and (
                        delta is None or not math.isclose(delta, true_r2 - alt_r2, rel_tol=1e-8, abs_tol=1e-12)
                    ):
                        raise ValueError("alternative R2 difference does not match true minus alternative R2")
                    index = int(factor.rsplit("_", 1)[1]) - 1
                    rows["alternative_models"].append({
                        **identity, "candidate": name, **_metrics(values["belief_probe"], "true"),
                        **_metrics(alternative, "alternative"), "delta_r_squared": delta,
                        **_ci_fields(comparison, "r_squared_difference_ci", "delta_r_squared_ci"),
                        "ci_reason": comparison.get("ci_reason"),
                        "parameters": candidate["parameters"],
                        "selection_next_observation_kl": candidate["selection_next_observation_kl"],
                        "test_next_observation_kl": candidate["test_next_observation_kl"],
                        "selection_affine_residual_ratio": candidate["selection_affine_residual_ratio_per_factor"][index],
                        "selection_min_permutation_mse": candidate["selection_min_permutation_mse_per_factor"][index],
                        "n_test_episodes": comparison["n_groups"], "n_bootstrap_resamples": comparison["n_resamples"],
                    })
    return rows


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _cell(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return _json(value)
    return str(value)


def _table(headers, rows):
    def line(items):
        return "| " + " | ".join(_cell(item).replace("|", "\\|").replace("\n", " ") for item in items) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]) + "\n"


def _estimate(row, value, low, high, *, percent=False):
    suffix = "%" if percent else ""
    return f"{_cell(row[value])}{suffix} [{_cell(row[low])}, {_cell(row[high])}]{suffix}"


def _delta(row):
    return _estimate(row, "delta_r_squared", "delta_r_squared_ci_low", "delta_r_squared_ci_high")


def _reference_method(row):
    details = row["bayes_reference_details"]
    if row["bayes_reference_kind"] == "exact_optimal_accuracy":
        analytic = details.get("analytic_cross_check") or {}
        guess = analytic.get("joint_guess")
        rule = f"Always guess ({', '.join(map(str, guess))}). " if guess is not None else ""
        return rule + "Statewise-dominant token and stationary uniform source prior give analytic optimal accuracy; Monte Carlo is only a cross-check."
    if row["bayes_reference_kind"] == "monte_carlo_optimal_accuracy":
        return "Monte Carlo evaluation of the optimal pending-token prediction rule on independent complete episodes."
    grid = details.get("grid_refinements", {})
    resolution = grid.get("selected_subdivisions")
    suffix = f" Selected grid: {resolution} subdivisions." if resolution is not None else ""
    return "Finite-horizon belief-grid planning with convex value interpolation; numerical upper bound, not an exact attainable optimum." + suffix


def _benchmark_tables(success):
    grid_rows, estimates = [], []
    for row in success:
        details = row["bayes_reference_details"]
        grid = details.get("grid_refinements", {})
        for refinement in grid.get("refinements", []):
            subdivisions = refinement.get("grid", {}).get("subdivisions", refinement.get("subdivisions"))
            value = _number(refinement["upper_bound_fraction"], "grid upper_bound_fraction")
            grid_rows.append((row["condition"], subdivisions, value * 100, subdivisions == grid.get("selected_subdivisions")))
        if row["bayes_reference_kind"] == "exact_optimal_accuracy" and "mean" in details:
            estimate, label = details, "Passive optimal-rule MC cross-check"
        elif "attainable_policy" in details:
            estimate, label = details["attainable_policy"], "Attainable full-belief policy (MC)"
        else:
            continue
        mean = _number(estimate["mean"], "benchmark estimate mean")
        low, high = _interval(estimate["ci95"], "benchmark estimate ci95")
        estimates.append((row["condition"], label, f"{_cell(mean * 100)} [{_cell(low * 100)}, {_cell(high * 100)}]", estimate.get("n_episodes")))
    parts = []
    if grid_rows:
        parts.extend([
            "### Controlled grid-refinement audit\n",
            "The selected value is the minimum computed numerical upper bound. Resolution stability is not a certified error bar or proof of equality to the continuous-belief optimum; there is no extrapolation.\n",
            _table(("Condition", "Grid subdivisions", "Numerical upper bound %", "Selected"), grid_rows),
        ])
    if estimates:
        parts.extend([
            "### Independent benchmark policy checks\n",
            "These are Monte Carlo estimates with 95% CIs over independent full episodes. When an exact passive maximum is provided, its MC cross-check does not replace it. Controlled attainable-policy estimates are neither exact optima nor rigorous lower confidence bounds.\n",
            _table(("Condition", "Evaluation", "Expected success % [95% CI]", "Episodes"), estimates),
        ])
    return "\n".join(parts)


def _markdown(rows, reports, success_link=None):
    success = rows["success"]
    passive = next(row for row in success if row["condition"] == "token_guess")
    if passive["bayes_reference_kind"] == "exact_optimal_accuracy":
        passive_description = "The passive Bayes reference is the exact Bayes maximum from the statewise-dominant-token rule, not a Monte Carlo estimate. Its CI is not applicable; a separate MC cross-check is reported when available."
    else:
        passive_description = "The passive Bayes reference is a Monte Carlo estimate of optimal token accuracy."
    details_link = f"[Full source evaluation]({success_link})" if success_link is not None else "[Full benchmark details](report_manifest.json)"
    parts = [
        "# Wing task success and specificity controls\n",
        "## Task success at the final checkpoint\n",
        "![Final-checkpoint task success with Bayes references](task_success.png)\n\n[Vector figure (SVG)](task_success.svg) | [Full-precision success CSV](success.csv)\n",
        "Absolute task-success percentages, not probe R² and not percent of a benchmark. Each estimate uses complete 1024-step episodes, including the initial steps, with no warmup. Bars are final checkpoints, not training curves.\n",
        _table(
            ("Condition", "Metric", "Expected success % [95% CI]", "Episodes", "Agent steps", "Bayes reference kind", "Bayes reference % [95% CI]"),
            ((row["condition"], row["metric"], _estimate(row, "mean_percent", "ci95_low_percent", "ci95_high_percent", percent=True),
              row["n_episodes"], row["agent_steps"], row["bayes_reference_kind"],
              _estimate(row, "bayes_reference_percent", "bayes_ci95_low_percent", "bayes_ci95_high_percent", percent=True)) for row in success),
        ),
        passive_description + " Controlled references are numerical upper bounds, not exact or demonstrated attainable Bayes maxima; an absent or inapplicable reference CI is shown as NA.\n",
        _table(("Condition", "Reference label", "Methodological summary"),
               ((row["condition"], row["bayes_reference_label"], _reference_method(row)) for row in success)),
        details_link + " contains the analytic argument, Monte Carlo cross-checks, grid refinements, convergence diagnostics, and attainable-policy evaluation where available. Full details are also retained in [success.csv](success.csv) and [report_manifest.json](report_manifest.json).\n",
        _benchmark_tables(success),
        _table(("Condition", "Evaluated checkpoint"), ((row["condition"], row["checkpoint"]) for row in success)),
        "## Interpretation and sampling scope\n",
        "These are descriptive linear-accessibility outcomes, not evidence of causal use or a causal mechanism. CIs are episode-conditional for one trained seed per condition, not training-seed uncertainty. Probe surplus and alternative-model CIs use paired episode bootstrap with fixed fitted probes. Null repetitions are control draws, not additional trained agents. Multiple comparisons are descriptive.\n",
        "The main probe view fixes the last encoder layer by index: no post-selected best layer. All layers are retained below and in CSV. Alternative candidates are selected on training histories only, never test histories or activations; the finite search is not a worst-case guarantee. Factors are never averaged, including the unrewarded factor in reward_factor_1.\n",
        "Task-success full episodes are separate from the original probe collection, whose warmup and episode/sample counts remain recorded here. Controlled reports use their recorded cycle-2 parameters, not cycle-1 rotation strength.\n",
        _table(("Condition", "Parameters", "Analysis seed", "Fit/test samples", "Fit/test episodes", "Probe fit/test warmup", "Bootstrap resamples"), (
            (condition, reports[condition]["metadata"]["parameters"], reports[condition]["metadata"]["seed"],
             f"{reports[condition]['metadata']['n_fit']}/{reports[condition]['metadata']['n_test']}",
             f"{reports[condition]['metadata']['train_episodes']}/{reports[condition]['metadata']['test_episodes']}",
             f"{reports[condition]['metadata']['train_collection'].get('warmup', 'NA')}/{reports[condition]['metadata']['test_collection'].get('warmup', 'NA')}",
             reports[condition]["metadata"]["n_bootstrap_resamples"]) for condition in CONDITIONS)),
        "## Last-layer belief accessibility\n",
    ]
    main = [row for row in rows["probe_comparison"] if row["is_last_layer"] and row["baseline"] == "train_mean"]
    parts.append(_table(("Condition", "Factor", "Role", "Layer", "Probe MSE", "Probe R²", "Delta-contrast R²"), (
        (row["condition"], row["factor"], row["factor_role"], row["layer"], row["probe_mse"], row["probe_r_squared"], row["delta_contrast_r_squared"]) for row in main)))
    parts.extend([
        "## Last-layer probe versus output and reward baselines\n",
        "`ntp`/`log_ntp` use the target factor marginal; `joint_ntp`/`log_joint_ntp` use the full four-token distribution. Policy probabilities and centered log policy are separate nuisance features, not NTP. Reward features are model-derived expected immediate reward for all actions, not observed rewards fed into belief targets. MSE is unscaled belief-coordinate squared error; R² and its CIs are dimensionless, not percentages. ΔR² = probe − baseline; MSE improvement = baseline − probe (positive favors the probe).\n",
    ])
    comparison_headers = ("Condition", "Factor", "Layer", "Baseline", "Probe MSE", "Probe R²", "Baseline MSE", "Baseline R²", "ΔR² [95% CI]", "MSE improvement [95% CI]")
    def comparison_table(values):
        return _table(comparison_headers, (
            (row["condition"], row["factor"], row["layer"], row["baseline"], row["probe_mse"], row["probe_r_squared"],
             row["baseline_mse"], row["baseline_r_squared"], _delta(row),
             _estimate(row, "mse_improvement", "mse_improvement_ci_low", "mse_improvement_ci_high")) for row in values))
    parts.append(comparison_table(row for row in rows["probe_comparison"] if row["is_last_layer"] and row["baseline"] in BASELINES))
    parts.extend([
        "## Suffix-belief baselines: 0 through 32 frames\n",
        "The measured suffix lengths are 0, 1, 2, 4, 8, 16, and 32 (not every integer). Each uses the model reset prior and a training-fitted affine map to the full belief target; length 0 is a reset-prior baseline, not a policy-stationary prior.\n",
        _table(("Condition", "Factor", "Role", "Suffix length", "Baseline MSE", "Baseline R²"), (
            (row["condition"], row["factor"], row["factor_role"], row["suffix_length"], row["baseline_mse"], row["baseline_r_squared"]) for row in rows["suffix_baselines"])),
        "## Last-layer matched and random controls\n",
        "Mean ± sample standard deviation (ddof=1) across null repetitions, not a CI. Matched activations preserve local/output structure, so nonzero R² is not automatically a failure. These are not exchangeability-based significance tests. Every replicate, null seed, fit seed, and available matching diagnostics is retained in [null_replicates.csv](null_replicates.csv); [null_controls.csv](null_controls.csv) also stores the metric arrays.\n",
        _table(("Condition", "Factor", "Layer", "Control", "Repeats", "MSE mean ± SD", "R² mean ± SD"), (
            (row["condition"], row["factor"], row["layer"], row["control"], row["n_repeats"],
             f"{_cell(row['mse_mean'])} ± {_cell(row['mse_std'])}", f"{_cell(row['r_squared_mean'])} ± {_cell(row['r_squared_std'])}")
            for row in rows["null_controls"] if row["is_last_layer"])),
        "## Last-layer shuffled-history stress test\n",
        "Original unshuffled evaluation is shown beside the frozen original probe and a probe refitted on shuffled histories. Shuffling recomputes both activations and exact beliefs. This is a distribution shift, not a null required to yield zero R².\n",
        _table(("Condition", "Factor", "Role", "Layer", "Evaluation", "MSE", "R²"), (
            (row["condition"], row["factor"], row["factor_role"], row["layer"], row["evaluation"], row["probe_mse"], row["probe_r_squared"])
            for row in rows["history_shuffle"] if row["is_last_layer"])),
        "## Last-layer alternative-model comparisons\n",
        "Every training-selected candidate is evaluated separately for each factor. ΔR² = true-target probe R² − alternative-target probe R², each using its own target variance; the paired 95% interval is an R²-difference interval, not an MSE interval. Negative values favor alternative linear accessibility. KL is the mean over samples and factors in nats; for two independent factors the joint-token KL is twice this value. Test KL is diagnostic only, not a selection input.\n",
        _table(("Condition", "Factor", "Role", "Layer", "Candidate", "True R²", "Alternative R²", "ΔR² [95% CI]", "Parameters", "Selection KL", "Test KL"), (
            (row["condition"], row["factor"], row["factor_role"], row["layer"], row["candidate"], row["true_r_squared"], row["alternative_r_squared"],
             _delta(row), row["parameters"], row["selection_next_observation_kl"], row["test_next_observation_kl"])
            for row in rows["alternative_models"] if row["is_last_layer"])),
        "### Training-only candidate search audit\n",
        "Unselected candidates were not evaluated with test belief probes. Selection score is the minimum per-factor affine residual ratio, subject to the recorded KL budget and non-equivalence checks. Their per-factor residual and permutation diagnostics are preserved in [candidate_selection.csv](candidate_selection.csv).\n",
        _table(("Condition", "Candidate", "Selected", "Eligible", "Parameters", "Selection score", "Selection KL", "KL budget", "Test KL"), (
            (row["condition"], row["candidate"], row["selected"], row["eligible"], row["parameters"], row["selection_score"],
             row["selection_next_observation_kl"], row["max_mean_kl_nats"], row["test_next_observation_kl"]) for row in rows["candidate_selection"])),
        "## All-layer probe comparisons (no test-layer selection)\n",
        comparison_table(rows["probe_comparison"]),
        "## Machine-readable results and provenance\n",
        " | ".join(f"[{name}.csv]({name}.csv)" for name in TABLE_NAMES) + "\n\n[Reproducibility manifest](report_manifest.json) records input hashes, seed sources, source provenance, and output hashes. CSV retains full floating-point precision; NA/blank means unavailable, never zero.\n",
    ])
    return "\n".join(parts)


def _render_success(rows):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    settings = dict(matplotlib.rcParamsDefault)
    settings.update({"backend": "Agg", "svg.hashsalt": GENERATOR, "svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10})
    with matplotlib.rc_context(settings):
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.69, wspace=0.32)
        fig.suptitle("Final-checkpoint task success", y=0.97, fontsize=17)
        fig.text(0.5, 0.91, "Complete 1024-step episodes · no warmup · absolute percentages with 95% CIs", ha="center", fontsize=11)
        try:
            for axis, row, color in zip(axes, rows, ("#4477AA", "#228833", "#AA3377"), strict=True):
                reference = row["bayes_reference_percent"]
                mean, low, high = row["mean_percent"], row["ci95_low_percent"], row["ci95_high_percent"]
                ref_high = row["bayes_ci95_high_percent"]
                ceiling = max(reference, mean, high, reference if ref_high is None else ref_high)
                axis.set_ylim(0, ceiling + max(3.0, ceiling * 0.08))
                axis.set_xlim(-0.7, 0.7)
                axis.bar([0], [mean], width=0.62, color=color, zorder=3)
                axis.vlines(0, low, high, color="black", linewidth=1.4, zorder=5)
                axis.hlines([low, high], -0.055, 0.055, color="black", linewidth=1.4, zorder=5)
                axis.axhline(reference, color="#222222", linestyle="--", linewidth=1.4, zorder=4)
                if row["bayes_ci95_low_percent"] is not None:
                    axis.axhspan(row["bayes_ci95_low_percent"], ref_high, color="#555555", alpha=0.13, zorder=2)
                kind, qualifier = {
                    "exact_optimal_accuracy": ("Passive Bayes maximum (exact)", "dominant-token rule"),
                    "monte_carlo_optimal_accuracy": ("Passive Bayes MC estimate", "optimal-accuracy estimate"),
                    "numerical_upper_bound": ("Controlled Bayes numerical upper bound", "not an exact maximum"),
                }[row["bayes_reference_kind"]]
                axis.text(0.5, 1.25, f"{kind}\n{reference:.2f}% ({qualifier})",
                          transform=axis.transAxes, ha="center", va="top", fontsize=10, fontweight="bold")
                axis.set_title(TITLES[row["condition"]], pad=10, fontsize=12)
                metric = "Joint token accuracy" if row["condition"] == "token_guess" else "Rewarded-factor arrival occupancy"
                axis.set_ylabel(f"{metric} (%)", fontsize=10)
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
                axis.set_xticks([0], [f"Final checkpoint\n{row['agent_steps']:,} agent steps\n{row['n_episodes']} episodes"])
                axis.text(0, mean / 2, f"{mean:.2f}%", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
                axis.set_axisbelow(True)
                axis.grid(axis="y", color="#dddddd", linewidth=0.7)
                axis.spines[["top", "right"]].set_visible(False)
            passive_kind = next(row["bayes_reference_kind"] for row in rows if row["condition"] == "token_guess")
            passive_description = "exact passive Bayes maximum" if passive_kind == "exact_optimal_accuracy" else "passive Bayes MC estimate"
            fig.text(0.5, 0.045, f"Dashed: {passive_description}; controlled numerical upper bounds (not exact maxima).\nOne trained seed per condition; uncertainty is conditional on the evaluated checkpoint.", ha="center", fontsize=10)
            outputs = {}
            for extension, metadata in (("png", {"Software": GENERATOR}), ("svg", {"Date": None, "Creator": GENERATOR})):
                buffer = io.BytesIO()
                fig.savefig(buffer, format=extension, dpi=160, metadata=metadata)
                content = buffer.getvalue()
                if extension == "svg":
                    content = b"\n".join(line.rstrip() for line in content.splitlines()) + b"\n"
                outputs[f"task_success.{extension}"] = content
            return outputs
        finally:
            plt.close(fig)


def _relative(path, root):
    return Path(os.path.relpath(Path(path).resolve(), root)).as_posix()


def _portable(value, root):
    if isinstance(value, dict):
        return {(_relative(key, root) if isinstance(key, str) and key.startswith("/") else key): _portable(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item, root) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return _relative(value, root)
    return value


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _input_record(data, path, root):
    if path is None:
        return {"path": None, "sha256": _digest(_json(data).encode()), "hash_basis": "canonical_json_in_memory", "schema_version": data["schema_version"]}
    path = Path(path)
    path = path if path.is_absolute() else root / path
    content = path.read_bytes()
    if json.loads(content) != data:
        raise ValueError(f"input path does not match supplied report: {path}")
    return {"path": _relative(path, root), "sha256": _digest(content), "hash_basis": "file_bytes", "schema_version": data["schema_version"]}


def _check_outputs(output_dir, overwrite):
    if output_dir.is_symlink():
        raise ValueError("output directory cannot be a symlink")
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists() or (output_dir / name).is_symlink()]
    if not existing:
        return
    if not overwrite:
        raise FileExistsError("report outputs already exist; use a new directory or --overwrite for an intact owned report")
    manifest_path = output_dir / "report_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("--overwrite requires an existing owned report manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("generator") != GENERATOR or set(manifest.get("outputs", {})) != set(OUTPUT_NAMES) - {"report_manifest.json"}:
        raise ValueError("--overwrite refuses outputs not owned by this report generator")
    for name in existing:
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"refusing unsafe output path: {path}")
        if name != "report_manifest.json" and _digest(path.read_bytes()) != manifest["outputs"][name]["sha256"]:
            raise ValueError(f"refusing to overwrite modified output: {name}")


def write_report(report_by_condition, success, output_dir, *, report_paths=None, success_path=None, overwrite=False, repo_root=REPO_ROOT):
    import matplotlib
    import numpy

    root, output_dir = Path(repo_root).resolve(), Path(output_dir)
    rows = rows_from_reports(report_by_condition, success)
    _check_outputs(output_dir, overwrite)
    if report_paths is not None and set(report_paths) != set(CONDITIONS):
        raise ValueError("report_paths must identify all three condition inputs")
    paths = {} if report_paths is None else report_paths
    source = Path(__file__).resolve()
    inputs = {condition: _input_record(report_by_condition[condition], paths.get(condition), root) for condition in CONDITIONS}
    inputs["success"] = _input_record(success, success_path, root)
    destinations = {(output_dir / name).resolve() for name in OUTPUT_NAMES}
    if any(record["path"] is not None and (root / record["path"]).resolve() in destinations for record in inputs.values()):
        raise ValueError("report outputs cannot overwrite input files")
    portable_rows = _portable(rows, root)
    outputs = {}
    for name in TABLE_NAMES:
        buffer = io.StringIO(newline="")
        records = portable_rows[name]
        fields = list(records[0]) if records else ["condition", "factor", "layer", "candidate"]
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({key: _json(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
        outputs[f"{name}.csv"] = buffer.getvalue().encode()
    success_link = None if inputs["success"]["path"] is None else quote(_relative(root / inputs["success"]["path"], output_dir.resolve()), safe="/")
    outputs["tables.md"] = _markdown(portable_rows, _portable(report_by_condition, root), success_link=success_link).encode()
    outputs.update(_render_success(portable_rows["success"]))
    manifest = {
        "schema_version": 1, "generator": GENERATOR,
        "inputs": inputs,
        "source": {"path": _relative(source, root), "sha256": _digest(source.read_bytes())},
        "seed_sources": {
            condition: {"analysis_seed": report_by_condition[condition]["metadata"]["seed"],
                        "analysis_seed_field": "metadata.seed", "null_seed_rule": "analysis_seed + 1000 + repeat (zero-based)",
                        "n_null_repeats": report_by_condition[condition]["metadata"]["n_null_repeats"],
                        "n_bootstrap_resamples": report_by_condition[condition]["metadata"]["n_bootstrap_resamples"]}
            for condition in CONDITIONS
        },
        "control_metadata": _portable({condition: report_by_condition[condition]["metadata"] for condition in CONDITIONS}, root),
        "success_metadata": _portable(success["metadata"], root),
        "success_conditions": _portable(success["conditions"], root),
        "rendering": {"backend": "Agg", "svg_hashsalt": GENERATOR, "svg_date": None, "png_software": GENERATOR,
                      "dpi": 160, "plot": "final_checkpoint_absolute_percent", "horizon": 1024, "warmup": 0},
        "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": numpy.__version__},
        "row_counts": {name: len(records) for name, records in rows.items()},
        "outputs": {name: {"sha256": _digest(content), "bytes": len(content)} for name, content in sorted(outputs.items())},
    }
    outputs["report_manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_outputs(output_dir, overwrite)
    for name in OUTPUT_NAMES:
        path = output_dir / name
        with path.open("wb" if overwrite and path.exists() else "xb") as handle:
            handle.write(outputs[name])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validated Wing task-success and descriptive specificity-control report")
    parser.add_argument("--success", type=Path, required=True, help="Schema-1 final-checkpoint task-success JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition.replace('_', '-')}-report", type=Path, default=DEFAULT_REPORT_PATHS[condition])
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only an intact report owned by this generator")
    args = parser.parse_args(argv)
    report_paths = {condition: getattr(args, f"{condition}_report") for condition in CONDITIONS}
    report_paths = {condition: path if path.is_absolute() else REPO_ROOT / path for condition, path in report_paths.items()}
    try:
        reports = {condition: json.loads(path.read_text()) for condition, path in report_paths.items()}
        success = json.loads(args.success.read_text())
        write_report(reports, success, args.output_dir, report_paths=report_paths, success_path=args.success.resolve(), overwrite=args.overwrite)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {len(OUTPUT_NAMES)} report files to {args.output_dir}")


if __name__ == "__main__":
    main()
