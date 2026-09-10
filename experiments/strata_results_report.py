from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import harness
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/strata_token_guess_cycle_2/ppo/results/branch_analysis_20260909"
RUNS = {
    "token_cycle_1_context": "strata_token_guess_cycle_1/ppo/results/20260909T061802Z-7d4e2248",
    "token_cycle_2": "strata_token_guess_cycle_2/ppo/results/20260909T083950Z-45ea794b",
    "explore_cycle_3_both": "strata_two_factor_explore_cycle_3/reward_both_state_0/results/20260909T084200Z-e5f5f91b",
    "explore_cycle_3_factor_1": "strata_two_factor_explore_cycle_3/reward_factor_1_state_0/results/20260909T084527Z-64ab39a1",
}


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def dump_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def metrics(value):
    result = {key: value.get(key) for key in ("mse", "target_variance", "global_mse_ratio", "r_squared")}
    result["global_r2"] = 1.0 - value["global_mse_ratio"]
    delta = value.get("delta", {})
    for key in ("mse", "target_variance", "r_squared"):
        result[f"null_{key}"] = delta.get(key)
    result["null_global_r2"] = 1.0 - delta["global_mse_ratio"] if delta else None
    result["selected_rcond"] = value.get("fit", {}).get("selected_rcond")
    result["cv_folds"] = value.get("fit", {}).get("folds")
    result["cv_validation_mse"] = value.get("fit", {}).get("mean_validation_mse")
    assert np.isclose(result["global_r2"], result["r_squared"], atol=1e-10)
    return result


def flatten_values(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_values(child, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_values(child, f"{prefix}/{index}")
    else:
        yield prefix, value


def token_references(parameters, suffix_length=18):
    from envs.strata.model import strata_model
    model = strata_model(**parameters)
    edges = model.edge_transition_matrices
    emission = model.emission_matrix
    prior = model.initial_distribution
    np.testing.assert_allclose(prior @ model.transition_matrix, prior, atol=1e-14)
    paths = np.eye(3)[None, :, :]
    bounds = []
    for length in range(suffix_length + 1):
        probabilities = (prior @ paths) @ emission
        lower = probabilities.max(axis=-1).sum()
        upper = ((paths @ emission).max(axis=-1) @ prior).sum()
        bounds.append({"suffix_length": length, "lower": float(lower), "upper": float(upper), "gap": float(upper - lower)})
        if length < suffix_length:
            paths = np.concatenate([paths @ edge for edge in edges], axis=0)
    assert all(row["lower"] <= row["upper"] + 1e-12 for row in bounds)
    assert np.min(np.diff([row["lower"] for row in bounds])) > -1e-12
    assert np.max(np.diff([row["upper"] for row in bounds])) < 1e-12
    np.testing.assert_allclose(paths.sum(axis=0), np.linalg.matrix_power(model.transition_matrix, suffix_length), atol=1e-12)
    one_token = (prior @ edges) @ emission
    return {
        "parameters": parameters,
        "emission_matrix": emission.tolist(),
        "transition_matrix": model.transition_matrix.tolist(),
        "stationary_token_fractions": (prior @ emission).tolist(),
        "best_constant_accuracy": float((prior @ emission).max()),
        "best_current_visible_token_accuracy": float(one_token.max(axis=1).sum()),
        "best_current_visible_token_guesses": one_token.argmax(axis=1).tolist(),
        "full_information_source_state_upper": float(prior @ emission.max(axis=1)),
        "stationary_bayes_suffix_bounds": bounds,
        "strict_32_token_optimal_accuracy_bounds": {"lower": bounds[-1]["lower"], "upper": bounds[-1]["upper"], "scope": "An 18-token predictor is feasible with a 32-token window; the infinite-past genie upper bounds both. Bounds apply in expectation to stationary 32-token histories and retained post-32-warmup decisions, not every empirical sample mean."},
        "numerics": "Float64 evaluation of exact finite-sum formulas, not interval-arithmetic certification. Monotone lower/upper refinement and total transition mass checked to 1e-12. No Monte Carlo standard errors.",
        "finite_episode_note": "The stationary upper also bounds expected finite-prefix accuracy because the reset latent prior is stationary and more past information cannot reduce Bayes accuracy. The stationary suffix lower is NOT a lower bound for early finite prefixes. Length-t lower is exact for t visible tokens from the stationary prior.",
        "method": "Exact enumeration over token strings. Lower sums max(prior @ K_history @ emission); genie upper sums prior[state] * max((K_history @ emission)[state]). The genie reveals the hidden source state before the suffix, not the pending token. No Monte Carlo or confidence intervals.",
    }


def numerical_verification(recipes):
    from envs.strata import model as domain
    from experiments.strata_two_factor_explore_cycle_3.design import exact_baselines, identifiability_summary
    parameters = {"alpha": 0.98, "t0": 0.30, "t1": 0.80}
    kernels = domain.controlled_kernels(**parameters, strength=1.0)
    recorded = recipes["explore_cycle_3_both"]["analytic_design"]
    np.testing.assert_allclose(kernels, recorded["kernels"], atol=1e-14, rtol=0)
    exact = exact_baselines(kernels)
    np.testing.assert_allclose(exact["reward_states"]["0"]["best_reactive"], recorded["baselines"]["reward_states"]["0"]["best_reactive"], atol=1e-14)
    library_root = Path(harness.__file__).resolve().parents[1]
    if not Path(domain.__file__).resolve().is_relative_to(library_root):
        raise ValueError("Strata domain and harness must come from the same installed source checkout")
    library_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=library_root, text=True).strip()
    if library_commit != "32a15242b01450472c522f0bac250e1a2f7e2a6d":
        raise ValueError("use the exact recorded training library at 32a15242")
    new_token_reference = token_references(parameters)
    refined_path = ROOT / "experiments/strata_token_guess_cycle_2/ppo/results/parameter_reference_20260909.json"
    if refined_path.exists():
        refined = load(refined_path)
        assert refined["parameters"] == parameters
        assert refined["finite_prefix_lower"] <= refined["stationary_bayes_upper"]
        new_token_reference["renewal_refined_bounds"] = refined
        new_token_reference["renewal_refinement_source"] = {"path": relative(refined_path), "sha256": digest(refined_path)}
    return {
        "python": sys.version,
        "packages": {package: importlib.metadata.version(package) for package in ("numpy", "matplotlib", "gymnasium", "ray", "torch", "rl-harness")},
        "domain_file": domain.__file__,
        "domain_sha256": digest(Path(domain.__file__)),
        "analysis_library_commit": library_commit,
        "compatibility": "Exact recorded training library 32a15242; explicit parameters and recorded controlled kernels match to 1e-14. No library defaults patched. This extraction does not load checkpoints.",
        "token_cycle_1_context": token_references({"alpha": 0.97, "t0": 0.38, "t1": 0.54}),
        "token_cycle_2": new_token_reference,
        "explore_cycle_3": {"state_0_baselines": exact["reward_states"]["0"], "identifiability": identifiability_summary(kernels)},
        "explore_cycle_2_code_only_reference": exact_baselines(domain.controlled_kernels(alpha=0.97, t0=0.38, t1=0.54, strength=1.0))["reward_states"]["0"],
        "historical_recipe_tests": {"command": "python -m pytest -q tests/test_strata_token_guess_cycle_2.py tests/test_strata_two_factor_explore_cycle_3.py", "archived_result": "16 passed using exact training library 32a15242 via PYTHONPATH, without patched defaults.", "execution_note": "Historical validation from the original report; this generator does not run pytest or certify the current experiment source."},
    }


def extract(output, *, no_plots=False):
    tables = {key: [] for key in ("probes", "endpoints", "policy", "success", "cev", "training_returns", "run_status")}
    sources = []
    summary = {
        "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "generator": {"path": relative(Path(__file__).resolve()), "sha256": digest(Path(__file__)), "portability_note": "Paths resolve from the installed packages and this repository. Archived report/source hashes describe original generation, not this normalized generator."},
        "scope": "Three new conditions plus old token cycle 1 as separately labeled compact-results context; no old reviewed geometry scores pooled.",
        "source_selection": "checkpoint_reports JSON objects, sorted by agent_steps; initialization must be actual step zero; final is max agent_steps, never best test layer/checkpoint. Per-checkpoint probe_battery must exactly equal its summary entry.",
        "units": {"success": "100 * held-out policy.mean_reward", "both_reward": "percentage of rewarded-factor arrival-state indicators averaged over two factors, NOT joint success", "belief_mse": "mean squared error per coordinate", "global_r2": "1 - mse / mean centered test-target variance", "null_r2": "R2 of unit emission-null contrast of the full affine belief prediction; not an independently optimized scalar probe"},
        "protocol": {
            "rollouts": "Per checkpoint 20,000 fit and 20,000 test retained post-warmup rows, 8 vector environments, first 32 decisions per episode omitted, 1024-step episodes with randomized first length; stochastic learned policy at temperature 1.5.",
            "splits": "Independent named fit/test streams. Ten-fold random ROW CV inside fit selects SVD rcond, not stream/episode-grouped inner CV. Test streams independent, but adjacent rows correlated; checkpoints reuse seed roots. No synthetic CIs.",
            "distribution": "Passive token-history distribution for token task; checkpoint-dependent learned-policy occupancy for explore. Comparisons between explore checkpoints/arms are not common-distribution matched probes.",
            "controls": "Token compact reports have train mean/current visible token/shuffled label only: NTP/log-NTP absent. Explore additionally has exact-belief NTP and log-NTP affine baselines and joint-token/preceding-action cell means. Delta decoding above an NTP baseline is not evidence of causal use.",
            "training_return": "Raw compact training curves are episode-return aggregates, not frozen checkpoint success. Early randomized episode lengths are not exported in curves, so do NOT divide all returns by 1024. Final tune_summary episode length is checked before final normalization.",
            "targets": "Token: delayed arrival belief (source @ transition), pending-token probability is source @ emission, not arrival @ emission. Explore: current predictive belief from full token/executed-action history; reward uses next arrival. Neither actor observes rewards. Belief target may use more than actor's strict 32-frame history.",
            "geometry": "Activation CEV is whole residual geometry, not a belief-aligned rank or proof of factor-subspace orthogonality. Product consistency confirms factorized beliefs, not factorized neural representations.",
        },
        "code_only": {"strata_two_factor_explore_cycle_2": "Recipes exist at old alpha=.97,t0=.38,t1=.54; no committed run results in this checkout."},
        "runs": {},
    }
    recipes = {}
    for name, location in RUNS.items():
        folder = ROOT / "experiments" / location
        condition = load(folder / "condition_summary.json")
        recipe = load(folder / "resolved_recipe.json")
        recipes[name] = recipe
        manifest = load(folder / "run_manifest.json")
        tune = load(folder / "tune_summary.json")
        assert tune["num_trials"] == 1
        trial = tune["trials"][0]
        reports = sorted(condition["checkpoint_reports"], key=lambda report: report["agent_steps"])
        assert reports[0]["agent_steps"] == 0 and reports[0]["is_initialization"]
        final_steps = reports[-1]["agent_steps"]
        latest = [report for report in reports if report["agent_steps"] == final_steps]
        assert len(latest) == 1
        final_metrics = trial["metrics"]
        budget = recipe["total_env_steps"]
        sampled = final_metrics["env_runners/num_env_steps_sampled_lifetime"]
        assert sampled == final_steps and sampled >= budget
        assert manifest["status"] == "completed" and manifest["error"] is None and trial["error"] is None
        assert condition["seed"] == recipe["seed"] == manifest["runtime"]["seed"]
        status = {
            "run": name, "run_id": folder.name, "seed": condition["seed"], "seed_count": 1,
            "status": manifest["status"], "remote_artifacts_status": manifest.get("remote_artifacts", {}).get("status"),
            "budget": budget, "final_steps": final_steps, "completion_percent": 100 * final_steps / budget,
            "final_iteration": reports[-1]["training_iteration"], "checkpoint_count": len(reports),
            "train_return_mean": final_metrics["env_runners/episode_return_mean"],
            "train_episode_len_mean": final_metrics["env_runners/episode_len_mean"],
            "train_episode_len_min": final_metrics["env_runners/episode_len_min"],
            "train_episode_len_max": final_metrics["env_runners/episode_len_max"],
            "train_final_success_percent": 100 * final_metrics["env_runners/episode_return_mean"] / final_metrics["env_runners/episode_len_mean"],
            "experiment_commit": manifest["git"]["experiment_repository"]["commit"],
            "experiment_dirty": manifest["git"]["experiment_repository"]["dirty"],
            "library_commit": manifest["git"]["library"]["commit"],
            "library_dirty": manifest["git"]["library"]["dirty"],
            "started_at": manifest["started_at"], "ended_at": manifest["ended_at"],
            "remote_artifacts_uri": manifest.get("remote_artifacts", {}).get("base_uri"),
        }
        tables["run_status"].append(status)
        run = {"status": status, "source_dir": relative(folder), "recipe": {key: value for key, value in recipe.items() if key != "analytic_design"}, "framework_versions": manifest["framework_versions"], "endpoints": {}, "success_curve": []}
        if "analytic_design" in recipe:
            run["design_state_0_reference"] = recipe["analytic_design"]["baselines"]["reward_states"]["0"]
        for file_name in ("condition_summary.json", "resolved_recipe.json", "run_manifest.json", "tune_summary.json", "training_curves.jsonl"):
            path = folder / file_name
            sources.append({"path": relative(path), "sha256": digest(path)})
        for line_number, line in enumerate((folder / "training_curves.jsonl").read_text().splitlines(), 1):
            tables["training_returns"].append({"run": name, "source_line": line_number, **json.loads(line)})
        for report in reports:
            steps = report["agent_steps"]
            battery = folder / "checkpoint_probes" / f"steps_{steps:09d}" / "probe_battery.json"
            assert load(battery) == report
            sources.append({"path": relative(battery), "sha256": digest(battery)})
            phase = "initial" if steps == 0 else "final" if steps == final_steps else "intermediate"
            base = {"run": name, "seed": condition["seed"], "agent_steps": steps, "iteration": report["training_iteration"], "phase": phase, "n_fit": report["n_fit"], "n_test": report["n_test"], "source": relative(battery)}
            assert report["n_fit"] == report["n_test"] == 20000
            success = {**base, "success_percent": 100 * report["policy"]["mean_reward"], "train_probe_success_percent": 100 * report["train_policy"]["mean_reward"]}
            tables["success"].append(success)
            run["success_curve"].append({key: success[key] for key in ("agent_steps", "success_percent")})
            for split in ("policy", "train_policy"):
                for pointer, value in flatten_values(report[split]):
                    tables["policy"].append({**base, "split": split, "json_pointer": f"/{split}{pointer}", "value": value})
            for layer, layer_report in report["layers"].items():
                for probe_type in ("probe_fits", "shuffled_training_labels"):
                    for factor, value in layer_report[probe_type].items():
                        row = {**base, "layer": layer, "factor": factor, "probe_type": probe_type, "json_pointer": f"/layers/{layer}/{probe_type}/{factor}", **metrics(value)}
                        tables["probes"].append(row)
                        if phase != "intermediate":
                            tables["endpoints"].append(row)
                cev = layer_report["cev"]
                tables["cev"].append({**base, "target": layer, **{key: cev[key] for key in ("cev90_dimension", "cev95_dimension", "cev99_dimension", "participation_ratio", "rank")}, "cev_first_8": cev["cumulative_explained_variance"][:8]})
            for control, factors in report["controls"].items():
                for factor, value in factors.items():
                    row = {**base, "layer": "control", "factor": factor, "probe_type": control, "json_pointer": f"/controls/{control}/{factor}", **metrics(value)}
                    tables["probes"].append(row)
                    if phase != "intermediate":
                        tables["endpoints"].append(row)
            for target, cev in report["cev"].items():
                tables["cev"].append({**base, "target": target, **{key: cev[key] for key in ("cev90_dimension", "cev95_dimension", "cev99_dimension", "participation_ratio", "rank")}, "cev_first_8": cev["cumulative_explained_variance"][:8]})
            if phase != "intermediate":
                run["endpoints"][phase] = {
                    "agent_steps": steps, "success_percent": success["success_percent"], "policy": report["policy"],
                    "metadata": report["metadata"], "product_consistency_max_abs": report["product_consistency_max_abs"],
                    "layers": {layer: {"belief": {factor: metrics(value) for factor, value in layer_report["probe_fits"].items()}, "shuffled": {factor: metrics(value) for factor, value in layer_report["shuffled_training_labels"].items()}, "cev95": layer_report["cev"]["cev95_dimension"]} for layer, layer_report in report["layers"].items()},
                    "controls": {control: {factor: metrics(value) for factor, value in factors.items()} for control, factors in report["controls"].items()},
                }
        curve = run["success_curve"]
        peak = max(curve, key=lambda row: row["success_percent"])
        run["descriptive_curve_diagnostics"] = {
            "highest_saved_success_percent": peak["success_percent"],
            "highest_saved_success_steps": peak["agent_steps"],
            "last_minus_initial_pp": curve[-1]["success_percent"] - curve[0]["success_percent"],
            "highest_saved_minus_last_pp": peak["success_percent"] - curve[-1]["success_percent"],
            "warning": "Descriptive maximum among saved checkpoint evaluations, not validation-selected model performance. Final results always use max agent_steps.",
        }
        summary["runs"][name] = run
    assert not list((ROOT / "experiments/strata_two_factor_explore_cycle_2").glob("*/results/*/condition_summary.json"))
    summary["numerical_verification"] = numerical_verification(recipes)
    summary["sources"] = sources
    compact_fields = ("run", "phase", "layer", "factor", "mse", "target_variance", "global_r2", "null_mse", "null_global_r2")
    tables["key_metrics"] = [{key: row[key] for key in compact_fields} for row in tables["endpoints"] if row["probe_type"] == "probe_fits"]
    tables["control_metrics"] = [{key: row[key] for key in (*compact_fields, "probe_type")} for row in tables["endpoints"] if row["layer"] == "control" or (row["layer"] == "layer_3" and row["probe_type"] == "shuffled_training_labels")]
    summary["table_rows"] = {name: len(rows) for name, rows in tables.items()}
    summary["limitations"] = [
        "One seed (42) per condition. No across-seed uncertainty estimates or causal-use conclusions.",
        "Both/single explore budgets differ (50M/30M), the controlled factors have different occupancies, and independent train/test seeds are shared across checkpoints; do not claim a controlled causal reward-relevance effect.",
        "Token versus explore shares transformer width/depth/heads/context but differs in inputs, action space, number of factors, target timing, dynamics/control, gamma, learning rate, entropy, and budget.",
        "The explore full-information 98% upper is NOT Bayes/POMDP optimal. Exact constant and deterministic current-token-only references have narrower information classes than the actor.",
        "A new token-parameter Bayes upper is computed by exact suffix/genie enumeration. No old Strata1 Bayes number is reused. No optimal 32-frame explore controller is known here.",
        "Fresh matched-distribution/grouped-CV probes and the token NTP/log-NTP controls require checkpoint/rollout analysis; compact aggregates cannot reconstruct them or confidence intervals.",
    ]
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        dump_csv(output / f"{name}.csv", rows)
    if not no_plots:
        plot(summary, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, run in summary["runs"].items():
        print(name, json.dumps(run["descriptive_curve_diagnostics"]))
    for name in ("token_cycle_1_context", "token_cycle_2"):
        reference = summary["numerical_verification"][name]
        print(name, "BOUNDS", reference["stationary_bayes_suffix_bounds"][-1])
    print("OUTPUT", output)


def plot(summary, output):
    names = ["token_cycle_2", "explore_cycle_3_both", "explore_cycle_3_factor_1"]
    titles = ["Token guessing · cycle 2", "Explore · both factors rewarded", "Explore · factor 1 rewarded"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    references = summary["numerical_verification"]
    for index, (axis, name, title) in enumerate(zip(axes, names, titles)):
        run = summary["runs"][name]
        curve = run["success_curve"]
        steps = np.array([row["agent_steps"] for row in curve])
        scores = np.array([row["success_percent"] for row in curve])
        axis.plot(steps / 1e6, scores, "o-", linewidth=2, markersize=4.5, label="Held-out stochastic policy")
        axis.scatter([0], [scores[0]], s=75, marker="D", zorder=4, label=f"Actual init: {scores[0]:.2f}%")
        axis.annotate(f"Final: {scores[-1]:.2f}%", (steps[-1] / 1e6, scores[-1]), xytext=(-6, -18), textcoords="offset points", ha="right", fontsize=10)
        if index == 0:
            reference = references[name]
            bound = reference.get("renewal_refined_bounds")
            upper = bound["finite_prefix_upper"] if bound else reference["stationary_bayes_suffix_bounds"][-1]["upper"]
            axis.axhline(100 * upper, color="#b42318", linestyle="--", label=f"Bayes upper: {100 * upper:.3f}% (numerical)")
            axis.axhline(100 * reference["best_current_visible_token_accuracy"], color="#26734d", linestyle=":", label=f"Best 1-token rule: {100 * reference['best_current_visible_token_accuracy']:.2f}%")
            axis.set_ylabel("Task accuracy / mean rewarded-factor success (%)")
        else:
            reference = run["design_state_0_reference"]
            axis.axhline(100 * reference["oracle_upper_bound"], color="#b42318", linestyle="--", label="Full-info upper: 98% (NOT Bayes)")
            axis.axhline(100 * reference["best_reactive"], color="#26734d", linestyle=":", label="Best deterministic token rule: 47.90%")
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Environment steps (millions; symlog)")
        axis.set_xscale("symlog", linthresh=0.04, linscale=0.7)
        axis.set_xlim(-0.008, float(steps[-1] / 1e6) * 1.06)
        ticks = [0, 0.1, 1, 2.5] if index == 0 else [0, 0.1, 1, 10, 50 if index == 1 else 30]
        axis.set_xticks(ticks, labels=[str(tick) for tick in ticks])
        axis.set_ylim(25, max(102, scores.max() + 3))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    fig.suptitle("Strata new conditions · seed 42 · actual initialization and all saved checkpoints", fontsize=14)
    fig.text(0.5, 0.01, "20,000 post-warmup test decisions/checkpoint; learned stochastic policy, temperature 1.5. No CIs.\nBoth-reward score averages factor indicators, not a joint-success event. Reference lines are analytical expectations, not empirical CIs.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output / "success_over_training.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output", type=Path, default=OUTPUT,
                        help="New report directory; existing paths are never overwritten.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Output path already exists: {args.output_dir}. Use --output-dir NEW_DIRECTORY.")
    extract(args.output_dir, no_plots=args.no_plots)


if __name__ == "__main__":
    main()
