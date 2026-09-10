from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import harness


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/gol_reward_state_action_symmetry_cycle_1"
OUTPUT = STUDY / "results/branch_analysis_20260909"
RUNS = (
    ("variant_2", "20260909T084818Z-6603a47f"),
    ("variant_2", "20260909T104731Z-c4aff136"),
    ("variant_2_quarter", "20260909T111613Z-484ad76e"),
    ("variant_3", "20260909T110146Z-8c31130d"),
    ("variant_3_quarter", "20260909T113037Z-a43b263e"),
)
TARGETS = ("fine_belief", "coarse_belief", "m1_minus_m2", "e_minus_s")
LABELS = ("Fine belief", "Coarse belief", "M1 - M2", "E - S")
METRICS = (
    "r_squared", "mse", "target_variance", "global_mse_ratio",
    "branch_baseline_mse", "fine_evaluation_mse", "fine_mse_ratio",
    "fine_mse_improvement", "n_evaluated",
)
CONTROLS = (
    "previous_action_latest_token", "next_token", "next_reward",
    "combined_marginals", "categorical_plus_marginals",
    "single_permuted_training_labels", "shuffled_history_broken_alignment",
)


def load(path):
    return json.loads(path.read_text())


def rel(path):
    return str(path.relative_to(ROOT))


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def provenance():
    library = Path(harness.__file__).resolve().parents[1]
    expected_library = "32a15242b01450472c522f0bac250e1a2f7e2a6d"
    recorded_experiment = "1db27ffc096aac88e9281528675f0e86273a6d0b"
    local_library = git("rev-parse", "HEAD", cwd=library)
    library_status = git("status", "--porcelain", cwd=library)
    paths = [f"experiments/gol_reward_state_action_symmetry_cycle_1/{name}.py" for name in ("analysis", "shared", "process", "design")]
    return {
        "experiment_checkout": git("rev-parse", "HEAD"),
        "experiment_branch": git("branch", "--show-current") or "detached HEAD",
        "completed_runs_recorded_experiment_commit": recorded_experiment,
        "scientific_source_diff_from_recorded_experiment": git("diff", recorded_experiment, "--", *paths),
        "manifest_library_commit_all_five_runs": expected_library,
        "local_library_path": str(library), "local_library_commit": local_library,
        "local_library_clean": not library_status,
        "local_library_exact_revision_match": local_library == expected_library,
        "library_commit_changed_files": git("diff", "--name-only", expected_library, "HEAD", cwd=library).splitlines(),
        "library_relevant_code_diff": git("diff", expected_library, "HEAD", "--", "envs/gol", "envs/hmm", "learners", "analysis", "harness", cwd=library),
        "generator": {"path": rel(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "execution": "Saved-data extraction and optional plotting only; the harness package is imported to locate its installed source checkout. No environment/design execution, checkpoint loading, fitting, or rollouts. Git records describe this invocation; no source-equivalence or clean-checkout claim is inferred from a revision alone.",
    }


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def write_csv(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def extract():
    records, metrics, policies, inventory, sources = [], [], [], [], []
    checks = []
    for arm, run_id in RUNS:
        directory = STUDY / arm / "results" / run_id
        manifest = load(directory / "run_manifest.json")
        recipe = load(directory / "resolved_recipe.json")
        tune = load(directory / "tune_summary.json")
        trial = tune["trials"][0]
        final_metrics = trial["metrics"]
        curve_path = directory / "training_curves.jsonl"
        training = [json.loads(line) for line in curve_path.read_text().splitlines()] if curve_path.exists() else []
        report_paths = sorted(directory.glob("checkpoint_probes/steps_*/probe_*.json"))
        reports = [load(path) for path in report_paths]
        reports_with_paths = sorted(zip(reports, report_paths), key=lambda pair: pair[0]["agent_steps"])
        summary_path = directory / "condition_summary.json"
        summary = load(summary_path) if summary_path.exists() else None
        if summary is not None:
            assert summary["checkpoint_reports"] == [report for report, _ in reports_with_paths]
            checks.append({"run_id": run_id, "check": "condition_summary_matches_individual_reports", "passed": True})
        benchmark = recipe["analytic_design"]["variants"][f"variant_{recipe['variant']}"]
        steps = final_metrics["env_runners/num_env_steps_sampled_lifetime"]
        row = {
            "arm": arm, "run_id": run_id, "status": manifest["status"],
            "trial_status": trial["status"], "seed": manifest["runtime"]["seed"],
            "speed": recipe["speed"], "variant": recipe["variant"],
            "resume_from": manifest["runtime"]["resume_from"],
            "budget": recipe["total_env_steps"], "completed_steps": steps,
            "budget_fraction": steps / recipe["total_env_steps"],
            "training_iterations": final_metrics["training_iteration"],
            "completed_budget": manifest["status"] == "completed" and steps >= recipe["total_env_steps"],
            "probe_checkpoints": len(reports), "training_curve_rows": len(training),
            "episodes_completed": final_metrics["env_runners/num_episodes_lifetime"],
            "experiment_revision": manifest["git"]["experiment_repository"]["commit"],
            "experiment_dirty": manifest["git"]["experiment_repository"]["dirty"],
            "library_revision": manifest["git"]["library"]["commit"],
            "library_dirty": manifest["git"]["library"]["dirty"],
            "manifest_source": rel(directory / "run_manifest.json"),
            "tune_source": rel(directory / "tune_summary.json"),
            "error": trial["error"],
        }
        inventory.append(row)
        if row["completed_budget"]:
            assert [report["training_iteration"] for report, _ in reports_with_paths] == [0, 1, 2, 4, 8, 16, 32, 64, 77]
            assert reports_with_paths[0][0]["agent_steps"] == 0
            assert reports_with_paths[-1][0]["agent_steps"] == steps == 2524368
            assert len(training) == 77 and training[-1]["steps"] == steps
            assert not any(key in point for point in training for key in ("return_mean", "episode_return_mean"))
        record = {"inventory": row, "benchmark": benchmark, "analytic_design": recipe["analytic_design"], "resolved_recipe": recipe,
                  "framework_versions": manifest["framework_versions"], "training_curves": training, "checkpoints": []}
        for index, (report, path) in enumerate(reports_with_paths):
            assert (report["variant"], report["speed"]) == (recipe["variant"], recipe["speed"])
            assert (report["n_fit"], report["n_test"], report["metadata"]["n_envs_per_split"]) == (10000, 10000, 16)
            assert set(report["controls"]) == set(CONTROLS)
            assert report["metadata"]["sequence_lookback"] == 192
            point = dict(report)
            point["source"] = rel(path)
            point["missing_controls"] = {"log_next_token": "not fitted or saved; not reconstructed"}
            record["checkpoints"].append(point)
            common = {"arm": arm, "run_id": run_id, "variant": report["variant"], "speed": report["speed"],
                      "phase": "initial" if index == 0 else "final" if index == len(reports) - 1 else "intermediate",
                      "agent_steps": report["agent_steps"], "training_iteration": report["training_iteration"],
                      "checkpoint_label": report["checkpoint_label"], "source": rel(path)}
            for control, targets in {"post_final_layer_norm": report["probe_fits"], **report["controls"]}.items():
                for target in TARGETS:
                    values = targets[target]
                    assert values["n_evaluated"] == 10000
                    assert math.isclose(values["r_squared"], 1.0 - values["mse"] / values["target_variance"], abs_tol=1e-12)
                    assert values["target_variance"] == report["probe_fits"][target]["target_variance"]
                    ci = values.get("mse_environment_bootstrap_ci95", [None, None])
                    metrics.append({**common, "target": target, "control": control, "availability": "saved",
                                    **{key: values[key] for key in METRICS},
                                    "mse_ci95_low": ci[0], "mse_ci95_high": ci[1],
                                    "n_fit": report["n_fit"], "n_test": report["n_test"],
                                    "bootstrap_clusters": 16 if ci[0] is not None else None,
                                    "bootstrap_resamples": 200 if ci[0] is not None else None,
                                    "bootstrap_unit": "held_out_environment_trajectory_fixed_probe" if ci[0] is not None else None})
            for target in TARGETS:
                metrics.append({**common, "target": target, "control": "log_next_token", "availability": "not_fitted_or_saved"})
            policy = report["policy"]
            lookup = report["previous_action_latest_token_controller"]
            policies.append({**common, "reward_mean": policy["reward_mean"], "expected_next_reward_mean": policy["expected_next_reward_mean"],
                             "n_evaluated": policy["n_evaluated"], "lookup_reward_mean": lookup["reward_mean"],
                             "lookup_expected_next_reward_mean": lookup["expected_next_reward_mean"], "lookup_n_evaluated": lookup["n_evaluated"],
                             "uniform_random_analytic": benchmark["uniform_random_action_occupancy"],
                             "best_constant_analytic": max(benchmark["constant_action_occupancies"]),
                             "full_information_upper_analytic_not_bayes": benchmark["full_information_occupancy_upper_bound"],
                             "fraction_random_to_full_information_gap_closed": (policy["reward_mean"] - benchmark["uniform_random_action_occupancy"]) / (benchmark["full_information_occupancy_upper_bound"] - benchmark["uniform_random_action_occupancy"]),
                             **{f"action_fraction_{action}": value for action, value in zip(("a1", "a2", "aE", "aS"), policy["action_fractions"])},
                             **{f"constant_{action}_analytic": value for action, value in zip(("a1", "a2", "aE", "aS"), benchmark["constant_action_occupancies"])}})
        source_paths = [directory / name for name in ("run_manifest.json", "resolved_recipe.json", "tune_summary.json")]
        source_paths += report_paths + ([curve_path] if curve_path.exists() else []) + ([summary_path] if summary_path.exists() else [])
        sources.extend({"source": rel(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_paths)
        records.append(record)
    assert len(policies) == 36
    assert sum(row["availability"] == "saved" for row in metrics) == 1152
    assert sum(row["availability"] == "not_fitted_or_saved" for row in metrics) == 144
    checks.append({"check": "all_saved_metric_R2_MSE_variance_identities_and_sample_counts", "passed": True})
    return records, metrics, policies, inventory, sources, checks


def plots(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    completed = [record for record in records if record["inventory"]["completed_budget"]]
    colors = ("#0072B2", "#56B4E9", "#D55E00", "#CC79A7")

    def xaxis(ax):
        ax.set_xscale("symlog", linthresh=32784)
        ax.set_xlim(-1000, 2700000)
        ax.set_xticks([0, 32784, 262272, 1049088, 2524368])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "Init" if x == 0 else f"{x / 1e6:.2f}M"))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Environment steps (symlog; initialization included)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, record in zip(axes.flat, completed):
        cp = record["checkpoints"]
        b = record["benchmark"]
        steps = [point["agent_steps"] for point in cp]
        ax.plot(steps, [point["policy"]["reward_mean"] for point in cp], "o-", label="Learned policy: mean reward")
        ax.plot(steps, [point["policy"]["expected_next_reward_mean"] for point in cp], ":", label="Learned policy: predicted mean reward")
        ax.plot(steps, [point["previous_action_latest_token_controller"]["reward_mean"] for point in cp], "s--", label="Myopic branch lookup (fresh rollout)")
        ax.axhline(b["uniform_random_action_occupancy"], color="gray", linestyle=":", label="Uniform random (analytic)")
        ax.axhline(max(b["constant_action_occupancies"]), color="gray", linestyle="--", label="Best constant (analytic)")
        ax.axhline(b["full_information_occupancy_upper_bound"], color="black", linestyle="-.", label="Full-information upper (not Bayes)")
        ax.set_title(record["inventory"]["arm"] + ", seed 42")
        ax.set_ylabel("Reward per continuing step")
        ax.set_ylim(0.05, 0.45)
        xaxis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("GOL task performance: 10,000 held-out steps/checkpoint; no policy-return CI saved")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(OUTPUT / "task_success_from_initialization.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, target, label in zip(axes.flat, TARGETS, LABELS):
        for color, record in zip(colors, completed):
            cp = record["checkpoints"]
            ax.plot([p["agent_steps"] for p in cp], [p["probe_fits"][target]["r_squared"] for p in cp], "o-", color=color, label=record["inventory"]["arm"])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(label)
        ax.set_ylabel("Held-out affine R-squared")
        ax.set_ylim(-0.2, 1.0)
        xaxis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("GOL affine accessibility, not causal use: each checkpoint has its own on-policy occupancy")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUTPUT / "geometry_r2_from_initialization.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(17, 14))
    for row, record in enumerate(completed):
        cp = record["checkpoints"]
        steps = [p["agent_steps"] for p in cp]
        for col, target in enumerate(TARGETS):
            ax = axes[row, col]
            values = [p["probe_fits"][target] for p in cp]
            ax.plot(steps, [v["mse"] for v in values], "o-", label="Probe MSE")
            ax.fill_between(steps, [v["mse_environment_bootstrap_ci95"][0] for v in values], [v["mse_environment_bootstrap_ci95"][1] for v in values], alpha=0.2, label="Fixed-probe environment bootstrap 95%")
            ax.plot(steps, [v["target_variance"] for v in values], "--", color="gray", label="Target variance")
            ax.plot(steps, [p["controls"]["previous_action_latest_token"][target]["mse"] for p in cp], ":", color="black", label="Train-fitted branch MSE")
            ax.set_title(record["inventory"]["arm"] + " / " + LABELS[col], fontsize=10)
            ax.set_ylabel("Mean squared coordinate error")
            ax.set_ylim(bottom=0)
            xaxis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("GOL geometry errors and changing target variance; 16 held-out trajectories, 200 resamples, fixed probes")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUTPUT / "geometry_mse_variance_from_initialization.png", dpi=150)
    plt.close(fig)


def print_digest(records):
    for record in records:
        inv = record["inventory"]
        print(inv["arm"], inv["run_id"], inv["status"], inv["completed_steps"], inv["training_iterations"])
        if not record["checkpoints"]:
            continue
        print("benchmarks", json.dumps(record["benchmark"], separators=(",", ":")))
        for cp in (record["checkpoints"][0], record["checkpoints"][-1]):
            print(cp["agent_steps"], "policy", cp["policy"]["reward_mean"], "lookup", cp["previous_action_latest_token_controller"]["reward_mean"])
            for target in TARGETS:
                values = cp["probe_fits"][target]
                print(target, "r2", values["r_squared"], "mse", values["mse"], "variance", values["target_variance"])
                print("controls", json.dumps({name: cp["controls"][name][target]["r_squared"] for name in CONTROLS}, separators=(",", ":")))
        print("trajectory", json.dumps([{ "steps": cp["agent_steps"], "reward": cp["policy"]["reward_mean"], **{target: cp["probe_fits"][target]["r_squared"] for target in TARGETS}} for cp in record["checkpoints"]], separators=(",", ":")))


def main(argv=None):
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT,
                        help="New report directory; existing paths are never overwritten.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Output path already exists: {args.output_dir}. Use --output-dir NEW_DIRECTORY.")
    records, metrics, policies, inventory, sources, checks = extract()
    payload = {"schema_version": 1, "experiment_checkout": git("rev-parse", "HEAD"),
               "generator": {"path": rel(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "scope": "Existing saved summaries only. No checkpoint loading, model fitting, design execution or rollouts.",
               "sampling": {"n_fit": 10000, "n_test": 10000, "n_envs_each_split": 16, "selected_steps_per_environment": 625,
                            "warmup_per_environment": 64, "selected_decision_times_inclusive": [64, 688],
                            "bootstrap_resamples": 200, "bootstrap_unit": "whole held-out environment trajectory, fixed trained network and fitted probe",
                            "same_training_seed_all_runs": 42, "independent_model_seeds_per_arm": 1,
                            "finite_raw_observation_lookback": 192, "max_frames_including_current": 193,
                            "selected_fraction_before_full_context": 0.2048},
               "missing": {"log_next_token": "No saved fit; do not substitute next_token or fit a new probe.",
                           "input_shuffle_with_recomputed_beliefs": "Not implemented; shuffled histories retain original labels.",
                           "joint_reward_token_probe": "Not saved; combined_marginals contains separate marginals only.",
                           "episode_returns": "No completed episodes in this continuing task.",
                           "policy_return_confidence_intervals": "Not saved; no new bootstrap from unavailable raw samples.",
                           "geometry_scatter": "No raw activations/beliefs in compact results; curves are decodability, not geometric topology."},
               "checks": checks, "sources": sources, "runs": records}
    verification = provenance()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    OUTPUT = args.output_dir
    write_json("saved_results.json", payload)
    write_json("provenance_verification.json", verification)
    write_csv("run_inventory.csv", inventory)
    write_csv("all_probe_metrics.csv", metrics)
    write_csv("endpoint_probe_metrics.csv", [row for row in metrics if row["phase"] in ("initial", "final")])
    write_csv("policy_trajectories.csv", policies)
    write_csv("endpoint_policy_metrics.csv", [row for row in policies if row["phase"] in ("initial", "final")])
    if not args.no_plots:
        plots(records)
    print_digest(records)
    print("OUTPUT", OUTPUT)


if __name__ == "__main__":
    main()
