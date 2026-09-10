from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


CHECKPOINT_ORDER = ("init", "penultimate", "final")
CHECKPOINT_LABELS = {
    "init": "initialization",
    "penultimate": "penultimate",
    "final": "final",
}
RUNS = {
    "token_guess": {
        "label": "Token guess",
        "condition_summary": (
            "experiments/strata_token_guess_cycle_2/"
            "context_length_64_entropy_4x/ppo/results/"
            "20260909T190939Z-ea45871d/condition_summary.json"
        ),
        "run_manifest": (
            "experiments/strata_token_guess_cycle_2/"
            "context_length_64_entropy_4x/ppo/results/"
            "20260909T190939Z-ea45871d/run_manifest.json"
        ),
        "selected": {
            "init": "initial_checkpoint",
            "penultimate": "iteration_000256_steps_002138080",
            "final": "checkpoint_000000",
        },
    },
    "reward_both": {
        "label": "Reward both",
        "condition_summary": (
            "experiments/strata_two_factor_explore_cycle_3/"
            "context_length_64_entropy_4x/reward_both_state_0/results/"
            "20260909T191941Z-20ee67e0/condition_summary.json"
        ),
        "run_manifest": (
            "experiments/strata_two_factor_explore_cycle_3/"
            "context_length_64_entropy_4x/reward_both_state_0/results/"
            "20260909T191941Z-20ee67e0/run_manifest.json"
        ),
        "selected": {
            "init": "initial_checkpoint",
            "penultimate": "iteration_008192_steps_034305400",
            "final": "checkpoint_000000",
        },
    },
    "reward_factor_1": {
        "label": "Reward factor 1",
        "condition_summary": (
            "experiments/strata_two_factor_explore_cycle_3/"
            "context_length_64_entropy_4x/reward_factor_1_state_0/results/"
            "20260909T194203Z-b1c16992/condition_summary.json"
        ),
        "run_manifest": (
            "experiments/strata_two_factor_explore_cycle_3/"
            "context_length_64_entropy_4x/reward_factor_1_state_0/results/"
            "20260909T194203Z-b1c16992/run_manifest.json"
        ),
        "selected": {
            "init": "initial_checkpoint",
            "penultimate": "iteration_004096_steps_017153039",
            "final": "checkpoint_000000",
        },
    },
}
COLORS = {
    "init": "#7f7f7f",
    "penultimate": "#1f77b4",
    "final": "#d62728",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_archived(
    condition_summary: dict[str, Any],
    expected: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    trajectory = [
        {
            "checkpoint": item["checkpoint"],
            "agent_steps": item["agent_steps"],
            "training_iteration": item["training_iteration"],
            "mean_reward": item["policy"]["mean_reward"],
        }
        for item in condition_summary["checkpoint_reports"]
    ]
    by_label = {item["checkpoint"]: item for item in trajectory}
    try:
        selected = {
            key: by_label[checkpoint]
            for key, checkpoint in expected.items()
        }
    except KeyError as error:
        raise ValueError(f"missing selected checkpoint: {error}") from error
    return trajectory, selected


def _compact_factor(factor: dict[str, Any]) -> dict[str, Any]:
    return {
        "baselines": factor["baselines"],
        "layers": {
            name: {
                "metrics": values["metrics"],
                "comparisons": values["comparisons"],
                "nulls": values["nulls"],
                "activation_geometry": {
                    "effective_dimensions": values[
                        "activation_variance_geometry"
                    ]["effective_dimensions"],
                    "participation_ratio": values[
                        "activation_variance_geometry"
                    ]["participation_ratio"],
                    "rank": values[
                        "activation_variance_geometry"
                    ]["rank"],
                },
            }
            for name, values in factor["layers"].items()
        },
        "target_geometry": {
            "effective_dimensions": factor["target_variance_geometry"][
                "effective_dimensions"
            ],
            "participation_ratio": factor["target_variance_geometry"][
                "participation_ratio"
            ],
            "rank": factor["target_variance_geometry"]["rank"],
        },
        "control_definitions": factor["control_definitions"],
    }


def _load_inputs(
    root: Path,
    analysis_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "selected_checkpoint_design": (
            "initialization, penultimate, and final checkpoints only"
        ),
        "runs": {},
    }
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "analysis_inputs": {},
        "runs": {},
        "design": {
            "no_retraining": True,
            "selected_checkpoints_only": True,
            "checkpoint_count": 9,
            "randomize_first_episode_length": True,
            "layer_selection": "all layers reported; final layer fixed for summaries",
            "test_metrics_used_for_selection": False,
            "initialization_baseline": (
                "separately restored and sampled checkpoint; not matched-history initialization_features"
            ),
            "confidence_intervals": (
                "paired 95% percentile intervals from 200 held-out whole-episode bootstrap resamples"
            ),
        },
    }
    for run_name, config in RUNS.items():
        summary_path = root / config["condition_summary"]
        manifest_path = root / config["run_manifest"]
        condition_summary = _read_json(summary_path)
        manifest = _read_json(manifest_path)
        trajectory, archived = _selected_archived(
            condition_summary,
            config["selected"],
        )
        run_output = {
            "label": config["label"],
            "task_performance_trajectory": trajectory,
            "selected_checkpoints": {},
        }
        run_provenance = {
            "experiment_commit": manifest["git"]["experiment_repository"][
                "commit"
            ],
            "run_harness_commit": manifest["git"]["library"]["commit"],
            "run_seed": manifest["runtime"]["seed"],
            "b2": manifest["remote_artifacts"],
            "condition_summary": {
                "path": config["condition_summary"],
                "sha256": _sha256(summary_path),
            },
            "run_manifest": {
                "path": config["run_manifest"],
                "sha256": _sha256(manifest_path),
            },
            "checkpoints": {},
        }
        for checkpoint_key in CHECKPOINT_ORDER:
            analysis_path = (
                analysis_dir
                / f"{run_name}_{checkpoint_key}_summary.json"
            )
            analysis = _read_json(analysis_path)
            expected = archived[checkpoint_key]
            for field in ("checkpoint", "agent_steps", "training_iteration"):
                if analysis[field] != expected[field]:
                    raise ValueError(
                        f"{run_name} {checkpoint_key} {field} mismatch"
                    )
            if analysis["metadata"]["n_fit"] != 20_000:
                raise ValueError("analysis fit budget must be 20,000")
            if analysis["metadata"]["n_test"] != 20_000:
                raise ValueError("analysis test budget must be 20,000")
            run_output["selected_checkpoints"][checkpoint_key] = {
                "checkpoint": analysis["checkpoint"],
                "agent_steps": analysis["agent_steps"],
                "training_iteration": analysis["training_iteration"],
                "archived_task_mean_reward": expected["mean_reward"],
                "fresh_task_mean_reward": analysis["policy"]["mean_reward"],
                "task_mean_reward_difference": (
                    analysis["policy"]["mean_reward"]
                    - expected["mean_reward"]
                ),
                "policy": analysis["policy"],
                "factors": {
                    name: _compact_factor(factor)
                    for name, factor in analysis["factors"].items()
                },
                "sampling": {
                    key: analysis["metadata"][key]
                    for key in (
                        "sampling_distribution",
                        "policy_mode",
                        "policy_temperature",
                        "n_envs",
                        "warmup_per_episode",
                        "context_length",
                        "episode_length",
                        "n_fit",
                        "n_test",
                        "rollout_seed_spawn_keys",
                        "train_episodes_represented",
                        "test_episodes_represented",
                        "representation",
                        "target_timing",
                        "belief_conditioning",
                        "fit_method",
                        "bootstrap_unit",
                        "n_null_repeats",
                        "n_bootstrap_resamples",
                    )
                },
                "scope_warning": analysis["scope_warning"],
            }
            analysis_provenance = analysis["metadata"]["provenance"]
            run_provenance["checkpoints"][checkpoint_key] = {
                "checkpoint": analysis["checkpoint"],
                "agent_steps": analysis["agent_steps"],
                "training_iteration": analysis["training_iteration"],
                "summary_input": {
                    "file": analysis_path.name,
                    "sha256": _sha256(analysis_path),
                },
                "module_checkpoint_sha256": analysis_provenance[
                    "module_checkpoint_sha256"
                ],
                "analysis_repositories": analysis_provenance["repositories"],
                "analysis_source_sha256": analysis_provenance["source_sha256"],
                "framework_versions": analysis_provenance[
                    "framework_versions"
                ],
            }
        metrics["runs"][run_name] = run_output
        provenance["runs"][run_name] = run_provenance
    return metrics, provenance


def _metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run_name, run in metrics["runs"].items():
        for checkpoint_key in CHECKPOINT_ORDER:
            checkpoint = run["selected_checkpoints"][checkpoint_key]
            for factor_name, factor in checkpoint["factors"].items():
                for layer_name, layer in factor["layers"].items():
                    comparison = layer["comparisons"][
                        "nuisance/log_next_token_probability"
                    ]
                    nulls = layer["nulls"]
                    rows.append(
                        {
                            "run": run_name,
                            "checkpoint_role": checkpoint_key,
                            "checkpoint": checkpoint["checkpoint"],
                            "agent_steps": checkpoint["agent_steps"],
                            "training_iteration": checkpoint[
                                "training_iteration"
                            ],
                            "archived_task_mean_reward": checkpoint[
                                "archived_task_mean_reward"
                            ],
                            "fresh_task_mean_reward": checkpoint[
                                "fresh_task_mean_reward"
                            ],
                            "factor": factor_name,
                            "layer": layer_name,
                            "probe_r_squared": layer["metrics"]["r_squared"],
                            "probe_normalized_mse": layer["metrics"][
                                "normalized_mse"
                            ],
                            "null_contrast_r_squared": layer["metrics"][
                                "contrasts"
                            ]["prediction_null_1"]["r_squared"],
                            "current_observation_r_squared": factor[
                                "baselines"
                            ]["nuisance/current_observation"]["r_squared"],
                            "next_token_probability_r_squared": factor[
                                "baselines"
                            ]["nuisance/next_token_probability"]["r_squared"],
                            "log_next_token_probability_r_squared": factor[
                                "baselines"
                            ][
                                "nuisance/log_next_token_probability"
                            ]["r_squared"],
                            "probe_minus_log_ntp_r_squared": comparison[
                                "delta_r_squared"
                            ],
                            "probe_minus_log_ntp_ci_low": comparison[
                                "delta_r_squared_ci"
                            ][0],
                            "probe_minus_log_ntp_ci_high": comparison[
                                "delta_r_squared_ci"
                            ][1],
                            "permuted_label_null_r_squared_mean": nulls[
                                "permuted_labels"
                            ]["r_squared_mean"],
                            "gaussian_feature_null_r_squared_mean": nulls[
                                "gaussian_features"
                            ]["r_squared_mean"],
                            "matched_feature_null_r_squared_mean": nulls[
                                "matched_features"
                            ]["r_squared_mean"],
                            "activation_participation_ratio": layer[
                                "activation_geometry"
                            ]["participation_ratio"],
                            "activation_cev90_dimension": layer[
                                "activation_geometry"
                            ]["effective_dimensions"]["0.9"],
                        }
                    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _task_figure(metrics: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, (run_name, run) in zip(axes, metrics["runs"].items()):
        trajectory = run["task_performance_trajectory"]
        x = np.asarray([item["agent_steps"] for item in trajectory]) / 1e6
        y = np.asarray([item["mean_reward"] for item in trajectory])
        axis.plot(x, y, color="#444444", linewidth=1.5)
        for checkpoint_key in CHECKPOINT_ORDER:
            selected = run["selected_checkpoints"][checkpoint_key]
            axis.scatter(
                selected["agent_steps"] / 1e6,
                selected["archived_task_mean_reward"],
                color=COLORS[checkpoint_key],
                s=36,
                zorder=3,
                label=CHECKPOINT_LABELS[checkpoint_key],
            )
        axis.set_title(run["label"])
        axis.set_xlabel("environment steps (millions)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("held-out mean reward")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Task performance trajectory (archived checkpoint evaluations)",
        fontsize=11,
    )
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def _geometry_panels(metrics: dict[str, Any]) -> list[tuple[str, str, Any]]:
    panels = []
    for run_name, run in metrics["runs"].items():
        factor_names = list(
            run["selected_checkpoints"]["final"]["factors"]
        )
        for factor_name in factor_names:
            title = run["label"]
            if len(factor_names) > 1:
                title += f" / {factor_name.replace('_', ' ')}"
            panels.append((run_name, factor_name, title))
    return panels


def _geometry_figure(metrics: dict[str, Any], output: Path) -> None:
    panels = _geometry_panels(metrics)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7),
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for axis, (run_name, factor_name, title) in zip(flat_axes, panels):
        run = metrics["runs"][run_name]
        for checkpoint_key in CHECKPOINT_ORDER:
            factor = run["selected_checkpoints"][checkpoint_key]["factors"][
                factor_name
            ]
            layer_names = sorted(
                factor["layers"],
                key=lambda name: int(name.split("_")[-1]),
            )
            x = np.arange(1, len(layer_names) + 1)
            y = [
                factor["layers"][name]["metrics"]["r_squared"]
                for name in layer_names
            ]
            baseline = factor["baselines"][
                "nuisance/log_next_token_probability"
            ]["r_squared"]
            axis.plot(
                x,
                y,
                marker="o",
                color=COLORS[checkpoint_key],
                label=CHECKPOINT_LABELS[checkpoint_key],
            )
            axis.axhline(
                baseline,
                color=COLORS[checkpoint_key],
                linestyle="--",
                alpha=0.45,
                linewidth=1,
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xlabel("transformer block")
        axis.grid(alpha=0.2)
    flat_axes[0].set_ylabel("held-out R²")
    flat_axes[3].set_ylabel("held-out R²")
    for axis in flat_axes[len(panels):]:
        axis.axis("off")
    flat_axes[min(len(panels), len(flat_axes)) - 1].legend(
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Linear belief accessibility; dashed lines are log-NTP controls",
        fontsize=11,
    )
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def _format_number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _report_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Strata context-64 entropy-4x selected-checkpoint analysis",
        "",
        "## Design",
        "",
        "This post-hoc analysis restores only initialization, penultimate, and final "
        "RLModule checkpoints for each run. It does not retrain models or select "
        "layers/checkpoints using held-out probe performance.",
        "",
        "Initialization is a separately restored and sampled checkpoint baseline, "
        "not a matched-history `initialization_features` comparison: on-policy "
        "histories differ across checkpoints.",
        "",
        "Each checkpoint uses 20,000 process-weighted learned-policy samples for "
        "fit and 20,000 independent samples for test, eight environments, a 64-step "
        "per-episode warmup, and whole-episode grouped fitting/bootstrap. The "
        "representation is each transformer's current-position block residual "
        "before final layer normalization.",
        "",
        "## Task performance",
        "",
        "| run | initialization | penultimate | final |",
        "| --- | ---: | ---: | ---: |",
    ]
    for run in metrics["runs"].values():
        values = [
            run["selected_checkpoints"][key]["archived_task_mean_reward"]
            for key in CHECKPOINT_ORDER
        ]
        lines.append(
            f"| {run['label']} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |"
        )
    lines.extend(
        [
            "",
            "These are archived held-out policy means from the original checkpoint "
            "reports. Fresh deterministic re-collection is recorded separately in "
            "`metrics.json`.",
            "",
            "## Fixed final-layer geometry",
            "",
            "| run / factor | init R² | penultimate R² | final R² | final log-NTP R² | final probe − log-NTP R² (95% episode bootstrap) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in metrics["runs"].values():
        factor_names = list(run["selected_checkpoints"]["final"]["factors"])
        for factor_name in factor_names:
            values = {}
            for checkpoint_key in CHECKPOINT_ORDER:
                factor = run["selected_checkpoints"][checkpoint_key][
                    "factors"
                ][factor_name]
                final_layer = max(
                    factor["layers"],
                    key=lambda name: int(name.split("_")[-1]),
                )
                values[checkpoint_key] = factor["layers"][final_layer]
            factor = run["selected_checkpoints"]["final"]["factors"][
                factor_name
            ]
            baseline = factor["baselines"][
                "nuisance/log_next_token_probability"
            ]["r_squared"]
            comparison = values["final"]["comparisons"][
                "nuisance/log_next_token_probability"
            ]
            label = run["label"]
            if len(factor_names) > 1:
                label += f" / {factor_name.replace('_', ' ')}"
            interval = comparison["delta_r_squared_ci"]
            lines.append(
                f"| {label} | "
                f"{_format_number(values['init']['metrics']['r_squared'])} | "
                f"{_format_number(values['penultimate']['metrics']['r_squared'])} | "
                f"{_format_number(values['final']['metrics']['r_squared'])} | "
                f"{_format_number(baseline)} | "
                f"{_format_number(comparison['delta_r_squared'])} "
                f"[{_format_number(interval[0])}, {_format_number(interval[1])}] |"
            )
    token = metrics["runs"]["token_guess"]["selected_checkpoints"]["final"][
        "factors"
    ]["strata"]["layers"]["layer_3"]
    both = metrics["runs"]["reward_both"]["selected_checkpoints"]["final"][
        "factors"
    ]
    single = metrics["runs"]["reward_factor_1"]["selected_checkpoints"][
        "final"
    ]["factors"]
    token_comparison = token["comparisons"][
        "nuisance/log_next_token_probability"
    ]
    both_factor_1 = both["factor_1"]["layers"]["layer_3"]
    both_factor_2 = both["factor_2"]["layers"]["layer_3"]
    single_factor_1 = single["factor_1"]["layers"]["layer_3"]
    single_factor_2 = single["factor_2"]["layers"]["layer_3"]
    lines.extend(
        [
            "",
            "All layers, controls, covariance-matched Gaussian and "
            "observation-matched feature nulls, permuted-label nulls, "
            "emission-null-direction contrasts, and activation geometry are in "
            "`metrics.json` and `metrics.csv`.",
            "",
            "## Findings",
            "",
            "- All three archived policy scores rise substantially from initialization; "
            "penultimate and final scores are close, so the requested late-checkpoint "
            "comparison does not hinge on a large task-performance swing.",
            f"- Token-guess final-layer accessibility reaches "
            f"{token['metrics']['r_squared']:.3f} R² and exceeds the log-NTP "
            f"control by {token_comparison['delta_r_squared']:.3f} "
            f"[{token_comparison['delta_r_squared_ci'][0]:.3f}, "
            f"{token_comparison['delta_r_squared_ci'][1]:.3f}]. Its Strata "
            f"emission-null-direction R² is "
            f"{token['metrics']['contrasts']['prediction_null_1']['r_squared']:.3f}.",
            f"- Reward-both final-layer probes decode both factors "
            f"(factor 1 {both_factor_1['metrics']['r_squared']:.3f} R²; "
            f"factor 2 {both_factor_2['metrics']['r_squared']:.3f} R²). "
            f"Factor 1 robustly exceeds log-NTP; factor 2's overall paired "
            f"difference is "
            f"{both_factor_2['comparisons']['nuisance/log_next_token_probability']['delta_r_squared']:.3f} "
            "with an interval spanning zero.",
            f"- Reward-factor-1 is selective at the final layer: factor 1 reaches "
            f"{single_factor_1['metrics']['r_squared']:.3f} R², while factor 2 "
            f"is {single_factor_2['metrics']['r_squared']:.3f} R² despite a "
            f"{single['factor_2']['baselines']['nuisance/log_next_token_probability']['r_squared']:.3f} "
            "log-NTP baseline. This is an accessibility association with the "
            "rewarded factor, not evidence of causal use.",
            "",
            "## Timing and controls",
            "",
            "- Targets are read from `info[\"belief_current\"]` before the current action.",
            "- Token guess uses the delay-one arrival belief for the pending hidden token; "
            "the completed action is scored against `event.raw_token_before`.",
            "- Two-factor beliefs already include the preceding executed action through "
            "the prior edge-belief update, never reward information.",
            "- Reward both pays factor indices 0 and 1; reward factor 1 pays factor "
            "index 0 only. Report labels `factor_1` and `factor_2` are one-indexed.",
            "- The token-guess NTP control is recovered from the filtered source belief "
            "and predicts the pending hidden token; the two-factor NTP control projects "
            "the current factor belief through the Strata emission matrix.",
            "- First-episode lengths are randomized, and recurrent histories continue "
            "through resets before the 64-step warmup filter is applied.",
            "- Fit/test streams are independent: token spawn keys 800/801 and "
            "two-factor spawn keys 700/701 under run seed 42.",
            "",
            "## Limitations",
            "",
            "- Linear decodability is descriptive accessibility, not evidence of causal "
            "use and not a unique identification of the network's representation.",
            "- Only one training seed (42) is available; bootstrap intervals quantify "
            "held-out episode variation, not training-seed uncertainty.",
            "- On-policy state/history occupancy changes across checkpoints, so "
            "initialization-to-trained differences combine representation and occupancy changes.",
            "- The exact training harness revision differs from the current analysis "
            "checkout. Source hashes and both revisions are recorded in `provenance.json`; "
            "the checked diff adds analysis/checkpoint utilities while the verified rollout "
            "ordering and environment timing are unchanged.",
            "- No confidence intervals are inferred for archived task-performance means. "
            "Only explicitly computed whole-episode bootstrap intervals are shown.",
            "- Five stochastic null repetitions are a diagnostic battery, not a precise "
            "estimate of a null distribution.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render compact outputs for selected Strata checkpoints"
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists")
    output.mkdir(parents=True)
    metrics, provenance = _load_inputs(
        root,
        args.analysis_dir.resolve(),
    )
    report_source = Path(__file__).resolve()
    provenance["report_generator"] = {
        "path": str(report_source.relative_to(root)),
        "sha256": _sha256(report_source),
    }
    metrics_path = output / "metrics.json"
    provenance_path = output / "provenance.json"
    report_path = output / "report.md"
    with metrics_path.open("x") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with provenance_path.open("x") as handle:
        json.dump(
            provenance,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    rows = _metric_rows(metrics)
    _write_csv(output / "metrics.csv", rows)
    _task_figure(metrics, output / "task_performance")
    _geometry_figure(metrics, output / "belief_geometry")
    report_path.write_text(_report_markdown(metrics))
    owned = sorted(
        path
        for path in output.iterdir()
        if path.name != "report_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "generator": "experiments.strata_context_64_entropy_4x_posthoc.report",
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in owned
        },
    }
    with (output / "report_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
