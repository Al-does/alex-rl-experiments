from __future__ import annotations

import argparse
import csv
import hashlib
import io
from itertools import permutations
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from analysis.plots import plot_belief_comparison
from analysis.probes import split_group_indices, variance_geometry
from analysis.probes.controls import (
    fit_grouped_affine, gaussian_feature_null, matched_feature_null,
    paired_comparison, paired_target_comparison, score_prediction,
)
from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_1 import analysis
from experiments.strata_token_guess_cycle_1.geometry_data import collect_geometry_data, replay_beliefs
from experiments.strata_token_guess_cycle_1.process import environment_config
from harness.seeding import named_seed_sequences


PARAMETERS = {"alpha": 0.97, "t0": 0.38, "t1": 0.54}
SUFFIXES = (0, 1, 2, 4, 8, 16, 32)
VERTICES = np.array([[0, 0], [0.5, np.sqrt(3) / 2], [1, 0]], dtype=float)
COLORS = np.array([[0.88, 0.28, 0.20], [0.18, 0.67, 0.35], [0.23, 0.39, 0.87]])
RUN_RESULTS = Path(__file__).resolve().parent / "ppo/results/20260909T061802Z-7d4e2248"


def ntp_null_direction(parameters=PARAMETERS):
    model = strata_model(**parameters)
    pending_map = np.linalg.solve(model.transition_matrix, model.emission_matrix)
    _, singular_values, vt = np.linalg.svd(np.vstack([np.ones(3), pending_map.T]))
    direction = vt[-1]
    direction *= 1 if direction[0] >= 0 else -1
    if singular_values[-1] > 1e-10 or np.max(np.abs(direction @ pending_map)) > 1e-10:
        raise ValueError("expected one belief contrast invisible to pending-token probabilities")
    return direction, pending_map


def _log(probabilities):
    return np.log(np.maximum(probabilities, 1e-12))


def _fit(train_x, train_y, test_x, groups, seed):
    weight, bias, fit = fit_grouped_affine(train_x, train_y, groups, seed=seed)
    return test_x @ weight + bias, fit


def _score(predicted, target, direction):
    return {
        **score_prediction(predicted, target),
        "ntp_null_contrast": score_prediction((predicted @ direction)[:, None], (target @ direction)[:, None]),
        "outside_simplex_fraction": float(np.mean(((predicted < -1e-9) | (predicted > 1 + 1e-9)).any(axis=1))),
        "max_sum_error": float(np.max(np.abs(predicted.sum(axis=1) - 1))),
    }


@torch.inference_mode()
def policy_features(module, data):
    features = torch.as_tensor(data.activations[data.mask, -1], dtype=torch.float32)
    logits = module.action_distribution_inputs(module.encoder.final_norm(features))
    return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def alternative_parameters():
    candidates = []
    for changes in (
        {"alpha": 0.93}, {"alpha": 0.95}, {"alpha": 0.985},
        {"t0": 0.30}, {"t0": 0.34}, {"t0": 0.42}, {"t0": 0.46},
        {"t1": 0.46}, {"t1": 0.50}, {"t1": 0.58}, {"t1": 0.62},
        {"alpha": 0.95, "t0": 0.34, "t1": 0.58},
    ):
        candidates.append(PARAMETERS | changes)
    return candidates


def select_alternatives(train, *, seed=42):
    true_b, true_p = replay_beliefs(train, **PARAMETERS)
    target, p = true_b[train.mask], true_p[train.mask]
    groups = train.episode_ids[train.mask]
    fit_rows, validation = split_group_indices(groups, test_fraction=0.25, seed=seed)
    candidates = []
    for index, parameters in enumerate(alternative_parameters()):
        candidate_b, candidate_p = replay_beliefs(train, **parameters)
        b, q = candidate_b[train.mask], candidate_p[train.mask]
        design = np.column_stack([np.ones(len(fit_rows)), target[fit_rows]])
        coefficients = np.linalg.lstsq(design, b[fit_rows], rcond=1e-10)[0]
        predicted = np.column_stack([np.ones(len(validation)), target[validation]]) @ coefficients
        residual = score_prediction(predicted, b[validation])["normalized_mse"]
        permutation_mse = min(float(np.mean((target[validation][:, order] - b[validation]) ** 2)) for order in permutations(range(3)))
        kl = float(np.mean(np.sum(p[validation] * (_log(p[validation]) - _log(q[validation])), axis=1)))
        candidates.append({
            "id": f"alternative_{index + 1}", "parameters": parameters,
            "validation_kl_nats": kl, "validation_affine_residual_ratio": residual,
            "validation_min_permutation_mse": permutation_mse,
            "eligible": residual is not None and residual >= 0.01 and permutation_mse > 1e-8 and kl <= 0.02,
        })
    chosen = sorted((item for item in candidates if item["eligible"]), key=lambda item: (-item["validation_affine_residual_ratio"], item["id"]))[:3]
    return chosen, {
        "selection_scope": "training episodes only; no activations or held-out test histories",
        "criterion": "largest affine residual ratio under predictive KL <= 0.02 nats, residual >= 0.01, and non-permutation-equivalence",
        "candidates": candidates, "selected_ids": [item["id"] for item in chosen],
        "status": "selected" if chosen else "no_eligible_alternative",
        "interpretation": "finite nearby-model search; not a worst-case guarantee or evidence the network uses these models",
    }


def analyze_samples(train, test, *, policy_train, policy_test, initial=None, seed=42, repeats=5, resamples=200):
    train_all_b, train_all_p = replay_beliefs(train, **PARAMETERS)
    test_all_b, test_all_p = replay_beliefs(test, **PARAMETERS)
    alignment = max(float(np.max(np.abs(train_all_b - train.beliefs))), float(np.max(np.abs(test_all_b - test.beliefs))))
    if alignment > 1e-9:
        raise ValueError("replayed source/arrival timing does not match environment targets")
    b_train, b_test = train_all_b[train.mask], test_all_b[test.mask]
    p_train, p_test = train_all_p[train.mask], test_all_p[test.mask]
    x_train, x_test = train.activations[train.mask], test.activations[test.mask]
    groups_train, groups_test = train.episode_ids[train.mask], test.episode_ids[test.mask]
    direction, pending_map = ntp_null_direction()
    if initial is not None:
        for actual, reference in zip(initial, (train, test), strict=True):
            for key in ("observations", "episode_ids", "episode_steps", "mask", "beliefs"):
                if not np.array_equal(getattr(actual, key), getattr(reference, key)):
                    raise ValueError("initialization control must use identical passive histories and beliefs")
    features = {
        "ntp": (p_train, p_test), "log_ntp": (_log(p_train), _log(p_test)),
        "current_token": (train.observations[train.mask], test.observations[test.mask]),
        "policy_probabilities": (policy_train, policy_test),
        "centered_log_policy": (_log(policy_train) - _log(policy_train).mean(axis=1, keepdims=True), _log(policy_test) - _log(policy_test).mean(axis=1, keepdims=True)),
    }
    for length in SUFFIXES:
        features[f"suffix_{length}"] = tuple(replay_beliefs(data, **PARAMETERS, suffix_length=length)[0][data.mask] for data in (train, test))
    baselines = {}
    predictions = {"train_mean": np.broadcast_to(b_train.mean(axis=0), b_test.shape)}
    baselines["train_mean"] = _score(predictions["train_mean"], b_test, direction)
    for name, (train_features, test_features) in features.items():
        predicted, fit = _fit(train_features, b_train, test_features, groups_train, seed)
        predictions[name] = predicted
        baselines[name] = {**_score(predicted, b_test, direction), "fit": fit}
    chosen, search = select_alternatives(train, seed=seed)
    alternatives = {}
    for candidate in chosen:
        alt_train, alt_p_train = replay_beliefs(train, **candidate["parameters"])
        alt_test, alt_p_test = replay_beliefs(test, **candidate["parameters"])
        alternatives[candidate["id"]] = (alt_train[train.mask], alt_test[test.mask])
        candidate["test_kl_nats"] = float(np.mean(np.sum(p_test * (_log(p_test) - _log(alt_p_test[test.mask])), axis=1)))
        candidate["test_posterior_mse"] = float(np.mean((b_test - alt_test[test.mask]) ** 2))
    report = {
        "schema_version": 1,
        "metadata": {
            "seed": seed, "parameters": PARAMETERS, "n_fit": len(b_train), "n_test": len(b_test),
            "train_episodes": len(np.unique(groups_train)), "test_episodes": len(np.unique(groups_test)),
            "warmup_per_episode": 32, "sampling_distribution": "learned stochastic policy, process-weighted",
            "representation": "each encoder block residual at current position, before final LayerNorm",
            "probe_cv": "training-only whole-episode SVD-cutoff CV", "bootstrap": "paired test episodes; fixed fitted probes, not training-seed uncertainty",
            "null_repeats": repeats, "bootstrap_resamples": resamples,
            "belief_timing": "arrival posterior at current decision; source posterior filters observed delayed tokens; no action/reward conditioning",
            "pending_ntp_map_from_arrival_belief": pending_map.tolist(), "ntp_null_direction": direction.tolist(),
            "ntp_null_map_residual": (direction @ pending_map).tolist(), "log_floor": 1e-12,
            "comparison_note": "Strata NTP-null contrast differs from the Wing state-0 minus state-2 direction",
        },
        "replay_max_abs_error": alignment, "baselines": baselines, "layers": {}, "alternative_search": search,
        "target_geometry": variance_geometry(b_test),
        "scope_warning": "Decodability does not establish causal use or unique belief coding; one trained seed; layer/control comparisons are descriptive.",
        "policy": {
            "conditional_expected_accuracy": float(np.mean(np.sum(policy_test * p_test, axis=1))),
            "same_histories_bayes_accuracy": float(np.mean(p_test.max(axis=1))),
            "same_histories_best_fixed_guess_accuracy": float(p_test.mean(axis=0).max()),
            "bayes_prefers_token_0_fraction": float(np.mean(p_test[:, 0] > p_test[:, 1])),
            "policy_mean_token_0_probability": float(policy_test[:, 0].mean()),
        },
    }
    decoded = {}
    for layer in range(x_train.shape[1]):
        name = f"layer_{layer + 1}"
        tx, vx = x_train[:, layer], x_test[:, layer]
        predicted, fit = _fit(tx, b_train, vx, groups_train, seed)
        decoded[name] = predicted
        result = {
            "belief_probe": {**_score(predicted, b_test, direction), "fit": fit},
            "surplus": {key: paired_comparison(predicted, baseline, b_test, groups_test, seed=seed, n_resamples=resamples) for key, baseline in predictions.items()},
            "output_probes": {}, "nulls": {}, "alternative_targets": {}, "activation_geometry": variance_geometry(vx),
        }
        if initial is not None:
            initial_prediction, initial_fit = _fit(initial[0].activations[initial[0].mask, layer], b_train, initial[1].activations[initial[1].mask, layer], groups_train, seed)
            result["initialization_probe"] = {**_score(initial_prediction, b_test, direction), "fit": initial_fit}
            result["over_initialization"] = paired_comparison(predicted, initial_prediction, b_test, groups_test, seed=seed, n_resamples=resamples)
        for control in ("ntp", "log_ntp"):
            output, output_fit = _fit(tx, features[control][0], vx, groups_train, seed)
            result["output_probes"][control] = {**score_prediction(output, features[control][1]), "fit": output_fit}
        for repeat in range(repeats):
            null_seed = seed + 1000 + repeat
            rng = np.random.default_rng(null_seed)
            shuffled, shuffled_fit = _fit(tx, b_train[rng.permutation(len(b_train))], vx, groups_train, seed)
            nulls = {"shuffled_labels": {**_score(shuffled, b_test, direction), "fit": shuffled_fit}}
            for key, (null_train, null_test, sampling) in (
                ("gaussian_activations", gaussian_feature_null(tx, len(vx), seed=null_seed)),
                ("token_matched_activations", matched_feature_null(tx, vx, train.observations[train.mask], test.observations[test.mask], seed=null_seed)),
                ("ntp_matched_activations", matched_feature_null(tx, vx, np.floor(p_train[:, :1] / 0.025), np.floor(p_test[:, :1] / 0.025), seed=null_seed)),
            ):
                null_prediction, null_fit = _fit(null_train, b_train, null_test, groups_train, seed)
                nulls[key] = {**_score(null_prediction, b_test, direction), "fit": null_fit, "sampling": sampling}
            result["nulls"][str(repeat)] = nulls
        for candidate in chosen:
            alt_train, alt_test = alternatives[candidate["id"]]
            alt_prediction, alt_fit = _fit(tx, alt_train, vx, groups_train, seed)
            result["alternative_targets"][candidate["id"]] = {
                **score_prediction(alt_prediction, alt_test), "fit": alt_fit,
                "true_minus_alternative": paired_target_comparison(predicted, b_test, alt_prediction, alt_test, groups_test, seed=seed, n_resamples=resamples),
            }
        report["layers"][name] = result
    return report, b_test, decoded


def render_simplex(target, decoded, report):
    target, decoded = np.asarray(target), np.asarray(decoded)
    layer_name = f"layer_{len(report['layers'])}"
    score = report["layers"][layer_name]["belief_probe"]
    if not np.allclose(decoded.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("raw decoder must remain in the affine sum-one plane")
    contrast = score["ntp_null_contrast"]["r_squared"]
    contrast_label = "N/A" if contrast is None else f"{contrast:.3f}"
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure = plot_belief_comparison(
            target, decoded, coordinates=VERTICES, point_colors=target @ COLORS,
            state_labels=("State 0", "State 1", "State 2"), seed=report["metadata"]["seed"],
            title="Strata: Bayes target vs probe-decoded belief geometry",
        )
        try:
            figure.set_size_inches(12.8, 8.0)
            figure.text(0.5, 0.925,
                        f"Final checkpoint: {report['metadata']['agent_steps']:,} steps  |  Last layer ({layer_name}), pre-final-LayerNorm residual  |  α=0.97, t₀=0.38, t₁=0.54\n"
                        f"{report['metadata']['n_fit']:,} fit and {len(target):,} independent held-out samples; whole-episode cross-validation",
                        ha="center", va="top", fontsize=9)
            figure.text(0.5, 0.21, f"NTP-null contrast R² = {contrast_label}", ha="center", va="center", fontsize=10)
            figure.text(0.5, 0.005,
                        "Targets are delay-one decision-time arrival beliefs. Strata's NTP-null direction is computed from its own emission map.",
                        ha="center", va="bottom", fontsize=8.5)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=180, metadata={"Software": __name__})
            return buffer.getvalue()
        finally:
            plt.close(figure)


def training_rows(summary):
    rows = []
    for item in summary["checkpoint_reports"]:
        if item["n_test"] != 20_000 or item["metadata"]["warmup_per_episode"] != 32:
            raise ValueError("archived success history has a different sampling protocol")
        step = int(item["agent_steps"])
        if item["is_initialization"] != (step == 0):
            raise ValueError("initialization flag and environment step disagree")
        rows.append({"agent_steps": step, "training_iteration": item["training_iteration"], "token_accuracy": item["policy"]["token_accuracy"], "n_test": item["n_test"], "warmup_per_episode": 32})
    rows.sort(key=lambda row: row["agent_steps"])
    if not rows or rows[0]["agent_steps"] != 0 or len({row["agent_steps"] for row in rows}) != len(rows):
        raise ValueError("a unique actual initialization and unique checkpoints are required")
    return rows


def render_training(rows, *, bayes_lower, bayes_upper, constant_accuracy):
    means = np.array([row["token_accuracy"] for row in rows]) * 100
    steps = [row["agent_steps"] for row in rows]
    overshoot = means.max() > bayes_upper * 100
    ceiling = float(means.max() + max(1.0, means.max() * 0.02)) if overshoot else 100 * bayes_upper
    axis_note = "Axes expanded: empirical estimates can exceed the population bound" if overshoot else "The upper bound sets the graph ceiling"
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axis = plt.subplots(figsize=(10, 6), dpi=180)
        figure.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.24)
        figure.suptitle("Strata token accuracy over training", y=0.98, fontsize=17)
        figure.text(0.5, 0.91, f"Bayes maximum bracket: {100 * bayes_lower:.5f}%–{100 * bayes_upper:.5f}%\n{axis_note}; best constant guess = {100 * constant_accuracy:.5f}%", ha="center", va="top", fontsize=11)
        try:
            axis.plot(steps, means, "o-", color="#4477AA", linewidth=1.7, markersize=5, label="Recorded checkpoint test accuracy")
            axis.scatter([0], [means[0]], marker="*", s=160, color="#EEAA33", edgecolor="black", zorder=5)
            axis.annotate("Init", (0, means[0]), xytext=(8, -26), textcoords="offset points", fontweight="bold")
            axis.axhspan(100 * bayes_lower, 100 * bayes_upper, color="#555555", alpha=0.18)
            axis.axhline(100 * bayes_upper, color="black", linestyle="--", clip_on=False, label="Bayes maximum (numerical upper bound)")
            axis.axhline(100 * constant_accuracy, color="#777777", linestyle=":", label="Best fixed token")
            axis.set_ylim(0, ceiling)
            axis.set_xlim(-0.02 * max(steps), 1.03 * max(steps))
            axis.set_xlabel("Training environment steps (millions; linear)")
            axis.set_ylabel("Token accuracy (%)")
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value / 1e6:g}"))
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            axis.grid(alpha=0.25); axis.legend(loc="lower right", frameon=False)
            axis.annotate(f"{means[-1]:.3f}%", (steps[-1], means[-1]), xytext=(-5, -20), textcoords="offset points", ha="right", fontweight="bold")
            figure.text(0.5, 0.075, "Actual initialization and sampled checkpoints; lines connect recorded points only.\n20,000 held-out steps per checkpoint; first 32 steps/episode excluded; no saved confidence intervals.\nThe Bayes band bounds expected accuracy, not individual finite-sample measurements.", ha="center", fontsize=9.5)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=180, metadata={"Software": __name__})
            return buffer.getvalue()
        finally:
            plt.close(figure)


def compact_summary(report):
    layers = {}
    for name, layer in report["layers"].items():
        layers[name] = {
            "r_squared": layer["belief_probe"]["r_squared"],
            "mse": layer["belief_probe"]["mse"],
            "outside_simplex_fraction": layer["belief_probe"]["outside_simplex_fraction"],
            "max_sum_error": layer["belief_probe"]["max_sum_error"],
            "initialization_outside_simplex_fraction": layer.get("initialization_probe", {}).get("outside_simplex_fraction"),
            "ntp_null_r_squared": layer["belief_probe"]["ntp_null_contrast"]["r_squared"],
            "initialization_r_squared": layer.get("initialization_probe", {}).get("r_squared"),
            "initialization_ntp_null_r_squared": layer.get("initialization_probe", {}).get("ntp_null_contrast", {}).get("r_squared"),
            "delta_r2_vs_ntp": layer["surplus"]["ntp"]["delta_r_squared"],
            "delta_r2_vs_ntp_ci95": layer["surplus"]["ntp"]["delta_r_squared_ci"],
            "delta_r2_vs_log_ntp": layer["surplus"]["log_ntp"]["delta_r_squared"],
            "delta_r2_vs_log_ntp_ci95": layer["surplus"]["log_ntp"]["delta_r_squared_ci"],
            "null_r_squared_means": {key: float(np.mean([value[key]["r_squared"] for value in layer["nulls"].values()])) for key in next(iter(layer["nulls"].values()))},
            "alternatives": {
                key: {"r_squared": value["r_squared"], "true_minus_alternative_r_squared": value["true_minus_alternative"]["r_squared_difference"], "ci95": value["true_minus_alternative"]["r_squared_difference_ci"]}
                for key, value in layer["alternative_targets"].items()
            },
        }
    return {
        "metadata": report["metadata"], "layers": layers,
        "baselines": {key: {metric: value[metric] for metric in ("mse", "r_squared", "ntp_null_contrast", "outside_simplex_fraction", "max_sum_error")} for key, value in report["baselines"].items()},
        "policy": report["policy"], "selected_alternatives": report["alternative_search"]["selected_ids"],
        "scope_warning": report["scope_warning"],
    }


def _write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Strata belief geometry and Bayes-referenced task success")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--run-results", type=Path, default=RUN_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args(argv)
    filenames = ("geometry.json", "summary.json", "bayes_reference.json", "belief_simplex.png", "task_success.png", "training_success.csv", "probe_summary.csv", "baselines.csv", "output_manifest.json")
    if any((args.output_dir / name).exists() for name in filenames):
        parser.error("outputs already exist; choose a new output directory")
    from ray.rllib.core.rl_module.rl_module import RLModule
    from experiments.strata_token_guess_cycle_1.benchmark import bayes_accuracy_bounds, finite_episode_accuracy_bounds, monte_carlo_bayes_accuracy
    from experiments.strata_token_guess_cycle_1 import geometry_data, benchmark
    from analysis.probes import controls
    import envs.strata.model as domain
    import subprocess

    recipe = json.loads((args.run_results / "resolved_recipe.json").read_text())
    source_manifest = json.loads((args.run_results / "run_manifest.json").read_text())
    if source_manifest["status"] != "completed" or recipe["environment"] != environment_config():
        parser.error("completed source run must match the Strata recipe")
    history = json.loads((args.run_results / "condition_summary.json").read_text())
    rows = training_rows(history)
    torch.set_num_threads(1)
    streams = named_seed_sequences(args.seed, analysis._STREAM_KEYS)
    module = RLModule.from_checkpoint(str(args.checkpoint.resolve()))
    train, test = (collect_geometry_data(module, n_steps=args.steps, seed=streams[name], device=torch.device("cpu")) for name in ("probe_train", "probe_test"))
    train_policy, test_policy = (policy_features(module, data) for data in (train, test))
    initial = None
    if args.initial_checkpoint is not None:
        initial_module = RLModule.from_checkpoint(str(args.initial_checkpoint.resolve()))
        initial = tuple(collect_geometry_data(initial_module, n_steps=args.steps, seed=streams[name], device=torch.device("cpu")) for name in ("probe_train", "probe_test"))
    report, target, predictions = analyze_samples(train, test, policy_train=train_policy, policy_test=test_policy, initial=initial, seed=args.seed)
    module_paths = {"final": args.checkpoint}
    if args.initial_checkpoint is not None:
        module_paths["initialization"] = args.initial_checkpoint
    sources = (Path(__file__), Path(analysis.__file__), Path(geometry_data.__file__), Path(benchmark.__file__), Path(controls.__file__), Path(domain.__file__))
    roots = (Path(__file__).resolve().parents[2], Path(domain.__file__).resolve().parents[2])
    report["metadata"].update({
        "agent_steps": rows[-1]["agent_steps"], "source_run": str(args.run_results.resolve()),
        "checkpoint_paths": {key: str(path.resolve()) for key, path in module_paths.items()},
        "checkpoint_sha256": {key: {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.iterdir()) if path.is_file()} for key, root in module_paths.items()},
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "analysis_revisions": {str(root): {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())} for root in roots},
        "training_provenance": source_manifest["git"], "torch_version": torch.__version__, "numpy_version": np.__version__,
    })
    reference = bayes_accuracy_bounds(**PARAMETERS, max_depth=24)
    if not reference["tolerance_met"]:
        raise ValueError("Bayes maximum was not localized to the requested numerical tolerance")
    if reference["finite_prefix_minimum_length"] > 32:
        raise ValueError("Bayes bounds must apply before the recorded warmup ends")
    reference["post_warmup_episode_bounds"] = finite_episode_accuracy_bounds(reference)
    reference["full_episode_bounds"] = finite_episode_accuracy_bounds(reference, warmup=0)
    reference["monte_carlo_check"] = monte_carlo_bayes_accuracy(**PARAMETERS)
    last = f"layer_{len(report['layers'])}"
    images = {
        "belief_simplex.png": render_simplex(target, predictions[last], report),
        "task_success.png": render_training(rows, bayes_lower=reference["finite_prefix_lower"], bayes_upper=reference["finite_prefix_upper"], constant_accuracy=reference["constant_guess_accuracy"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = compact_summary(report)
    for name, payload in (("geometry.json", report), ("summary.json", summary), ("bayes_reference.json", reference)):
        with (args.output_dir / name).open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    for name, data in images.items():
        with (args.output_dir / name).open("xb") as handle:
            handle.write(data)
    _write_csv(args.output_dir / "training_success.csv", rows)
    _write_csv(args.output_dir / "probe_summary.csv", [{"layer": name, **{key: value for key, value in layer.items() if isinstance(value, (float, int)) or value is None}} for name, layer in summary["layers"].items()])
    _write_csv(args.output_dir / "baselines.csv", [{"baseline": name, "r_squared": value["r_squared"], "mse": value["mse"], "ntp_null_r_squared": value["ntp_null_contrast"]["r_squared"], "outside_simplex_fraction": value["outside_simplex_fraction"], "max_sum_error": value["max_sum_error"]} for name, value in report["baselines"].items()])
    hashes = {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest() for name in filenames if name != "output_manifest.json"}
    with (args.output_dir / "output_manifest.json").open("x") as handle:
        json.dump({"source_commit": "ca3fc06", "outputs_sha256": hashes}, handle, indent=2, sort_keys=True); handle.write("\n")
    print(f"Wrote results and both PNGs to {args.output_dir}")
    print(json.dumps({"layers": summary["layers"], "bayes_interval": [reference["lower"], reference["upper"]], "post_warmup_interval": [reference["finite_prefix_lower"], reference["finite_prefix_upper"]], "policy": report["policy"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
