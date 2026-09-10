from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from analysis.plots import simplex_scatter, to_xy
from analysis.probes.controls import fit_grouped_affine, score_prediction
from experiments.wing_token_guess_cycle_1.process import environment_config, WING_ALPHA, WING_X
from experiments.wing_two_factor_explore_cycle_1.control_analysis import _provenance
from experiments.wing_two_factor_explore_cycle_1.control_data import collect_control_data, replay_beliefs
from experiments.wing_two_factor_explore_cycle_1.evaluate_success import STUDIES, resolve_checkpoint


VERTICES = np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2], [1.0, 0.0]])
VERTEX_COLORS = np.array([[0.88, 0.28, 0.20], [0.18, 0.67, 0.35], [0.23, 0.39, 0.87]])


def _validate_clouds(target, decoded):
    target, decoded = np.asarray(target, dtype=np.float64), np.asarray(decoded, dtype=np.float64)
    if target.ndim != 3 or target.shape[1:] != (2, 3) or target.shape != decoded.shape or len(target) == 0:
        raise ValueError("target and decoded clouds must be aligned nonempty (samples, 2, 3) arrays")
    if not np.isfinite(target).all() or not np.isfinite(decoded).all():
        raise ValueError("simplex clouds must be finite")
    if (target < -1e-10).any() or not np.allclose(target.sum(axis=-1), 1, rtol=0, atol=1e-10):
        raise ValueError("Bayes targets must lie on the probability simplex")
    if not np.allclose(decoded.sum(axis=-1), 1, rtol=0, atol=1e-8):
        raise ValueError("raw affine outputs must sum to one for a faithful two-dimensional simplex view")
    return target, decoded


def geometry_metrics(target, decoded):
    target, decoded = _validate_clouds(target, decoded)
    contrast = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    return {
        f"factor_{factor + 1}": {
            **score_prediction(decoded[:, factor], target[:, factor]),
            "contrast_0_minus_2": score_prediction((decoded[:, factor] @ contrast)[:, None], (target[:, factor] @ contrast)[:, None]),
            "outside_simplex_fraction": float(np.mean(((decoded[:, factor] < -1e-9) | (decoded[:, factor] > 1 + 1e-9)).any(axis=1))),
            "max_probability_sum_error": float(np.max(np.abs(decoded[:, factor].sum(axis=1) - 1))),
        }
        for factor in range(2)
    }


def render_simplexes(target, decoded, *, layer, n_fit, agent_steps, seed=42):
    target, decoded = _validate_clouds(target, decoded)
    metrics = geometry_metrics(target, decoded)
    points = np.concatenate([target.reshape(-1, 3), decoded.reshape(-1, 3), np.eye(3)])
    xy = to_xy(points, vertices=VERTICES)
    limits = np.stack([xy.min(axis=0) - 0.13, xy.max(axis=0) + 0.13])
    order = np.random.default_rng(seed).permutation(len(target))
    settings = dict(matplotlib.rcParamsDefault)
    settings.update({"backend": "Agg", "font.family": "DejaVu Sans", "font.size": 10})
    with matplotlib.rc_context(settings):
        figure, axes = plt.subplots(2, 2, figsize=(12.8, 10.6), dpi=180)
        figure.subplots_adjust(left=0.035, right=0.965, top=0.86, bottom=0.17, wspace=0.12, hspace=0.35)
        figure.suptitle("Token guess: Bayes target vs probe-decoded belief geometry", y=0.975, fontsize=17)
        figure.text(0.5, 0.94, f"Final checkpoint: {agent_steps:,} training steps  |  Layer {layer}, pre-final-LayerNorm residual\n{n_fit:,} fit samples and {len(target):,} independent held-out samples; whole-episode probe cross-validation", ha="center", va="top", fontsize=10)
        try:
            for factor in range(2):
                colors = target[order, factor] @ VERTEX_COLORS
                for column, cloud in enumerate((target[:, factor], decoded[:, factor])):
                    axis = axes[factor, column]
                    simplex_scatter(axis, cloud[order], colors=colors, s=2.5, alpha=0.40,
                                    labels=("State 0", "State 1", "State 2"), vertices=VERTICES)
                    axis.set_xlim(limits[:, 0])
                    axis.set_ylim(limits[:, 1])
                    axis.set_title(f"Factor {factor + 1}: " + ("exact Bayes target" if column == 0 else "affine decode of agent activations"), fontsize=12, pad=13)
                score = metrics[f"factor_{factor + 1}"]
                axes[factor, 1].text(
                    0.5, -0.05,
                    f"R² = {score['r_squared']:.3f}   |   MSE = {score['mse']:.5f}   |   0–2 contrast R² = {score['contrast_0_minus_2']['r_squared']:.3f}\n"
                    f"Raw predictions outside simplex: {100 * score['outside_simplex_fraction']:.1f}%",
                    transform=axes[factor, 1].transAxes, ha="center", va="top", fontsize=10,
                )
            figure.legend(handles=[Patch(color=color, label=f"State {state}") for state, color in enumerate(VERTEX_COLORS)],
                          loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=3, frameon=False)
            figure.text(0.5, 0.073, "Colors encode each point's true Bayes belief and are identical in each left/right pair.", ha="center", fontsize=10)
            figure.text(0.5, 0.025,
                        "Same held-out histories, same barycentric coordinates and axis limits; all points shown.\n"
                        "No clipping, renormalization, or simplex projection of predictions. This is a supervised linear decode, not PCA.\n"
                        "Two 3-state factor marginals are shown, not the full 9-state joint simplex. Targets are decision-time arrival beliefs.",
                        ha="center", fontsize=9.5)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=180, metadata={"Software": "experiments.wing_token_guess_cycle_1.simplex"})
            return buffer.getvalue()
        finally:
            plt.close(figure)


def collect_and_fit(module, *, seed=42, n_steps=20_000):
    config = environment_config()
    streams = np.random.SeedSequence(seed).spawn(4)
    train, test = (
        collect_control_data(module, env_config=config, n_steps=n_steps,
                             seed=int(stream.generate_state(1)[0]), device=torch.device("cpu"))
        for stream in streams[:2]
    )
    parameters = {"alpha": WING_ALPHA, "x": WING_X, "strength": None}
    train_beliefs, _ = replay_beliefs(train, **parameters)
    test_beliefs, _ = replay_beliefs(test, **parameters)
    alignment = max(float(np.max(np.abs(train_beliefs - train.beliefs))), float(np.max(np.abs(test_beliefs - test.beliefs))))
    if alignment > 1e-9:
        raise ValueError("exact replay and environment belief targets disagree")
    target = test_beliefs[test.mask]
    decoded = np.empty_like(target)
    fits = {}
    for factor in range(2):
        weight, bias, fit = fit_grouped_affine(train.activations[train.mask, -1], train_beliefs[train.mask, factor],
                                             train.episode_ids[train.mask], seed=seed + factor)
        decoded[:, factor] = test.activations[test.mask, -1] @ weight + bias
        fits[f"factor_{factor + 1}"] = fit
    return target, decoded, {
        "seed": seed, "n_fit": int(train.mask.sum()), "n_test": int(test.mask.sum()),
        "train_episodes": int(len(np.unique(train.episode_ids[train.mask]))),
        "test_episodes": int(len(np.unique(test.episode_ids[test.mask]))),
        "layer": train.activations.shape[1], "feature_width": train.activations.shape[2],
        "layer_selection": "last layer, not best test layer", "fits": fits,
        "replay_max_abs_error": alignment, "environment_config": config,
        "warmup_per_episode": 32, "sampling_distribution": "learned_stochastic_policy_process_weighted",
        "belief_timing": "delay-one current arrival belief = filtered source posterior @ transition; observed tokens only",
        "display": "raw affine outputs, common barycentric axes, no clipping or simplex projection; point colors from true targets",
        "vertices_by_state": VERTICES.tolist(), "vertex_colors_by_state": VERTEX_COLORS.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot held-out token-guess Bayes targets beside affine probe decodes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a .png path")
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        parser.error("output or sidecar already exists; choose fresh paths")
    from ray.rllib.core.rl_module.rl_module import RLModule

    root = Path(__file__).resolve().parents[2]
    checkpoint, source, run = resolve_checkpoint(root, *STUDIES["token_guess"])
    provenance = _provenance(checkpoint)
    torch.set_num_threads(1)
    module = RLModule.from_checkpoint(str(checkpoint))
    target, decoded, metadata = collect_and_fit(module, seed=args.seed, n_steps=args.steps)
    metadata.update({"agent_steps": int(run["trials"][0]["metrics"]["env_runners/num_env_steps_sampled_lifetime"]),
                     "checkpoint": str(checkpoint.relative_to(root)), "source_run": str(source.relative_to(root)),
                     "provenance": provenance, "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    metrics = geometry_metrics(target, decoded)
    png = render_simplexes(target, decoded, layer=metadata["layer"], n_fit=metadata["n_fit"], agent_steps=metadata["agent_steps"], seed=args.seed)
    payload = {"schema_version": 1, "metadata": metadata, "factors": metrics, "png_sha256": hashlib.sha256(png).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(png)
    with summary_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output} and {summary_path}")
    for name, score in metrics.items():
        print(f"{name}: R2={score['r_squared']:.12f}, outside={100 * score['outside_simplex_fraction']:.3f}%")


if __name__ == "__main__":
    main()
