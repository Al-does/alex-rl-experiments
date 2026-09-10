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

from analysis.plots import plot_belief_comparison
from analysis.probes.controls import fit_grouped_affine, score_prediction
from experiments.wing_token_guess_cycle_2 import analysis
from experiments.wing_token_guess_cycle_2.process import environment_config
from harness.seeding import named_seed_sequences


VERTICES = np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2], [1.0, 0.0]])
COLORS = np.array([[0.88, 0.28, 0.20], [0.18, 0.67, 0.35], [0.23, 0.39, 0.87]])
DEFAULT_RESULTS = Path(__file__).resolve().parent / "ppo/results/20260909T033311Z-e7115516"


def geometry_metrics(target, decoded):
    target, decoded = np.asarray(target, dtype=np.float64), np.asarray(decoded, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or target.shape != decoded.shape or len(target) == 0:
        raise ValueError("target and decoded must be aligned nonempty (samples, 3) arrays")
    if not np.isfinite(target).all() or not np.isfinite(decoded).all():
        raise ValueError("simplex clouds must be finite")
    if (target < -1e-10).any() or not np.allclose(target.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("Bayes targets must lie on the simplex")
    if not np.allclose(decoded.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("raw affine outputs must sum to one for a faithful simplex view")
    contrast = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    return {
        **score_prediction(decoded, target),
        "contrast_0_minus_2": score_prediction((decoded @ contrast)[:, None], (target @ contrast)[:, None]),
        "outside_simplex_fraction": float(np.mean(((decoded < -1e-9) | (decoded > 1 + 1e-9)).any(axis=1))),
        "max_probability_sum_error": float(np.max(np.abs(decoded.sum(axis=1) - 1))),
    }


def render_simplex(target, decoded, *, layer, n_fit, agent_steps, seed=42):
    target, decoded = np.asarray(target), np.asarray(decoded)
    metrics = geometry_metrics(target, decoded)
    contrast = metrics["contrast_0_minus_2"]["r_squared"]
    contrast_label = "N/A" if contrast is None else f"{contrast:.3f}"
    settings = dict(matplotlib.rcParamsDefault)
    settings.update({"backend": "Agg", "font.family": "DejaVu Sans", "font.size": 10})
    with matplotlib.rc_context(settings):
        figure = plot_belief_comparison(
            target, decoded, coordinates=VERTICES, point_colors=target @ COLORS,
            state_labels=("State 0", "State 1", "State 2"), seed=seed,
            title="Token guess cycle 2: target vs probe-decoded simplex",
        )
        try:
            figure.set_size_inches(12.8, 8.0)
            figure.text(0.5, 0.925,
                        f"Single 3-state Wing HMM  |  Final checkpoint: {agent_steps:,} steps  |  Layer {layer}, pre-final-LayerNorm residual\n"
                        f"{n_fit:,} fit samples and {len(target):,} independent held-out samples; whole-episode probe cross-validation",
                        ha="center", va="top", fontsize=9)
            figure.text(0.5, 0.21, f"0–2 contrast R² = {contrast_label}", ha="center", va="center", fontsize=10)
            figure.text(0.5, 0.005,
                        "Targets are decision-time arrival beliefs conditioned on delayed observed tokens only. First 32 steps/episode excluded.",
                        ha="center", va="bottom", fontsize=8.5)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=180, metadata={"Software": "experiments.wing_token_guess_cycle_2.simplex"})
            return buffer.getvalue()
        finally:
            plt.close(figure)


def collect_and_fit(module, *, seed=42, n_steps=20_000):
    streams = named_seed_sequences(seed, analysis._STREAM_KEYS)
    train, test = (
        analysis.collect_probe_data(module, n_steps=n_steps, seed=streams[name], device=torch.device("cpu"))
        for name in ("probe_train", "probe_test")
    )
    if train.episode_ids is None or test.episode_ids is None:
        raise ValueError("whole-episode fitting requires episode IDs")
    if train.factor_beliefs.shape[1:] != (1, 3) or test.factor_beliefs.shape[1:] != (1, 3):
        raise ValueError("cycle 2 requires a single three-state belief, not two factors")
    target = test.joint_beliefs
    weight, bias, fit = fit_grouped_affine(train.activations[:, -1], train.joint_beliefs, train.episode_ids, seed=seed)
    decoded = test.activations[:, -1] @ weight + bias
    return target, decoded, {
        "seed": seed, "n_fit": len(train.activations), "n_test": len(test.activations), "fit": fit,
        "train_episodes": int(len(np.unique(train.episode_ids))), "test_episodes": int(len(np.unique(test.episode_ids))),
        "layer": train.activations.shape[1], "feature_width": train.activations.shape[2], "layer_selection": "last layer, not best test layer",
        "rollout_seed_spawn_keys": {name: list(streams[name].spawn_key) for name in ("probe_train", "probe_test")},
        "belief_timing": "info.belief_current before the current guess; delayed token history only, never actions or rewards",
        "sampling_distribution": "learned_stochastic_policy_process_weighted", "warmup_per_episode": analysis.WARMUP,
        "environment_config": environment_config(), "factor_count": 1,
        "display": "raw outputs; identical axes and target-derived colors; no clipping or simplex projection",
        "test_token_accuracy": float(np.mean(test.actions == test.hidden_tokens)),
        "vertices_by_state": VERTICES.tolist(), "vertex_colors_by_state": COLORS.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-HMM Wing cycle-2 target/decoded simplex comparison")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args(argv)
    summary_path = args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".png" or args.output.exists() or summary_path.exists():
        parser.error("provide a fresh .png output and sidecar path")
    recipe = json.loads((args.run_results / "resolved_recipe.json").read_text())
    manifest = json.loads((args.run_results / "run_manifest.json").read_text())
    run = json.loads((args.run_results / "tune_summary.json").read_text())
    if manifest["status"] != "completed" or run["trials"][0]["error"] is not None or recipe["environment"] != environment_config():
        parser.error("the completed source run must match this cycle-2 environment")
    from ray.rllib.core.rl_module.rl_module import RLModule

    torch.set_num_threads(1)
    module = RLModule.from_checkpoint(str(args.checkpoint.resolve()))
    if module.observation_space.shape != (2,) or module.action_space.n != 2:
        parser.error("the checkpoint must have two-token observations and two actions")
    target, decoded, metadata = collect_and_fit(module, seed=args.seed, n_steps=args.steps)
    metadata.update({"agent_steps": int(run["trials"][0]["metrics"]["env_runners/num_env_steps_sampled_lifetime"]),
                     "checkpoint": str(args.checkpoint.resolve()), "source_run": str(args.run_results.resolve()),
                     "checkpoint_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.checkpoint.iterdir()) if path.is_file()},
                     "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), Path(analysis.__file__))},
                     "torch_version": torch.__version__, "numpy_version": np.__version__, "source_run_git": manifest["git"]})
    png = render_simplex(target, decoded, layer=metadata["layer"], n_fit=metadata["n_fit"], agent_steps=metadata["agent_steps"], seed=args.seed)
    metrics = geometry_metrics(target, decoded)
    payload = {"schema_version": 1, "metadata": metadata, "metrics": metrics, "png_sha256": hashlib.sha256(png).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(png)
    with summary_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output} and {summary_path}")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
