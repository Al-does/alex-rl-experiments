"""Extend the paired A/B-randomization comparison to checkpoint 23."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_pattern_checkpoint import (
    SEED,
    WARMUP,
    bootstrap_mean,
    distribution_change,
    episode_grid,
    observations,
    sha256,
    softmax,
)


@torch.inference_mode()
def replay_pair(module, grid: dict[str, np.ndarray]) -> np.ndarray:
    original = observations(grid)
    randomized = observations(grid, "randomized_token")
    count, horizon = original.shape[:2]
    logits = np.empty((count, horizon, 2, 3), dtype=np.float64)
    module.eval().to("cpu")
    for start in range(0, count, 16):
        end = min(start + 16, count)
        n = end - start
        state = _initial_state(module, 2 * n, torch.device("cpu"))
        for step in range(horizon):
            inputs = np.concatenate((
                original[start:end, step], randomized[start:end, step],
            ))
            hidden, state = module.encode_step(torch.from_numpy(inputs), state)
            scores = module.action_distribution_inputs(hidden).double().numpy()
            logits[start:end, step] = scores.reshape(2, n, 3).transpose(1, 0, 2)
        print(f"Paired replay: {end}/{count} episodes", flush=True)
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--download-report", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--previous-replay", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.raw_output):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    previous = json.loads(args.previous_report.read_text())
    download = json.loads(args.download_report.read_text())
    provenance = previous["provenance"]
    for path, expected in (
        (args.histories, provenance["history_sha256"]),
        (args.previous_replay, provenance["replay_sha256"]),
        (args.checkpoint / "module_state.pkl", download["module_state_sha256"]),
        (args.final_checkpoint / "module_state.pkl", provenance["final"]["sha256"]),
    ):
        if sha256(path) != expected:
            parser.error(f"hash mismatch: {path}")
    if download["checkpoint_name"] != "checkpoint_000023":
        parser.error("this extension expects checkpoint_000023")
    grid = episode_grid(dict(np.load(args.histories)))
    if grid["step"].shape != (64, 1024):
        parser.error("expected the original 64 complete 1,024-step histories")
    torch.set_num_threads(2)
    saved = np.load(args.previous_replay)
    final = load_module_only(args.final_checkpoint)
    if final.sequence_lookback != WARMUP:
        raise ValueError("unexpected final-model receptive field")
    # Check the lean two-stream replay against the original saved implementation.
    validation = replay_pair(final, {key: value[:4] for key, value in grid.items()})
    expected = saved["final_original_logits"][:4, :, :2]
    np.testing.assert_allclose(validation, expected, rtol=0, atol=5e-5)
    replay_error = float(np.max(np.abs(validation - expected)))
    del final
    module = load_module_only(args.checkpoint)
    if module.sequence_lookback != WARMUP:
        raise ValueError("unexpected checkpoint-23 receptive field")
    logits = replay_pair(module, grid)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_output, logits=logits)
    mask = grid["step"] >= WARMUP
    groups = grid["episode"][mask]
    checkpoints = {
        "checkpoint_000023": (download["agent_steps"], logits[mask]),
        "checkpoint_000024": (9799312, saved["previous_original_logits"][mask, :2, :]),
        "checkpoint_000025": (10192528, saved["final_original_logits"][mask, :2, :]),
    }
    results, variations = {}, {}
    for name, (steps, scores) in checkpoints.items():
        base, randomized = scores[:, 0], scores[:, 1]
        p = softmax(base)
        variations[name] = 0.5 * np.abs(p - softmax(randomized)).sum(axis=-1)
        results[name] = {
            "agent_steps": steps,
            **distribution_change(base, randomized, groups, 2000),
            "baseline_entropy_nats": float(np.mean(-np.sum(p * np.log(p), axis=1))),
            "baseline_mean_max_action_probability": float(p.max(axis=1).mean()),
        }
    paired = {}
    for other in ("checkpoint_000024", "checkpoint_000025"):
        key = f"checkpoint_000023_minus_{other}"
        paired[key] = bootstrap_mean(
            variations["checkpoint_000023"] - variations[other], groups, resamples=2000,
        )
        paired[key]["relative_mean_TV_change"] = float(
            variations["checkpoint_000023"].mean() / variations[other].mean() - 1
        )
    old = previous["results"]
    for name, label in (("checkpoint_000024", "previous"), ("checkpoint_000025", "final")):
        np.testing.assert_allclose(
            results[name]["total_variation"]["mean"],
            old[label]["randomized_AB"]["total_variation"]["mean"], rtol=0, atol=1e-15,
        )
    library = Path(inspect.getfile(load_module_only)).resolve().parents[1]
    report = {
        "protocol": {
            "training_seed": 42, "evaluation_seed": SEED, "split": 2,
            "episodes": 64, "horizon": 1024, "warmup": WARMUP,
            "scored_decisions": int(mask.sum()),
            "comparison": "original versus independently A/B-randomized inputs; C and final-policy executed actions preserved",
            "history_source": "identical saved final-policy histories and randomization draws as the preceding comparison",
            "bootstrap": "2,000 paired-episode resamples, decision-weighted means",
            "selection": "checkpoint 23 requested after observing checkpoint 24 > 25; descriptive extension, not independent confirmation of a training trend",
            "caveats": "one training seed, late adjacent checkpoints, frozen histories rather than earlier-policy on-policy visitation",
        },
        "provenance": {
            "download": download,
            "previous_report_path": str(args.previous_report),
            "previous_report_sha256": sha256(args.previous_report),
            "history_sha256": sha256(args.histories),
            "previous_replay_sha256": sha256(args.previous_replay),
            "raw_output_sha256": sha256(args.raw_output),
            "source_sha256": sha256(Path(__file__)),
            "helper_source_sha256": sha256(Path(inspect.getfile(observations))),
            "experiment_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "experiment_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "library_revision": subprocess.check_output(["git", "-C", str(library), "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        },
        "validation": {"four_episode_final_replay_max_logit_error": replay_error},
        "results": results,
        "paired_comparisons": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"results": results, "paired_comparisons": paired}, indent=2))


if __name__ == "__main__":
    main()
