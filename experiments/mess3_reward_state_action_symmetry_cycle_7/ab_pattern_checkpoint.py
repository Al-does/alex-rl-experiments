"""Matched A/B endings, token-embedding interventions, and adjacent checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_logit_evidence import (
    EXPECTED_MODULE_SHA256,
    WARMUP,
    collect_pairs,
    condition,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    environment_config,
)


SEED = 20261104
PREVIOUS_SHA256 = "8ce5304cf4bedfd511e858894c77f0b53557c30dc74b113a247ec01c0a4c4394"
PATTERN_FIRST = np.array([0, 0, 1, 1])
PATTERN_SECOND = np.array([0, 1, 0, 1])


def episode_grid(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = np.unique(data["episode"])
    indices = []
    for episode in ids:
        rows = np.flatnonzero(data["episode"] == episode)
        rows = rows[np.argsort(data["step"][rows])]
        np.testing.assert_array_equal(data["step"][rows], np.arange(len(rows)))
        indices.append(rows)
    indices = np.stack(indices)
    return {key: value[indices] for key, value in data.items()}


def observations(grid: dict[str, np.ndarray], token_key: str = "token") -> np.ndarray:
    tokens = grid[token_key]
    result = np.zeros((*tokens.shape, 6), dtype=np.float32)
    result[..., :3] = np.eye(3, dtype=np.float32)[tokens]
    actions = grid["previous_action"]
    nonreset = actions >= 0
    result[..., 3:][nonreset] = np.eye(3, dtype=np.float32)[actions[nonreset]]
    return result


def embedding_variant(module, mode: str):
    """Intervene on a copy; action columns, bias, and all later weights stay fixed."""
    if mode not in ("swap", "mean"):
        raise ValueError("embedding mode must be swap or mean")
    altered = copy.deepcopy(module)
    with torch.no_grad():
        weight = altered.encoder.input_projection.weight
        if mode == "swap":
            weight[:, [0, 1]] = weight[:, [1, 0]]
        else:
            weight[:, :2] = weight[:, :2].mean(dim=1, keepdim=True)
    return altered.eval()


def pattern_beliefs(grid: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply AA, AB, BA, BB to a common exact predictive prefix."""
    env = HMMEnv(environment_config(2))
    try:
        emission = np.asarray(env.model.emission_matrix, dtype=np.float64)
        transitions = np.stack(
            [env.task.transition_matrix_for_action(a) for a in range(3)]
        )
        initial = env.model.initial_distribution
    finally:
        env.close()
    count, horizon = grid["token"].shape
    prior = np.tile(initial, (count, 1))
    previous_prior = None
    rows = {key: [] for key in ("episode", "step", "rho")}
    for step in range(horizon):
        if step >= WARMUP:
            eligible = (
                (grid["token"][:, step - 1] != 2)
                & (grid["token"][:, step] != 2)
            )
            indices = np.flatnonzero(eligible)
            first = condition(
                np.repeat(previous_prior[indices], 4, axis=0),
                emission, np.tile(PATTERN_FIRST, len(indices)),
            )
            between = np.repeat(grid["previous_action"][indices, step], 4)
            predicted = np.einsum("ni,nij->nj", first, transitions[between])
            second = condition(
                predicted, emission, np.tile(PATTERN_SECOND, len(indices)),
            )
            rows["episode"].append(grid["episode"][indices, step])
            rows["step"].append(np.full(len(indices), step))
            rows["rho"].append(second[:, 2].reshape(-1, 4))
        posterior = condition(prior, emission, grid["token"][:, step])
        np.testing.assert_allclose(posterior[:, 2], grid["rho"][:, step], atol=1e-12)
        previous_prior = prior
        if step + 1 < horizon:
            prior = np.einsum(
                "ni,nij->nj", posterior,
                transitions[grid["previous_action"][:, step + 1]],
            )
    return {key: np.concatenate(parts) for key, parts in rows.items()}


@torch.inference_mode()
def replay(
    module,
    grid: dict[str, np.ndarray],
    *,
    mode: str,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    baseline = observations(grid)
    randomized = observations(grid, "randomized_token")
    swapped = baseline.copy()
    swapped[..., [0, 1]] = swapped[..., [1, 0]]
    streams = {
        "original": (baseline, randomized, swapped),
        "swap": (baseline,),
        "mean": (baseline, randomized),
    }[mode]
    module.eval().to("cpu")
    count, horizon = baseline.shape[:2]
    all_logits = np.empty((count, horizon, len(streams), 3), dtype=np.float64)
    pattern_rows = {key: [] for key in ("episode", "step", "logits")}
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        n = end - start
        state = _initial_state(module, n * len(streams), torch.device("cpu"))
        before_previous = None
        for step in range(horizon):
            if mode != "swap" and step >= WARMUP:
                eligible = np.flatnonzero(
                    (grid["token"][start:end, step - 1] != 2)
                    & (grid["token"][start:end, step] != 2)
                )
                if len(eligible):
                    fork = {
                        key: value[eligible].repeat_interleave(4, dim=0)
                        for key, value in before_previous.items()
                    }
                    first = np.repeat(baseline[start:end][eligible, step - 1], 4, axis=0)
                    first[:, :3] = np.eye(3)[np.tile(PATTERN_FIRST, len(eligible))]
                    second = np.repeat(baseline[start:end][eligible, step], 4, axis=0)
                    second[:, :3] = np.eye(3)[np.tile(PATTERN_SECOND, len(eligible))]
                    _, fork = module.encode_step(torch.from_numpy(first), fork)
                    hidden, _ = module.encode_step(torch.from_numpy(second), fork)
                    logits = module.action_distribution_inputs(hidden).double().numpy()
                    pattern_rows["episode"].append(grid["episode"][start:end][eligible, step])
                    pattern_rows["step"].append(np.full(len(eligible), step))
                    pattern_rows["logits"].append(logits.reshape(-1, 4, 3))
            before_current = state
            batch = np.concatenate([stream[start:end, step] for stream in streams])
            hidden, state = module.encode_step(torch.from_numpy(batch), state)
            logits = module.action_distribution_inputs(hidden).double().numpy()
            all_logits[start:end, step] = logits.reshape(len(streams), n, 3).transpose(1, 0, 2)
            before_previous = before_current
        print(f"{mode} replay: {end}/{count} episodes", flush=True)
    result = {"logits": all_logits}
    if mode != "swap":
        result.update({
            f"pattern_{key}": np.concatenate(parts)
            for key, parts in pattern_rows.items()
        })
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def bootstrap_mean(values: np.ndarray, groups: np.ndarray, *, resamples: int) -> dict:
    _, codes = np.unique(groups, return_inverse=True)
    totals = np.bincount(codes, weights=values)
    counts = np.bincount(codes)
    draws = np.random.default_rng(SEED).integers(
        len(totals), size=(resamples, len(totals)),
    )
    estimates = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "n_rows": len(values),
        "n_episodes": len(totals),
    }


def distribution_change(base: np.ndarray, altered: np.ndarray, groups, resamples) -> dict:
    variation = 0.5 * np.abs(softmax(base) - softmax(altered)).sum(axis=-1)
    preference_change = (base[:, 0] - base[:, 1]) - (altered[:, 0] - altered[:, 1])
    return {
        "total_variation": bootstrap_mean(variation, groups, resamples=resamples),
        "rms_logit_preference_change": float(np.sqrt(np.mean(preference_change ** 2))),
        "greedy_disagreements": int(np.sum(base.argmax(-1) != altered.argmax(-1))),
        "n_decisions": len(base),
    }


def pattern_contrasts(values: np.ndarray) -> dict[str, np.ndarray]:
    aa, ab, ba, bb = values.T
    return {
        "alternating_minus_repeated": (ab + ba - aa - bb) / 2,
        "current_A_minus_B": (aa + ba - ab - bb) / 2,
        "previous_A_minus_B": (aa + ab - ba - bb) / 2,
    }


def pattern_summary(result: dict, targets: dict, resamples: int) -> dict:
    order = np.lexsort((result["pattern_step"], result["pattern_episode"]))
    target_order = np.lexsort((targets["step"], targets["episode"]))
    for key in ("episode", "step"):
        np.testing.assert_array_equal(result[f"pattern_{key}"][order], targets[key][target_order])
    groups = result["pattern_episode"][order]
    logits = result["pattern_logits"][order]
    preference = logits[..., 0] - logits[..., 1]
    contrasts = pattern_contrasts(preference)
    evidence = pattern_contrasts(targets["rho"][target_order])["alternating_minus_repeated"]
    summaries = {}
    for name, value in contrasts.items():
        summaries[name] = {
            **bootstrap_mean(value, groups, resamples=resamples),
            "rms": float(np.sqrt(np.mean(value ** 2))),
        }
    pattern = contrasts["alternating_minus_repeated"]
    return {
        "logit_contrasts": summaries,
        "noop_probability_contrasts": {
            name: bootstrap_mean(value, groups, resamples=resamples)
            for name, value in pattern_contrasts(softmax(logits)[..., 0]).items()
        },
        "exact_reward_evidence_contrast": bootstrap_mean(evidence, groups, resamples=resamples),
        "fraction_pattern_changes_aligned_with_evidence": float(np.mean(pattern * evidence > 0)),
        "pattern_share_of_squared_factorial_contrasts": (
            float(np.sum(pattern ** 2) / sum(np.sum(v ** 2) for v in contrasts.values()))
            if np.any(preference != preference[:, :1]) else None
        ),
        "max_logit_spread_across_four_endings": float(np.ptp(logits, axis=1).max()),
        "prefixes_with_any_greedy_action_difference": int(np.sum(
            np.any(logits.argmax(-1) != logits[:, :1].argmax(-1), axis=1)
        )),
    }


def checkpoint_summary(variants: dict, grid: dict, targets: dict, resamples: int) -> dict:
    mask = grid["step"] >= WARMUP
    groups = grid["episode"][mask]
    original = variants["original"]["logits"][mask]
    swapped = variants["swap"]["logits"][mask, 0, :]
    merged = variants["mean"]["logits"][mask]
    base, randomized, globally_swapped = original[:, 0], original[:, 1], original[:, 2]
    error = float(np.max(np.abs(swapped - globally_swapped)))
    np.testing.assert_allclose(swapped, globally_swapped, rtol=0, atol=5e-5)
    np.testing.assert_allclose(merged[:, 0], merged[:, 1], rtol=0, atol=5e-5)
    probability = softmax(base)
    sorted_logits = np.sort(base, axis=1)
    return {
        "baseline": {
            "entropy_nats": bootstrap_mean(
                -np.sum(probability * np.log(probability), axis=1), groups, resamples=resamples,
            ),
            "mean_max_action_probability": float(probability.max(axis=1).mean()),
            "mean_top_two_logit_margin": float(np.mean(sorted_logits[:, -1] - sorted_logits[:, -2])),
        },
        "randomized_AB": distribution_change(base, randomized, groups, resamples),
        "global_AB_swap": distribution_change(base, globally_swapped, groups, resamples),
        "embedding_swap": distribution_change(base, swapped, groups, resamples),
        "mean_AB_embeddings": distribution_change(base, merged[:, 0], groups, resamples),
        "embedding_controls": {
            "max_swap_embedding_vs_global_input_swap_logit_error": error,
            "max_randomization_logit_error_with_mean_embeddings": float(np.max(
                np.abs(merged[:, 0] - merged[:, 1])
            )),
            "interpretation": (
                "Swapping input-projection columns equals global input relabeling; "
                "merging columns removes the distinction by construction. "
                "Neither alone localizes the source of the original sensitivity."
            ),
        },
        "matched_endings": pattern_summary(variants["original"], targets, resamples),
        "matched_endings_mean_embeddings": pattern_summary(variants["mean"], targets, resamples),
    }


def paired_checkpoint_summary(replays: dict, grid: dict, resamples: int) -> dict:
    mask = grid["step"] >= WARMUP
    groups = grid["episode"][mask]
    final = replays["final"]["original"]["logits"][mask]
    previous = replays["previous"]["original"]["logits"][mask]
    variations = {}
    for label, logits in (("final", final), ("previous", previous)):
        variations[label] = 0.5 * np.abs(
            softmax(logits[:, 0]) - softmax(logits[:, 1])
        ).sum(axis=1)
    return {
        "previous_minus_final_randomization_TV": bootstrap_mean(
            variations["previous"] - variations["final"], groups, resamples=resamples,
        ),
        "baseline_greedy_disagreements_between_checkpoints": int(np.sum(
            final[:, 0].argmax(-1) != previous[:, 0].argmax(-1)
        )),
        "previous_logit_perturbation_at_final_baseline": distribution_change(
            final[:, 0], final[:, 0] + previous[:, 1] - previous[:, 0], groups, resamples,
        ),
        "final_logit_perturbation_at_previous_baseline": distribution_change(
            previous[:, 0], previous[:, 0] + final[:, 1] - final[:, 0], groups, resamples,
        ),
        "recombination_caveat": (
            "The transferred-perturbation quantities recombine output logits "
            "to describe softmax sensitivity, not an intervention inside a network."
        ),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-download", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stem = "smoke" if args.smoke else "full"
    history_path = args.scratch / f"{stem}_histories.npz"
    raw_path = args.scratch / f"{stem}_replay.npz"
    for path in (args.output, history_path, raw_path):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    expected = (EXPECTED_MODULE_SHA256, PREVIOUS_SHA256)
    paths = (args.final_checkpoint, args.previous_checkpoint)
    for path, digest in zip(paths, expected):
        if sha256(path / "module_state.pkl") != digest:
            parser.error(f"unexpected checkpoint hash at {path}")
    torch.set_num_threads(2)
    final = load_module_only(args.final_checkpoint)
    previous = load_module_only(args.previous_checkpoint)
    if final.sequence_lookback != WARMUP or previous.sequence_lookback != WARMUP:
        raise ValueError("expected the original 128-step receptive field")
    data = collect_pairs(
        final, split=2, episodes=4 if args.smoke else 64, seed=SEED,
        horizon=256 if args.smoke else 1024,
    )
    args.scratch.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(history_path, **data)
    grid = episode_grid(data)
    targets = pattern_beliefs(grid)
    replays = {}
    for name, module in (("final", final), ("previous", previous)):
        print(f"Checkpoint: {name}", flush=True)
        replays[name] = {
            "original": replay(module, grid, mode="original"),
            "swap": replay(embedding_variant(module, "swap"), grid, mode="swap"),
            "mean": replay(embedding_variant(module, "mean"), grid, mode="mean"),
        }
    final_preference = (
        replays["final"]["original"]["logits"][:, :, 0, 0]
        - replays["final"]["original"]["logits"][:, :, 0, 1]
    )
    np.testing.assert_allclose(final_preference, grid["logit"], rtol=0, atol=5e-5)
    np.savez_compressed(raw_path, **{
        f"{checkpoint}_{mode}_{key}": value
        for checkpoint, variants in replays.items()
        for mode, result in variants.items()
        for key, value in result.items()
    })
    resamples = 100 if args.smoke else 2000
    results = {
        name: checkpoint_summary(variants, grid, targets, resamples)
        for name, variants in replays.items()
    }
    results["paired_checkpoint_comparison"] = paired_checkpoint_summary(replays, grid, resamples)
    library = Path(inspect.getfile(load_module_only)).resolve().parents[1]
    repository = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.run_manifest.read_text())
    report = {
        "protocol": {
            "condition": "Reward-State Quotient; training seed 42",
            "seed": SEED,
            "episodes": len(grid["episode"]),
            "episode_length": grid["step"].shape[1],
            "warmup": WARMUP,
            "histories": "fresh final-policy greedy trajectories; same inputs and fixed executed actions for all model copies",
            "matched_endings": "AA, AB, BA, BB replace the last two symbols only when both baseline symbols were A/B; shared prefix and actions",
            "pattern_contrast": "(AB + BA - AA - BB) / 2; positive Noop-minus-Tilt-X contrast predicted from finer reward evidence",
            "label_contrasts": "balanced current-label and previous-label contrasts reported separately",
            "embeddings": "swap or average columns 0 and 1 of encoder.input_projection.weight in independent model copies; restart all caches",
            "checkpoint_selection": "immediately preceding durable checkpoint, not preceding log-spaced probe checkpoint",
            "bootstrap": f"{resamples} paired episode resamples; sample-weighted means; fixed comparisons without fitted probes",
            "limitations": "one seed; frozen final-policy histories are off-policy for altered/earlier models; forced suffixes may be uncommon; probabilities are not reward estimates; no long-horizon optimality claim",
        },
        "provenance": {
            "training_run_id": manifest["run_id"],
            "training_revisions": manifest["git"],
            "analysis_revisions": {
                name: {
                    "commit": subprocess.check_output(
                        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                    ).strip(),
                    "dirty": bool(subprocess.check_output(
                        ["git", "-C", str(path), "status", "--porcelain"], text=True,
                    ).strip()),
                }
                for name, path in (("experiment", repository), ("library", library))
            },
            "final": {"checkpoint": "checkpoint_000025", "agent_steps": 10192528, "sha256": expected[0]},
            "previous": json.loads(args.previous_download.read_text()),
            "source_sha256": sha256(Path(__file__)),
            "collector_source_sha256": sha256(Path(inspect.getfile(collect_pairs))),
            "history_sha256": sha256(history_path),
            "replay_sha256": sha256(raw_path),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
