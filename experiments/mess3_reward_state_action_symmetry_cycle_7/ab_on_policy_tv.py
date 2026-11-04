"""A/B sensitivity on each checkpoint's own greedy and sampled trajectories."""

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
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_pattern_checkpoint import (
    SEED as BOOTSTRAP_SEED,
    WARMUP,
    bootstrap_mean,
    distribution_change,
    episode_grid,
    sha256,
    softmax,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    environment_config,
)


EVALUATION_SEED = 20261105


def choose_actions(probabilities: np.ndarray, uniforms: np.ndarray, mode: str) -> np.ndarray:
    if mode == "greedy":
        return probabilities.argmax(axis=1)
    if mode != "sampled":
        raise ValueError("action mode must be greedy or sampled")
    cumulative = probabilities.cumsum(axis=1)
    cumulative[:, -1] = 1.0
    return (uniforms[:, None] >= cumulative).sum(axis=1)


@torch.inference_mode()
def collect(
    module, *, mode: str, episodes: int, horizon: int, seed: int,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    config = environment_config(2)
    config.update(randomize_first_episode_length=False, episode_length=horizon)
    rows: dict[str, list[np.ndarray]] = {}
    module.eval().to("cpu")
    for start in range(0, episodes, batch_size):
        ids = np.arange(start, min(start + batch_size, episodes))
        n = len(ids)
        envs = [HMMEnv(config) for _ in ids]
        try:
            observations = [
                env.reset(seed=int(np.random.SeedSequence([seed, int(i), 0])
                                   .generate_state(1)[0]))[0]
                for env, i in zip(envs, ids)
            ]
            symbol_rng = [
                np.random.default_rng(np.random.SeedSequence([seed, int(i), 1]))
                for i in ids
            ]
            action_rng = [
                np.random.default_rng(np.random.SeedSequence([seed, int(i), 2]))
                for i in ids
            ]
            state = _initial_state(module, 2 * n, torch.device("cpu"))
            previous = np.full(n, -1, dtype=np.int64)
            for step in range(horizon):
                baseline = np.stack(observations)
                token = baseline[:, :3].argmax(axis=1)
                coins = np.array([rng.integers(2) for rng in symbol_rng])
                randomized_token = np.where(token == 2, 2, coins)
                altered = baseline.copy()
                altered[:, :3] = np.eye(3)[randomized_token]
                hidden, state = module.encode_step(
                    torch.from_numpy(np.concatenate((baseline, altered))), state,
                )
                logits = module.action_distribution_inputs(hidden).double().numpy()
                p = softmax(logits[:n])
                uniforms = np.array([rng.random() for rng in action_rng])
                actions = choose_actions(p, uniforms, mode)
                values = {
                    "episode": ids, "step": np.full(n, step),
                    "token": token, "randomized_token": randomized_token,
                    "previous_action": previous, "action": actions,
                    "logits": logits[:n], "shadow_logits": logits[n:],
                }
                for key, value in values.items():
                    rows.setdefault(key, []).append(value.copy())
                if step + 1 < horizon:
                    outcomes = [env.step(int(a)) for env, a in zip(envs, actions)]
                    observations = [outcome[0] for outcome in outcomes]
                    if any(outcome[2] or outcome[3] for outcome in outcomes):
                        raise RuntimeError("episode ended before the requested horizon")
                    previous = actions
        finally:
            for env in envs:
                env.close()
        print(f"{mode}: {start + n}/{episodes} own-policy episodes", flush=True)
    return {key: np.concatenate(parts) for key, parts in rows.items()}


def summary(data: dict[str, np.ndarray], resamples: int) -> dict:
    mask = data["step"] >= WARMUP
    base, shadow = data["logits"][mask], data["shadow_logits"][mask]
    p = softmax(base)
    return {
        **distribution_change(base, shadow, data["episode"][mask], resamples),
        "baseline_entropy_nats": float(np.mean(-np.sum(p * np.log(p), axis=1))),
        "baseline_mean_max_action_probability": float(p.max(axis=1).mean()),
        "selected_non_greedy_actions": int(np.sum(data["action"][mask] != base.argmax(-1))),
        "selected_action_fractions": (
            np.bincount(data["action"][mask], minlength=3) / mask.sum()
        ).tolist(),
    }


def compare(left: dict, right: dict, resamples: int) -> dict:
    grids = [episode_grid(data) for data in (left, right)]
    for key in ("episode", "step"):
        np.testing.assert_array_equal(grids[0][key], grids[1][key])
    mask = grids[0]["step"] >= WARMUP
    variations = [
        0.5 * np.abs(softmax(g["logits"]) - softmax(g["shadow_logits"])).sum(axis=-1)
        for g in grids
    ]
    token_diff = grids[0]["token"] != grids[1]["token"]
    action_diff = grids[0]["action"] != grids[1]["action"]
    return {
        "left_minus_right_TV": bootstrap_mean(
            (variations[0] - variations[1])[mask], grids[0]["episode"][mask],
            resamples=resamples,
        ),
        "token_difference_fraction_scored": float(token_diff[mask].mean()),
        "action_difference_fraction_scored": float(action_diff[mask].mean()),
        "episodes_with_different_tokens_including_warmup": int(np.sum(token_diff.any(axis=1))),
        "episodes_with_different_actions_including_warmup": int(np.sum(action_diff.any(axis=1))),
        "total_token_disagreements_including_warmup": int(token_diff.sum()),
        "total_action_disagreements_including_warmup": int(action_diff.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    arm = Path(__file__).resolve().parent / "variant_2_ctx32_ent"
    checkpoints = {
        "checkpoint_000023": (
            arm / "artifacts/ab_checkpoint23_tv_seed42/module", 9406096,
            "415118945953743695e96ed2c81791175edc9f4a647fb31f1236d742c8c0ac5c",
        ),
        "checkpoint_000024": (
            arm / "artifacts/ab_pattern_checkpoint_seed42/previous_module", 9799312,
            "8ce5304cf4bedfd511e858894c77f0b53557c30dc74b113a247ec01c0a4c4394",
        ),
        "checkpoint_000025": (
            arm / "artifacts/ab_logit_evidence_seed42/module", 10192528,
            "6ff2b60203e298c7967cd16e2a0ad5676fbc22c288e02e2515a38a06a635cc5b",
        ),
    }
    paths = {
        (mode, checkpoint): args.scratch / f"{mode}_{checkpoint}.npz"
        for mode in ("greedy", "sampled") for checkpoint in checkpoints
    }
    for path in (args.output, *paths.values()):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    for path, _, expected in checkpoints.values():
        if sha256(path / "module_state.pkl") != expected:
            parser.error(f"checkpoint hash mismatch: {path}")
    torch.set_num_threads(2)
    episodes, horizon = (4, 256) if args.smoke else (64, 1024)
    resamples = 100 if args.smoke else 2000
    data = {mode: {} for mode in ("greedy", "sampled")}
    args.scratch.mkdir(parents=True, exist_ok=True)
    for checkpoint, (path, _, _) in checkpoints.items():
        module = load_module_only(path)
        if module.sequence_lookback != WARMUP:
            raise ValueError("unexpected model receptive field")
        for mode in data:
            print(f"{checkpoint}: {mode}", flush=True)
            collected = collect(
                module, mode=mode, episodes=episodes, horizon=horizon,
                seed=EVALUATION_SEED,
            )
            np.savez_compressed(paths[mode, checkpoint], **collected)
            data[mode][checkpoint] = collected
    results, comparisons = {}, {}
    for mode, records in data.items():
        results[mode] = {
            name: summary(collected, resamples) for name, collected in records.items()
        }
        comparisons[mode] = {
            f"{left}_minus_{right}": compare(records[left], records[right], resamples)
            for left, right in (
                ("checkpoint_000023", "checkpoint_000024"),
                ("checkpoint_000024", "checkpoint_000025"),
                ("checkpoint_000023", "checkpoint_000025"),
            )
        }
    library = Path(inspect.getfile(load_module_only)).resolve().parents[1]
    report = {
        "protocol": {
            "training_seed": 42, "evaluation_seed": EVALUATION_SEED,
            "episodes_per_checkpoint_and_mode": episodes,
            "horizon": horizon, "warmup": WARMUP,
            "action_modes": ["greedy", "sampled"],
            "histories": "each checkpoint controls its own environment, without observation randomization",
            "shadow": "independent policy cache receives A/B-randomized inputs, preserving C and original executed actions; never controls the environment",
            "sampling": "unmodified softmax; inverse-CDF sampling with common uniforms across checkpoints",
            "seed_keys": "SeedSequence([evaluation_seed, episode, stream]); stream 0 environment, 1 symbol draws, 2 action uniforms; common across checkpoints and modes",
            "bootstrap": f"{resamples} paired-episode resamples, decision-weighted means",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "limitations": "one training seed, three late adjacent checkpoints, no causal identification of the role of residual A/B information",
        },
        "provenance": {
            "checkpoints": {
                name: {"agent_steps": steps, "module_state_sha256": digest}
                for name, (_, steps, digest) in checkpoints.items()
            },
            "source_sha256": sha256(Path(__file__)),
            "helper_source_sha256": sha256(Path(inspect.getfile(bootstrap_mean))),
            "raw": {
                f"{mode}_{checkpoint}": {"path": str(path), "sha256": sha256(path)}
                for (mode, checkpoint), path in paths.items()
            },
            "experiment_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "experiment_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "library_revision": subprocess.check_output(["git", "-C", str(library), "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        },
        "results": results, "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"results": results, "paired_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
