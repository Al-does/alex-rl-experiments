"""Scalar belief-component probes for the cycle-7 quotient ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from analysis.probes import (
    fit_affine_probe,
    global_mse_metrics,
    probe_predict,
    r2_score,
)
from harness.seeding import named_seed_sequences

from experiments.mess3_reward_state_action_symmetry_cycle_7.control_analysis import (
    N_ENVS,
    collect_history_data,
    replay_beliefs,
)


FIT_STEPS = 60_000
TEST_STEPS = 80_000
WARMUP = 64
RIDGE = 1e-6
_PROBE_STREAMS = {
    "probe_train": (510,),
    "probe_test": (511,),
}
TARGETS = {
    "rho": {
        "symbol": "rho_t",
        "definition": "b_3,t",
        "description": "posterior occupancy of reward state 3",
    },
    "delta": {
        "symbol": "delta_t",
        "definition": "b_1,t - b_2,t",
        "description": "posterior distinction between states 1 and 2",
    },
}


def component_targets(beliefs: np.ndarray) -> dict[str, np.ndarray]:
    """Project exact three-state beliefs onto manuscript scalar targets."""

    values = np.asarray(beliefs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("beliefs must have shape (n, 3)")
    return {
        "rho": values[:, 2:3],
        "delta": values[:, 0:1] - values[:, 1:2],
    }


def score_target(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
) -> dict[str, object]:
    """Fit one affine probe and return held-out normalized-error metrics."""

    weight, bias = fit_affine_probe(
        train_features,
        train_target,
        ridge=RIDGE,
    )
    predicted = probe_predict(weight, bias, test_features)
    metrics = global_mse_metrics(predicted, test_target)
    r_squared = r2_score(predicted, test_target)
    normalized = float(metrics["global_mse_ratio"])
    if abs(normalized - (1.0 - r_squared)) > 1e-10:
        raise ValueError("normalized MSE is inconsistent with 1 - R^2")
    return {
        **metrics,
        "r_squared": r_squared,
        "n_evaluated": int(len(test_target)),
        "fit": {
            "method": "affine_ridge_least_squares",
            "ridge": RIDGE,
            "n_features": int(train_features.shape[1]),
            "n_targets": int(train_target.shape[1]),
        },
    }


def analyze_module_components(
    module: object,
    *,
    variant: int,
    seed: int,
    fit_steps: int = FIT_STEPS,
    test_steps: int = TEST_STEPS,
    warmup: int = WARMUP,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Fit rho and delta probes on independent greedy-policy rollouts."""

    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if fit_steps < 1 or test_steps < 1:
        raise ValueError("fit_steps and test_steps must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    device = torch.device(device)
    streams = named_seed_sequences(seed, _PROBE_STREAMS)
    train = collect_history_data(
        module,
        variant=variant,
        n_steps=fit_steps,
        seed=streams["probe_train"],
        device=device,
        warmup=warmup,
    )
    test = collect_history_data(
        module,
        variant=variant,
        n_steps=test_steps,
        seed=streams["probe_test"],
        device=device,
        warmup=warmup,
    )
    replay_error = max(
        float(
            np.max(
                np.abs(
                    replay_beliefs(train, variant=variant) - train.beliefs
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    replay_beliefs(test, variant=variant) - test.beliefs
                )
            )
        ),
    )
    if replay_error > 1e-10:
        raise ValueError(
            "belief replay disagrees with environment diagnostics: "
            f"{replay_error}"
        )

    train_targets = component_targets(train.beliefs[train.mask])
    test_targets = component_targets(test.beliefs[test.mask])
    targets = {
        name: {
            **TARGETS[name],
            **score_target(
                train.activations[train.mask],
                test.activations[test.mask],
                train_targets[name],
                test_targets[name],
            ),
        }
        for name in TARGETS
    }
    return {
        "schema_version": 1,
        "metadata": {
            "variant": variant,
            "seed": seed,
            "n_fit": fit_steps,
            "n_test": test_steps,
            "n_envs": N_ENVS,
            "warmup": warmup,
            "device": str(device),
            "policy_mode": "greedy_argmax",
            "sampling_distribution": "process_weighted_rollout",
            "representation": "post_final_layer_norm",
            "fit_protocol": (
                "fixed-ridge affine probe fit on one rollout stream and "
                "scored on an independent rollout stream"
            ),
            "target_source": (
                "exact decision-time Bayesian belief from public environment "
                "diagnostics, independently replay-verified"
            ),
            "scope_warning": (
                "Affine decodability measures linear accessibility, not "
                "causal policy use."
            ),
        },
        "belief_replay_max_abs_error": replay_error,
        "targets": targets,
    }


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fit-steps", type=int, default=FIT_STEPS)
    parser.add_argument("--test-steps", type=int, default=TEST_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    repository = Path(__file__).resolve().parents[2]
    harness = Path(__import__("analysis").__file__).resolve().parents[1]
    module = load_module_only(args.checkpoint)
    report = analyze_module_components(
        module,
        variant=args.variant,
        seed=args.seed,
        fit_steps=args.fit_steps,
        test_steps=args.test_steps,
        warmup=args.warmup,
        device=args.device,
    )
    report["source"] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_files": {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(args.checkpoint.iterdir())
            if path.is_file()
        },
        "experiment_git_commit": _git(repository, "rev-parse", "HEAD"),
        "experiment_git_dirty": bool(
            _git(repository, "status", "--porcelain")
        ),
        "harness_git_commit": _git(harness, "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
