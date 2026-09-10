from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from analysis.belief_geometry import (
    evaluate_belief_geometry,
    prediction_null_basis,
)
from analysis.checkpoints import load_module_only
from analysis.probes import variance_geometry
from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_2.context_length_64_entropy_4x import (
    analysis as token_analysis,
)
from experiments.strata_token_guess_cycle_2.context_length_64_entropy_4x import (
    process as token_process,
)
from experiments.strata_two_factor_explore_cycle_3.context_length_64_entropy_4x import (
    analysis as factor_analysis,
)
from experiments.strata_two_factor_explore_cycle_3.context_length_64_entropy_4x import (
    process as factor_process,
)
from harness.seeding import named_seed_sequences, seed_sequence_to_int


DEFAULT_STEPS = 20_000
DEFAULT_NULL_REPEATS = 5
DEFAULT_BOOTSTRAP_RESAMPLES = 200
SMOKE_STEPS = 256
LOG_FLOOR = 1e-12
POLICY_TEMPERATURE = 1.5
_TOKEN_STREAM_KEYS = {
    "probe_train": (800,),
    "probe_test": (801,),
    "analysis": (804,),
}
_FACTOR_STREAM_KEYS = {
    "probe_train": (700,),
    "probe_test": (701,),
    "analysis": (704,),
}
_EMISSION_MATRIX = strata_model(
    alpha=factor_process.STRATA_ALPHA,
    t0=factor_process.STRATA_T0,
    t1=factor_process.STRATA_T1,
).emission_matrix
_NULL_BASIS = prediction_null_basis(_EMISSION_MATRIX)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def _module_path(checkpoint: Path) -> Path:
    nested = (
        checkpoint
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
    )
    return nested if nested.is_dir() else checkpoint


def _provenance(checkpoint: Path) -> dict[str, Any]:
    import analysis.belief_geometry as shared_geometry
    import analysis.checkpoints as shared_checkpoints
    import analysis.rollouts as shared_rollouts

    experiment_root = Path(__file__).resolve().parents[2]
    harness_root = Path(shared_geometry.__file__).resolve().parents[1]
    sources = (
        Path(__file__).resolve(),
        Path(token_analysis.__file__).resolve(),
        Path(factor_analysis.__file__).resolve(),
        Path(shared_geometry.__file__).resolve(),
        Path(shared_checkpoints.__file__).resolve(),
        Path(shared_rollouts.__file__).resolve(),
    )
    module_path = _module_path(checkpoint)
    return {
        "repositories": {
            str(experiment_root): _git_state(experiment_root),
            str(harness_root): _git_state(harness_root),
        },
        "source_sha256": {
            str(path): _sha256(path)
            for path in sources
        },
        "module_checkpoint_sha256": {
            path.name: _sha256(path)
            for path in sorted(module_path.iterdir())
            if path.is_file()
        },
        "framework_versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "ray", "torch")
            },
        },
        "command": list(sys.argv),
    }


def _policy_report(data: Any, *, token_guess: bool) -> dict[str, Any]:
    count = len(data.actions)
    report = {
        "mode": "learned_stochastic",
        "temperature": POLICY_TEMPERATURE,
        "mean_reward": float(np.mean(data.rewards)),
        "action_fractions": (
            np.bincount(data.actions, minlength=4 if token_guess else 9)
            / count
        ).tolist(),
    }
    if token_guess:
        report.update(
            token_accuracy=float(np.mean(data.actions == data.hidden_tokens)),
            hidden_token_fractions=(
                np.bincount(data.hidden_tokens, minlength=4) / count
            ).tolist(),
        )
    else:
        states = factor_process.decode_joint_indices(data.states)
        actions = factor_process.decode_joint_indices(data.actions)
        report.update(
            joint_state_fractions=(
                np.bincount(data.states, minlength=9) / count
            ).tolist(),
            factor_state_fractions={
                f"factor_{index + 1}": (
                    np.bincount(states[:, index], minlength=3) / count
                ).tolist()
                for index in range(2)
            },
            factor_action_fractions={
                f"factor_{index + 1}": (
                    np.bincount(actions[:, index], minlength=3) / count
                ).tolist()
                for index in range(2)
            },
        )
    return report


def _token_ntp(beliefs: np.ndarray) -> np.ndarray:
    transition = strata_model(
        alpha=token_process.STRATA_ALPHA,
        t0=token_process.STRATA_T0,
        t1=token_process.STRATA_T1,
    ).transition_matrix
    source = np.linalg.solve(transition.T, beliefs.T).T
    if (
        (source < -1e-10).any()
        or not np.allclose(source.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
        or not np.allclose(
            source @ transition,
            beliefs,
            atol=1e-10,
            rtol=0.0,
        )
    ):
        raise AssertionError("could not recover delay-one source beliefs")
    source = np.maximum(source, 0.0)
    source /= source.sum(axis=1, keepdims=True)
    return source @ _EMISSION_MATRIX


def _probability_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("probabilities must be a finite matrix")
    if (rows < -1e-12).any():
        raise ValueError("probabilities contain negative mass")
    rows = np.maximum(rows, 0.0)
    totals = rows.sum(axis=1, keepdims=True)
    if (totals <= 0.0).any():
        raise ValueError("probability row has no mass")
    return rows / totals


def _factor_battery(
    train: Any,
    test: Any,
    *,
    factor: int,
    token_guess: bool,
    seed: int,
    n_null_repeats: int,
    n_resamples: int,
) -> dict[str, Any]:
    train_beliefs = _probability_rows(train.factor_beliefs[:, factor])
    test_beliefs = _probability_rows(test.factor_beliefs[:, factor])
    train_ntp = (
        _token_ntp(train_beliefs)
        if token_guess
        else train_beliefs @ _EMISSION_MATRIX
    )
    test_ntp = (
        _token_ntp(test_beliefs)
        if token_guess
        else test_beliefs @ _EMISSION_MATRIX
    )
    layers = {
        f"layer_{index + 1}": train.activations[:, index]
        for index in range(train.activations.shape[1])
    }
    test_layers = {
        f"layer_{index + 1}": test.activations[:, index]
        for index in range(test.activations.shape[1])
    }
    contrasts = {
        f"prediction_null_{index + 1}": _NULL_BASIS[:, index]
        for index in range(_NULL_BASIS.shape[1])
    }
    result = evaluate_belief_geometry(
        layers,
        test_layers,
        train_beliefs,
        test_beliefs,
        train_groups=train.episode_ids,
        test_groups=test.episode_ids,
        nuisance_features={
            "current_observation": (train.observations, test.observations),
            "next_token_probability": (train_ntp, test_ntp),
            "log_next_token_probability": (
                np.log(np.maximum(train_ntp, LOG_FLOOR)),
                np.log(np.maximum(test_ntp, LOG_FLOOR)),
            ),
        },
        contrasts=contrasts,
        matched_keys=(train.observations, test.observations),
        seed=seed,
        n_null_repeats=n_null_repeats,
        n_resamples=n_resamples,
    )
    report = result.report
    report["activation_variance_geometry"] = {
        name: variance_geometry(test_layers[name])
        for name in test_layers
    }
    report["target_variance_geometry"] = variance_geometry(test_beliefs)
    report["control_definitions"] = {
        "current_observation": (
            "current delayed visible token one-hot"
            if token_guess
            else "current joint token plus preceding executed factor actions"
        ),
        "next_token_probability": (
            "pending hidden-token distribution from the delay-one source belief"
            if token_guess
            else "factor belief projected through the Strata emission matrix"
        ),
        "matched_feature_null": (
            "training activation rows sampled within exact current-observation cells"
        ),
        "initialization_comparison": (
            "reported as a separately sampled checkpoint, not fitted as same-history initialization features"
        ),
    }
    return report


def analyze_module(
    module: Any,
    *,
    study: str,
    condition: str,
    seed: int,
    n_steps: int,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
    device: torch.device,
    n_null_repeats: int = DEFAULT_NULL_REPEATS,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if study == "token_guess":
        if condition != "token_guess":
            raise ValueError("token_guess study requires token_guess condition")
        streams = named_seed_sequences(seed, _TOKEN_STREAM_KEYS)
        train = token_analysis.collect_probe_data(
            module,
            n_steps=n_steps,
            seed=streams["probe_train"],
            device=device,
        )
        test = token_analysis.collect_probe_data(
            module,
            n_steps=n_steps,
            seed=streams["probe_test"],
            device=device,
        )
        factor_count = 1
        target_timing = (
            "pre-action info.belief_current for the delayed current hidden token; "
            "the guess is scored against event.raw_token_before"
        )
    elif study == "two_factor":
        if condition not in factor_process.CONDITIONS:
            raise ValueError("unknown two-factor condition")
        streams = named_seed_sequences(seed, _FACTOR_STREAM_KEYS)
        train = factor_analysis.collect_probe_data(
            module,
            condition=condition,
            reward_state=factor_process.REWARD_STATE,
            n_steps=n_steps,
            seed=streams["probe_train"],
            device=device,
        )
        test = factor_analysis.collect_probe_data(
            module,
            condition=condition,
            reward_state=factor_process.REWARD_STATE,
            n_steps=n_steps,
            seed=streams["probe_test"],
            device=device,
        )
        factor_count = factor_process.FACTOR_COUNT
        target_timing = (
            "pre-action info.belief_current; the preceding executed action was "
            "already incorporated by the prior edge-belief update"
        )
    else:
        raise ValueError("study must be token_guess or two_factor")
    consistency = max(
        train.product_consistency_max_abs,
        test.product_consistency_max_abs,
    )
    if consistency > 1e-10:
        raise AssertionError(f"factor belief reconstruction failed: {consistency:.3e}")
    analysis_seed = seed_sequence_to_int(streams["analysis"], bits=32)
    factors = {
        ("strata" if factor_count == 1 else f"factor_{index + 1}"): _factor_battery(
            train,
            test,
            factor=index,
            token_guess=study == "token_guess",
            seed=analysis_seed + index,
            n_null_repeats=n_null_repeats,
            n_resamples=n_resamples,
        )
        for index in range(factor_count)
    }
    return {
        "schema_version": 1,
        "study": study,
        "condition": condition,
        "checkpoint": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "is_initialization": training_iteration == 0,
        "factors": factors,
        "policy": _policy_report(test, token_guess=study == "token_guess"),
        "train_policy": _policy_report(
            train,
            token_guess=study == "token_guess",
        ),
        "product_consistency_max_abs": consistency,
        "metadata": {
            "seed": seed,
            "analysis_seed": analysis_seed,
            "sampling_distribution": "process_weighted_rollout",
            "policy_mode": "learned_stochastic",
            "policy_temperature": POLICY_TEMPERATURE,
            "logits_already_temperature_scaled": True,
            "n_envs": 8,
            "warmup_per_episode": 64,
            "context_length": 64,
            "episode_length": 1024,
            "n_fit": len(train.activations),
            "n_test": len(test.activations),
            "sample_budgets_exclude_warmup": True,
            "complete_histories_collected_before_warmup_filter": True,
            "independent_train_test_rollouts": True,
            "rollout_seed_spawn_keys": {
                name: list(streams[name].spawn_key)
                for name in ("probe_train", "probe_test")
            },
            "train_episodes_represented": int(
                len(np.unique(train.episode_ids))
            ),
            "test_episodes_represented": int(
                len(np.unique(test.episode_ids))
            ),
            "representation": (
                "each transformer block current-position residual before final layer norm"
            ),
            "target_timing": target_timing,
            "belief_conditioning": (
                "observed tokens only, never actions or rewards"
                if study == "token_guess"
                else "observed tokens and executed actions, never rewards"
            ),
            "fit_method": "whole-episode grouped SVD-cutoff cross-validation",
            "bootstrap_unit": "held-out episode",
            "n_null_repeats": n_null_repeats,
            "n_bootstrap_resamples": n_resamples,
            "null_direction": _NULL_BASIS.tolist(),
            "log_probability_floor": LOG_FLOOR,
        },
        "scope_warning": (
            "Linear accessibility does not establish causal use or uniquely "
            "identify the model's internal representation. On-policy occupancy "
            "can differ across checkpoints."
        ),
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    factors = {}
    for factor_name, factor in report["factors"].items():
        baselines = {
            name: value["metrics"]
            for name, value in factor["baselines"].items()
        }
        layers = {}
        for layer_name, layer in factor["representations"].items():
            nulls = {}
            for null_name, repeats in layer["nulls"].items():
                values = [
                    item["metrics"]["r_squared"]
                    for item in repeats
                    if item["metrics"]["r_squared"] is not None
                ]
                nulls[null_name] = {
                    "r_squared_replicates": values,
                    "r_squared_mean": (
                        statistics.mean(values) if values else None
                    ),
                    "r_squared_sample_std": (
                        statistics.stdev(values) if len(values) > 1 else None
                    ),
                }
            layers[layer_name] = {
                "metrics": layer["metrics"],
                "comparisons": layer["comparisons"]["baselines"],
                "nulls": nulls,
                "activation_variance_geometry": factor[
                    "activation_variance_geometry"
                ][layer_name],
            }
        factors[factor_name] = {
            "baselines": baselines,
            "layers": layers,
            "target_variance_geometry": factor["target_variance_geometry"],
            "control_definitions": factor["control_definitions"],
        }
    return {
        "schema_version": 1,
        "study": report["study"],
        "condition": report["condition"],
        "checkpoint": report["checkpoint"],
        "agent_steps": report["agent_steps"],
        "training_iteration": report["training_iteration"],
        "is_initialization": report["is_initialization"],
        "policy": report["policy"],
        "factors": factors,
        "metadata": report["metadata"],
        "product_consistency_max_abs": report[
            "product_consistency_max_abs"
        ],
        "scope_warning": report["scope_warning"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Selected-checkpoint Strata belief-geometry analysis"
    )
    parser.add_argument(
        "--study",
        choices=("token_guess", "two_factor"),
        required=True,
    )
    parser.add_argument(
        "--condition",
        choices=("token_guess", "reward_both", "reward_factor_1"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--agent-steps", type=int, required=True)
    parser.add_argument("--training-iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    if args.output.exists() or summary_path.exists():
        parser.error("output or summary already exists")
    if args.steps <= 0 or args.agent_steps < 0 or args.training_iteration < 0:
        parser.error("steps and checkpoint coordinates must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    checkpoint = args.checkpoint.resolve()
    module_path = _module_path(checkpoint)
    if not module_path.is_dir():
        parser.error(f"module checkpoint does not exist: {module_path}")
    torch.set_num_threads(1)
    module = load_module_only(checkpoint)
    report = analyze_module(
        module,
        study=args.study,
        condition=args.condition,
        seed=args.seed,
        n_steps=SMOKE_STEPS if args.smoke else args.steps,
        checkpoint_label=args.checkpoint_label,
        agent_steps=args.agent_steps,
        training_iteration=args.training_iteration,
        device=torch.device(args.device),
        n_null_repeats=2 if args.smoke else DEFAULT_NULL_REPEATS,
        n_resamples=50 if args.smoke else DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    report["metadata"].update(
        {
            "checkpoint_path": str(checkpoint),
            "module_path": str(module_path),
            "provenance": _provenance(checkpoint),
        }
    )
    summary = _summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (args.output, report),
        (summary_path, summary),
    ):
        with path.open("x") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    print(f"Wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
