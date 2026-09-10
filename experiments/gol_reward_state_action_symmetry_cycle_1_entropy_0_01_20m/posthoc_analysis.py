from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.belief_geometry import (
    evaluate_belief_geometry,
    prediction_null_basis,
)
from analysis.checkpoints import load_module_only
from analysis.plots import plot_belief_comparison
from analysis.probes import (
    cluster_bootstrap_statistics,
    fit_grouped_affine,
    paired_comparison,
    percentile_interval,
    score_prediction,
    suffix_keys,
)
from envs.gol.model import AGGREGATION, STATES, controlled_kernels, gol_model
from experiments.gol_reward_state_action_symmetry_cycle_1_entropy_0_01_20m.analysis import (
    ProbeData,
    branch_keys,
    collect_probe_data,
    marginal_features,
    replay_representations,
    shuffled_histories,
)
from harness.seeding import named_seed_sequences, seed_sequence_to_int


EXPERIMENT = "gol_reward_state_action_symmetry_cycle_1_entropy_0_01_20m"
REPRESENTATION = "post_final_layer_norm"
SELECTED_CHECKPOINT_COUNT = 3
HISTORY_LENGTHS = (1, 2, 4, 8)
LOG_FLOOR = 1e-12
STREAM_KEYS = {
    "fit": (810,),
    "test": (811,),
    "shuffle_fit": (812,),
    "shuffle_test": (813,),
    "probe": (814,),
    "task_bootstrap": (815,),
}


def joint_outcome_features(
    beliefs: np.ndarray,
    kernels: np.ndarray,
) -> np.ndarray:
    beliefs = np.asarray(beliefs, dtype=np.float64)
    kernels = np.asarray(kernels, dtype=np.float64)
    if beliefs.ndim != 2 or beliefs.shape[1] != 4:
        raise ValueError("beliefs must have shape (N, 4)")
    if kernels.shape != (4, 2, 4, 4):
        raise ValueError("kernels must have action/token/source/destination axes")
    outcomes = np.empty((len(beliefs), 4, 2, 2), dtype=np.float64)
    destination_is_rewarded = np.arange(4) == 2
    for reward in range(2):
        edge = kernels[..., destination_is_rewarded == bool(reward)].sum(axis=-1)
        outcomes[..., reward] = np.einsum("ns,ats->nat", beliefs, edge)
    return outcomes.reshape(len(beliefs), -1)


def replay_beliefs(
    histories: np.ndarray,
    *,
    variant: int,
    speed: str,
) -> np.ndarray:
    histories = np.asarray(histories)
    if histories.ndim != 3 or histories.shape[2] != 6:
        raise ValueError("histories must have shape (rounds, environments, 6)")
    branch_keys(histories.reshape(-1, 6))
    if np.any(histories[0]):
        raise ValueError("histories must begin with reset observations")
    kernels = controlled_kernels(variant, speed)
    prior = gol_model(variant, speed).initial_distribution
    beliefs = np.empty((*histories.shape[:2], 4), dtype=np.float64)
    beliefs[0] = prior
    for step in range(1, len(histories)):
        token = histories[step, :, :2].argmax(axis=1)
        action = histories[step, :, 2:].argmax(axis=1)
        for environment in range(histories.shape[1]):
            unnormalized = (
                beliefs[step - 1, environment]
                @ kernels[action[environment], token[environment]]
            )
            beliefs[step, environment] = unnormalized / unnormalized.sum()
    return beliefs


def selected_history_keys(
    data: ProbeData,
    length: int,
) -> np.ndarray:
    rounds, n_envs, _ = data.history_observations.shape
    symbols = branch_keys(data.history_observations.reshape(-1, 6))
    groups = np.tile(np.arange(n_envs), rounds)
    steps = np.repeat(np.arange(rounds), n_envs)
    return suffix_keys(symbols, groups, steps, length)[data.sample_indices]


def categorical_prediction(
    train_keys: np.ndarray,
    test_keys: np.ndarray,
    train_targets: np.ndarray,
) -> np.ndarray:
    train_keys = np.asarray(train_keys)
    test_keys = np.asarray(test_keys)
    if train_keys.ndim == 1:
        train_keys = train_keys[:, None]
    if test_keys.ndim == 1:
        test_keys = test_keys[:, None]
    if train_keys.shape[1] != test_keys.shape[1]:
        raise ValueError("train and test keys must have equal widths")
    prediction = np.broadcast_to(
        train_targets.mean(axis=0),
        (len(test_keys), train_targets.shape[1]),
    ).copy()
    totals: dict[tuple[int, ...], np.ndarray] = {}
    counts: dict[tuple[int, ...], int] = {}
    for key, target in zip(train_keys, train_targets):
        key_tuple = tuple(int(value) for value in key)
        totals[key_tuple] = totals.get(
            key_tuple,
            np.zeros(train_targets.shape[1], dtype=np.float64),
        ) + target
        counts[key_tuple] = counts.get(key_tuple, 0) + 1
    centroids = {
        key: target / counts[key]
        for key, target in totals.items()
    }
    for index, key in enumerate(test_keys):
        prediction[index] = centroids.get(
            tuple(int(value) for value in key),
            prediction[index],
        )
    return prediction


@torch.inference_mode()
def policy_probabilities(module: Any, activations: np.ndarray) -> np.ndarray:
    logits = module.action_distribution_inputs(
        torch.as_tensor(activations, dtype=torch.float32)
    )
    return logits.softmax(dim=-1).cpu().numpy().astype(np.float64)


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        key: score.get(key)
        for key in (
            "mse",
            "target_variance",
            "normalized_mse",
            "r_squared",
            "outside_simplex_fraction",
            "max_sum_error",
            "n_evaluated",
        )
        if key in score
    }


def _compact_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        key: comparison.get(key)
        for key in (
            "mse_improvement",
            "mse_improvement_ci",
            "delta_r_squared",
            "delta_r_squared_ci",
            "residual_fraction_recovered",
            "ci_reason",
            "delta_r_squared_ci_reason",
            "n_groups",
            "n_resamples",
        )
    }


def _null_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        record["metrics"]["r_squared"]
        for record in records
        if record["metrics"]["r_squared"] is not None
    ]
    return {
        "n_repeats": len(records),
        "r_squared": [record["metrics"]["r_squared"] for record in records],
        "r_squared_mean": None if not values else float(np.mean(values)),
        "r_squared_std": (
            None if len(values) < 2 else float(np.std(values, ddof=1))
        ),
    }


def _history_controls(
    train: ProbeData,
    test: ProbeData,
    prediction: np.ndarray,
    *,
    groups: np.ndarray,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    result = {}
    for length in HISTORY_LENGTHS:
        train_keys = selected_history_keys(train, length)
        test_keys = selected_history_keys(test, length)
        train_key_set = {tuple(key) for key in train_keys}
        baseline = categorical_prediction(train_keys, test_keys, train.beliefs)
        result[f"visible_suffix_{length}"] = {
            "metrics": _compact_score(score_prediction(baseline, test.beliefs)),
            "probe_comparison": _compact_comparison(
                paired_comparison(
                    prediction,
                    baseline,
                    test.beliefs,
                    groups,
                    seed=seed,
                    n_resamples=n_resamples,
                )
            ),
            "fit_method": "training_suffix_centroid_with_train_mean_fallback",
            "key_width": length,
            "test_exact_key_coverage": float(
                np.mean([tuple(row) in train_key_set for row in test_keys])
            ),
        }
    return result


def _mean_with_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    estimates = cluster_bootstrap_statistics(
        groups,
        lambda indices: float(values[indices].mean()),
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "mean": float(values.mean()),
        "ci95": list(percentile_interval(estimates)),
        "n": len(values),
        "n_groups": len(np.unique(groups)),
        "n_resamples": n_resamples,
    }


def _task_metrics(
    data: ProbeData,
    probabilities: np.ndarray,
    kernels: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    reward = marginal_features(data.beliefs, kernels)["next_reward"]
    chosen = reward[np.arange(len(data.actions)), data.actions]
    return {
        "realized_reward": _mean_with_ci(
            data.rewards,
            data.env_indices,
            seed=seed,
            n_resamples=n_resamples,
        ),
        "expected_chosen_next_reward": _mean_with_ci(
            chosen,
            data.env_indices,
            seed=seed,
            n_resamples=n_resamples,
        ),
        "oracle_myopic_on_visited_beliefs": _mean_with_ci(
            reward.max(axis=1),
            data.env_indices,
            seed=seed,
            n_resamples=n_resamples,
        ),
        "action_fractions": (
            np.bincount(data.actions, minlength=4) / len(data.actions)
        ).tolist(),
        "mean_policy_probabilities": probabilities.mean(axis=0).tolist(),
        "interpretation": (
            "On-policy post-warmup process-weighted estimates. The oracle "
            "quantity is a counterfactual one-step reference on visited "
            "beliefs, not a long-run attainable policy bound."
        ),
    }


def _shuffled_control(
    module: Any,
    train: ProbeData,
    test: ProbeData,
    *,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    streams: dict[str, np.random.SeedSequence],
    n_resamples: int,
) -> dict[str, Any]:
    shifted_train = replay_representations(
        module,
        shuffled_histories(
            train.history_observations,
            streams["shuffle_fit"],
        ),
    )[train.sample_indices]
    shifted_test = replay_representations(
        module,
        shuffled_histories(
            test.history_observations,
            streams["shuffle_test"],
        ),
    )[test.sample_indices]
    seed = seed_sequence_to_int(streams["probe"])
    weight, bias, fit = fit_grouped_affine(
        shifted_train,
        train.beliefs,
        train_groups,
        seed=seed,
    )
    prediction = shifted_test @ weight + bias
    mean = np.broadcast_to(train.beliefs.mean(axis=0), test.beliefs.shape)
    return {
        "metrics": _compact_score(score_prediction(prediction, test.beliefs)),
        "train_mean_comparison": _compact_comparison(
            paired_comparison(
                prediction,
                mean,
                test.beliefs,
                test_groups,
                seed=seed,
                n_resamples=n_resamples,
            )
        ),
        "fit": {
            key: fit[key]
            for key in ("method", "n_features", "n_groups", "selected_rcond", "rank")
        },
        "protocol": (
            "Action-token observation rows are independently permuted within "
            "each complete continuing history after the reset row. The network "
            "is replayed from empty state and fitted to the original time-indexed "
            "belief targets, deliberately breaking alignment."
        ),
    }


def analyze_checkpoint(
    module: Any,
    initialization_module: Any,
    checkpoint: dict[str, Any],
    *,
    variant: int,
    speed: str,
    seed: int,
    n_steps: int,
    n_envs: int,
    n_null_repeats: int,
    n_resamples: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    lookback = int(module.sequence_lookback)
    if lookback != int(initialization_module.sequence_lookback):
        raise ValueError("selected checkpoints must share a receptive field")
    streams = named_seed_sequences(seed, STREAM_KEYS)
    collection = {
        "variant": variant,
        "speed": speed,
        "n_steps": n_steps,
        "n_envs": n_envs,
        "warmup": lookback,
    }
    train = collect_probe_data(module, seed=streams["fit"], **collection)
    test = collect_probe_data(module, seed=streams["test"], **collection)
    replay_error = max(
        float(
            np.max(
                np.abs(
                    replay_beliefs(
                        data.history_observations,
                        variant=variant,
                        speed=speed,
                    ).reshape(-1, 4)[data.sample_indices]
                    - data.beliefs
                )
            )
        )
        for data in (train, test)
    )
    if replay_error > 1e-10:
        raise ValueError(
            f"replayed beliefs disagree with environment diagnostics: {replay_error}"
        )
    initialization = tuple(
        replay_representations(
            initialization_module,
            data.history_observations,
        )[data.sample_indices]
        for data in (train, test)
    )
    train_policy = policy_probabilities(module, train.activations)
    test_policy = policy_probabilities(module, test.activations)
    kernels = controlled_kernels(variant, speed)
    train_marginals = marginal_features(train.beliefs, kernels)
    test_marginals = marginal_features(test.beliefs, kernels)
    train_joint = joint_outcome_features(train.beliefs, kernels)
    test_joint = joint_outcome_features(test.beliefs, kernels)
    train_branch = branch_keys(train.observations)
    test_branch = branch_keys(test.observations)
    train_groups = np.asarray([f"fit/{value}" for value in train.env_indices])
    test_groups = np.asarray([f"test/{value}" for value in test.env_indices])
    nuisance = {
        "current_token_previous_action": (
            np.eye(9)[train_branch],
            np.eye(9)[test_branch],
        ),
        "all_action_next_token_marginals": (
            train_marginals["next_token"],
            test_marginals["next_token"],
        ),
        "all_action_next_reward_marginals": (
            train_marginals["next_reward"],
            test_marginals["next_reward"],
        ),
        "separate_token_reward_marginals": (
            train_marginals["combined_marginals"],
            test_marginals["combined_marginals"],
        ),
        "joint_token_reward_outcomes": (train_joint, test_joint),
        "policy_probabilities": (train_policy, test_policy),
        "centered_log_policy": (
            np.log(np.maximum(train_policy, LOG_FLOOR))
            - np.log(np.maximum(train_policy, LOG_FLOOR)).mean(
                axis=1,
                keepdims=True,
            ),
            np.log(np.maximum(test_policy, LOG_FLOOR))
            - np.log(np.maximum(test_policy, LOG_FLOOR)).mean(
                axis=1,
                keepdims=True,
            ),
        ),
    }
    probe_seed = seed_sequence_to_int(streams["probe"])
    result = evaluate_belief_geometry(
        {REPRESENTATION: train.activations},
        {REPRESENTATION: test.activations},
        train.beliefs,
        test.beliefs,
        train_groups=train_groups,
        test_groups=test_groups,
        nuisance_features=nuisance,
        contrasts={
            "m1_minus_m2": np.array([1.0, -1.0, 0.0, 0.0]),
            "e_minus_s": np.array([0.0, 0.0, 1.0, -1.0]),
        },
        initialization_features={
            REPRESENTATION: initialization,
        },
        matched_keys=(train_branch, test_branch),
        seed=probe_seed,
        n_null_repeats=n_null_repeats,
        n_resamples=n_resamples,
    )
    report = result.report
    representation = report["representations"][REPRESENTATION]
    prediction = result.predictions[REPRESENTATION]
    baselines = {
        name.removeprefix("nuisance/"): {
            "metrics": _compact_score(record["metrics"]),
            "probe_comparison": _compact_comparison(
                representation["comparisons"]["baselines"][name]
            ),
        }
        for name, record in report["baselines"].items()
    }
    initialization_record = report["initialization"][REPRESENTATION]
    outcome_map = joint_outcome_features(np.eye(4), kernels).reshape(4, 4, 2, 2)
    null_basis = prediction_null_basis(outcome_map)
    null_metrics = (
        None
        if null_basis.shape[1] == 0
        else _compact_score(
            score_prediction(
                prediction @ null_basis,
                test.beliefs @ null_basis,
            )
        )
    )
    checkpoint_report = {
        "label": checkpoint["label"],
        "agent_steps": checkpoint["agent_steps"],
        "training_iteration": checkpoint["training_iteration"],
        "probe": {
            "metrics": _compact_score(representation["metrics"]),
            "contrasts": {
                name: _compact_score(value)
                for name, value in representation["metrics"]["contrasts"].items()
            },
            "coarse_aggregation_metrics": _compact_score(
                score_prediction(
                    prediction @ AGGREGATION,
                    test.beliefs @ AGGREGATION,
                )
            ),
            "fit": {
                key: representation["fit"][key]
                for key in (
                    "method",
                    "n_features",
                    "n_groups",
                    "selected_rcond",
                    "rank",
                )
            },
            "comparisons": {
                "baselines": baselines,
                "same_history_initialization": _compact_comparison(
                    representation["comparisons"]["initialization"]
                ),
            },
            "nulls": {
                name: _null_summary(records)
                for name, records in representation["nulls"].items()
            },
            "prediction_map_null": {
                "basis": null_basis.tolist(),
                "dimension": null_basis.shape[1],
                "metrics": null_metrics,
                "map": "all-action joint next-token/reward outcome probabilities",
            },
        },
        "initialization": {
            "metrics": _compact_score(initialization_record["metrics"]),
            "same_sample_histories": True,
        },
        "history_controls": _history_controls(
            train,
            test,
            prediction,
            groups=test_groups,
            seed=probe_seed,
            n_resamples=n_resamples,
        ),
        "shuffled_history_stress": _shuffled_control(
            module,
            train,
            test,
            train_groups=train_groups,
            test_groups=test_groups,
            streams=streams,
            n_resamples=n_resamples,
        ),
        "task_performance": _task_metrics(
            test,
            test_policy,
            kernels,
            seed=seed_sequence_to_int(streams["task_bootstrap"]),
            n_resamples=n_resamples,
        ),
        "verification": {
            "belief_replay_max_abs_error": replay_error,
            "warmup_frames_per_environment": lookback,
            "model_sequence_lookback": lookback,
            "first_scored_episode_step": int(
                min(train.episode_steps.min(), test.episode_steps.min())
            ),
            "fit_test_seed_spawn_keys": {
                name: list(streams[name].spawn_key)
                for name in ("fit", "test")
            },
        },
    }
    return checkpoint_report, test.beliefs, prediction


def select_checkpoints(
    summary: dict[str, Any],
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    records = sorted(
        summary["checkpoint_reports"],
        key=lambda record: record["agent_steps"],
    )
    selected = [records[0], *records[-2:]]
    if (
        len(selected) != SELECTED_CHECKPOINT_COUNT
        or selected[0]["agent_steps"] != 0
        or len({record["agent_steps"] for record in selected}) != len(selected)
    ):
        raise ValueError("checkpoint summary lacks initialization and two late checkpoints")
    result = []
    for record in selected:
        steps = int(record["agent_steps"])
        local = checkpoint_root / (
            "initial" if steps == 0 else f"steps_{steps:09d}"
        )
        if not (local / "module_state.pkl").is_file():
            raise FileNotFoundError(f"incomplete module-only checkpoint: {local}")
        result.append(
            {
                "label": (
                    "initial"
                    if steps == 0
                    else "penultimate"
                    if record is selected[-2]
                    else "final"
                ),
                "source_checkpoint_label": record["checkpoint_label"],
                "agent_steps": steps,
                "training_iteration": int(record["training_iteration"]),
                "path": local,
            }
        )
    return result


def checkpoint_provenance(
    selected: list[dict[str, Any]],
    durability: dict[str, Any],
) -> list[dict[str, Any]]:
    files = durability["files"]
    records = []
    for checkpoint in selected:
        source_label = checkpoint["source_checkpoint_label"]
        candidates = [
            item
            for item in files
            if item["relative_path"].endswith(
                "/learner_group/learner/rl_module/default_policy/module_state.pkl"
            )
            and (
                item["relative_path"].startswith("initial_checkpoint/")
                if checkpoint["agent_steps"] == 0
                else f"/{source_label}/" in f"/{item['relative_path']}"
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one durable module for {source_label}, found {len(candidates)}"
            )
        prefix = candidates[0]["relative_path"].removesuffix("/module_state.pkl")
        local_hashes = {}
        durable_hashes = {}
        for path in sorted(checkpoint["path"].iterdir()):
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            durable = [
                item
                for item in files
                if item["relative_path"] == f"{prefix}/{path.name}"
            ]
            if len(durable) != 1 or durable[0]["sha256"] != digest:
                raise ValueError(f"checkpoint hash mismatch: {path}")
            local_hashes[path.name] = digest
            durable_hashes[path.name] = durable[0]["sha256"]
        records.append(
            {
                "label": checkpoint["label"],
                "agent_steps": checkpoint["agent_steps"],
                "training_iteration": checkpoint["training_iteration"],
                "source_checkpoint_label": source_label,
                "durability_manifest_prefix": prefix,
                "file_sha256": local_hashes,
                "hashes_match_durability_manifest": local_hashes == durable_hashes,
            }
        )
    return records


def _repository_revision(root: Path) -> dict[str, Any]:
    return {
        "root": str(root.resolve()),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
            ).strip()
        ),
    }


def _source_provenance(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def write_figures(
    output: Path,
    checkpoints: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    plot_data: list[tuple[np.ndarray, np.ndarray]],
    *,
    variant: int,
    seed: int,
) -> list[str]:
    names = []
    for index in (0, len(checkpoints) - 1):
        checkpoint = checkpoints[index]
        targets, predictions = plot_data[index]
        selection = np.random.default_rng(seed).choice(
            len(targets),
            size=min(2_000, len(targets)),
            replace=False,
        )
        figure = plot_belief_comparison(
            targets[selection],
            predictions[selection],
            state_labels=STATES,
            title=(
                f"GoL variant {variant}: {checkpoint['label']} "
                f"({checkpoint['agent_steps']:,} steps)"
            ),
            seed=seed,
        )
        name = f"belief_geometry_{checkpoint['label']}.png"
        figure.savefig(output / name, bbox_inches="tight")
        plt.close(figure)
        names.append(name)
    steps = np.asarray([item["agent_steps"] for item in reports])
    reward = np.asarray(
        [item["task_performance"]["realized_reward"]["mean"] for item in reports]
    )
    probe = np.asarray([item["probe"]["metrics"]["r_squared"] for item in reports])
    reward_ci = np.asarray(
        [item["task_performance"]["realized_reward"]["ci95"] for item in reports]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].plot(steps, reward, marker="o")
    axes[0].fill_between(steps, reward_ci[:, 0], reward_ci[:, 1], alpha=0.2)
    axes[0].set_ylabel("Realized reward")
    axes[1].plot(steps, probe, marker="o")
    axes[1].set_ylabel("Held-out fine-belief R²")
    for axis in axes:
        axis.set_xlabel("Agent steps")
        axis.set_xscale("symlog", linthresh=1)
        axis.grid(alpha=0.25)
    figure.suptitle(f"GoL variant {variant}: selected-checkpoint history")
    figure.tight_layout()
    name = "selected_checkpoint_history.png"
    figure.savefig(output / name, bbox_inches="tight")
    plt.close(figure)
    names.append(name)
    return names


def _limitations(variant: int) -> str:
    quotient = (
        "Variant 2 has an exact coarse quotient; the reported prediction-map "
        "null direction exposes one-step predictive indistinguishability of M1/M2."
        if variant == 2
        else "Variant 3 coarse values are only an aggregation; they are not an exact quotient."
    )
    return f"""# Limitations

- Affine belief decoding measures held-out linear accessibility. It does not show causal use and does not uniquely identify the network's internal representation.
- Initialization is replayed on the exact same sampled histories. High initialization scores weaken any claim that training created the geometry.
- Exact predictive controls can explain apparent belief decoding; probe surplus over them is descriptive rather than a causal mediation result.
- Fit and test contain 16 independent continuing environment trajectories each. Bootstrap intervals resample complete held-out trajectories with fixed probes; they do not measure training-seed variation.
- Checkpoints induce different stochastic-policy, process-weighted sampling distributions. Their scores are not evaluations on one common occupancy distribution.
- Task reward and probe fit are separate quantities. The myopic oracle is evaluated on visited beliefs and is not a certified Bayes-optimal POMDP bound.
- History shuffling deliberately breaks the original target alignment and changes the input distribution. It is a stress control, not a null required to yield zero R².
- Only the actual initialization and final two available checkpoints were probed; intermediate geometry is unmeasured.
- {quotient}
"""


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.condition_summary.read_text())
    run_manifest = json.loads(args.run_manifest.read_text())
    durability = json.loads(args.durability_manifest.read_text())
    if summary["condition"] != f"variant_{args.variant}":
        raise ValueError("condition summary variant mismatch")
    if run_manifest["run_id"] != args.run_id:
        raise ValueError("run manifest ID mismatch")
    checkpoints = select_checkpoints(summary, args.checkpoint_root)
    checkpoint_records = checkpoint_provenance(checkpoints, durability)
    initialization_module = load_module_only(checkpoints[0]["path"]).to("cpu").eval()
    reports = []
    plot_data = []
    for checkpoint in checkpoints:
        module = load_module_only(checkpoint["path"]).to("cpu").eval()
        report, targets, prediction = analyze_checkpoint(
            module,
            initialization_module,
            checkpoint,
            variant=args.variant,
            speed=args.speed,
            seed=args.seed,
            n_steps=args.steps,
            n_envs=args.environments,
            n_null_repeats=args.null_repeats,
            n_resamples=args.bootstrap_resamples,
        )
        reports.append(report)
        plot_data.append((targets, prediction))
    figure_names = write_figures(
        output,
        checkpoints,
        reports,
        plot_data,
        variant=args.variant,
        seed=args.seed,
    )
    metrics = {
        "schema_version": 1,
        "study": EXPERIMENT,
        "variant": args.variant,
        "run_id": args.run_id,
        "speed": args.speed,
        "checkpoint_selection": (
            "actual initialization plus final two available checkpoint records"
        ),
        "checkpoints": reports,
        "scope_warning": (
            "Held-out affine accessibility is not evidence of causal use or a "
            "unique internal belief representation. Task reward is separate."
        ),
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    limitations_path = output / "limitations.md"
    limitations_path.write_text(_limitations(args.variant))
    experiment_root = Path(__file__).resolve().parent
    repository_root = experiment_root.parents[1]
    harness_root = Path(sys.modules["analysis"].__file__).resolve().parents[1]
    provenance = {
        "schema_version": 1,
        "run_id": args.run_id,
        "variant": args.variant,
        "training_provenance": {
            "experiment_repository": run_manifest["git"]["experiment_repository"],
            "library": run_manifest["git"]["library"],
            "framework_versions": run_manifest["framework_versions"],
            "dependency_lock": run_manifest["dependency_lock"],
        },
        "analysis_repositories": {
            "experiment_repository": _repository_revision(repository_root),
            "rl_harness": _repository_revision(harness_root),
        },
        "analysis_framework_versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "torch", "ray", "matplotlib", "scipy")
            },
        },
        "source_sha256": _source_provenance(
            [
                Path(__file__),
                experiment_root / "analysis.py",
                experiment_root / "process.py",
                experiment_root / "shared.py",
                harness_root / "analysis" / "belief_geometry.py",
                harness_root / "analysis" / "probes" / "controls.py",
                harness_root / "analysis" / "plots.py",
                harness_root / "analysis" / "rollouts.py",
                harness_root / "envs" / "gol" / "model.py",
                harness_root / "envs" / "hmm" / "env.py",
            ]
        ),
        "checkpoint_modules": checkpoint_records,
        "analysis_configuration": {
            "command": list(sys.argv),
            "seed": args.seed,
            "named_seed_spawn_keys": {
                name: list(value)
                for name, value in STREAM_KEYS.items()
            },
            "n_fit": args.steps,
            "n_test": args.steps,
            "n_environments_per_split": args.environments,
            "warmup": (
                "model.sequence_lookback frames per environment; verified in metrics"
            ),
            "n_null_repeats": args.null_repeats,
            "n_bootstrap_resamples": args.bootstrap_resamples,
            "device": "cpu",
            "policy_mode": "stochastic",
            "sampling_distribution": (
                "learned-policy process-weighted continuing rollouts"
            ),
            "fit_protocol": (
                "training-only whole-trajectory grouped SVD-cutoff CV; "
                "independent held-out trajectories"
            ),
            "target_dtype": "float64",
            "prediction_metrics": "raw, unprojected, and unnormalized",
            "plot_points": "deterministic subsample of at most 2000 held-out rows",
        },
        "timing_contract": {
            "target": (
                "b_t from info['belief_current'] at decision time, before "
                "env.step(a_t)"
            ),
            "representation": (
                "post-final-LayerNorm embedding used to parameterize action a_t"
            ),
            "actor_inputs": (
                "latest token one-hot (2) and previous executed action one-hot "
                "(4); all zero at reset"
            ),
            "reward_conditions_actor_or_filter": False,
            "continuing_sampling": (
                "batch truncation does not reset environment, filter, or model state"
            ),
            "warmup_masking": (
                "complete histories retained from reset; scoring starts only "
                "after the full model sequence lookback"
            ),
        },
        "outputs": ["metrics.json", "limitations.md", *figure_names],
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--variant", type=int, choices=(2, 3), required=True)
    value.add_argument("--run-id", required=True)
    value.add_argument("--condition-summary", type=Path, required=True)
    value.add_argument("--run-manifest", type=Path, required=True)
    value.add_argument("--durability-manifest", type=Path, required=True)
    value.add_argument("--checkpoint-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--speed", choices=("half", "quarter"), default="half")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--steps", type=int, default=10_000)
    value.add_argument("--environments", type=int, default=16)
    value.add_argument("--null-repeats", type=int, default=5)
    value.add_argument("--bootstrap-resamples", type=int, default=200)
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: list[str] | None = None) -> None:
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
