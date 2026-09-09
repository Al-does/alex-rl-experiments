from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
from itertools import permutations
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from analysis.probes import split_group_indices
from analysis.probes.controls import (
    fit_grouped_affine,
    gaussian_feature_null,
    matched_feature_null,
    paired_comparison,
    paired_target_comparison,
    score_prediction,
    suffix_keys,
)
from envs.wing.model import controlled_kernels
from experiments.wing_two_factor_explore_cycle_1.control_data import (
    ControlData,
    candidate_parameters,
    collect_control_data,
    replay_beliefs,
)


SUFFIX_LENGTHS = (0, 1, 2, 4, 8, 16, 32)
LOG_FLOOR = 1e-12
MAX_ALTERNATIVE_KL = 0.02
MIN_ALTERNATIVE_RESIDUAL = 0.01
ALTERNATIVE_COUNT = 3


def _fit(train_x, train_y, test_x, groups, seed):
    weight, bias, fit = fit_grouped_affine(train_x, train_y, groups, seed=seed)
    return test_x @ weight + bias, fit


def _log(probabilities):
    return np.log(np.maximum(probabilities, LOG_FLOOR))


def _mean_kl(target, alternative):
    return float(np.mean(np.sum(target * (_log(target) - _log(alternative)), axis=-1)))


def _symbols(data, controlled):
    tokens = np.where(data.tokens[:, 0] >= 0, data.tokens[:, 0] * 2 + data.tokens[:, 1], 4)
    if not controlled:
        return tokens
    actions = np.where(
        data.preceding_actions[:, 0] >= 0,
        data.preceding_actions[:, 0] * 3 + data.preceding_actions[:, 1], 9,
    )
    return tokens * 10 + actions


def _select_alternatives(train, parameters, *, seed):
    mask = train.mask
    target, ntp = replay_beliefs(train, **parameters)
    target, ntp = target[mask], ntp[mask]
    groups = train.episode_ids[mask]
    fit_rows, validation_rows = split_group_indices(groups, test_fraction=0.25, seed=seed)
    candidates = []
    for index, candidate in enumerate(candidate_parameters(**parameters)):
        beliefs, predictions = replay_beliefs(train, **candidate)
        beliefs, predictions = beliefs[mask], predictions[mask]
        irreducible, permutation_error = [], []
        for factor in range(2):
            x, y = target[:, factor], beliefs[:, factor]
            design = np.column_stack((np.ones(len(fit_rows)), x[fit_rows]))
            coefficients = np.linalg.lstsq(design, y[fit_rows], rcond=1e-10)[0]
            predicted = np.column_stack((np.ones(len(validation_rows)), x[validation_rows])) @ coefficients
            metrics = score_prediction(predicted, y[validation_rows])
            irreducible.append(metrics["normalized_mse"])
            permutation_error.append(min(
                float(np.mean((x[validation_rows][:, order] - y[validation_rows]) ** 2))
                for order in permutations(range(3))
            ))
        kl = _mean_kl(ntp[validation_rows], predictions[validation_rows])
        residual = min(value if value is not None else 0.0 for value in irreducible)
        eligible = (
            kl <= MAX_ALTERNATIVE_KL
            and residual >= MIN_ALTERNATIVE_RESIDUAL
            and min(permutation_error) > 1e-8
        )
        candidates.append({
            "id": f"alternative_{index + 1}",
            "parameters": candidate,
            "selection_next_observation_kl": kl,
            "selection_affine_residual_ratio_per_factor": irreducible,
            "selection_min_permutation_mse_per_factor": permutation_error,
            "selection_score": residual,
            "eligible": eligible,
        })
    selected = sorted(
        (candidate for candidate in candidates if candidate["eligible"]),
        key=lambda candidate: (-candidate["selection_score"], candidate["id"]),
    )[:ALTERNATIVE_COUNT]
    return selected, {
        "selection_scope": "training_episodes_only; no activations or test histories used",
        "criterion": "largest minimum factor affine residual subject to next-observation KL budget",
        "max_mean_kl_nats": MAX_ALTERNATIVE_KL,
        "kl_aggregation": "mean_over_samples_and_factors; joint_token_KL_is_twice_this_value_for_two_independent_factors",
        "min_affine_residual_ratio": MIN_ALTERNATIVE_RESIDUAL,
        "selection_fit_episodes": int(len(np.unique(groups[fit_rows]))),
        "selection_validation_episodes": int(len(np.unique(groups[validation_rows]))),
        "candidates": candidates,
        "selected_ids": [item["id"] for item in selected],
        "status": "selected" if selected else "no_eligible_alternative",
        "warning": "A finite candidate search is not a worst-case guarantee; alternatives are separate models, not averaged beliefs.",
    }


def _matched_pairs(keys, groups):
    cells = {}
    for index, key in enumerate(np.asarray(keys)):
        cells.setdefault(tuple(np.atleast_1d(key)), []).append(index)
    pairs = []
    for indices in cells.values():
        remaining = list(indices)
        while len(remaining) > 1:
            first = remaining.pop()
            partner = next((j for j, row in enumerate(remaining) if groups[row] != groups[first]), None)
            if partner is not None:
                pairs.append((first, remaining.pop(partner)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _reward_features(beliefs, parameters, condition, reward_state=0):
    if condition not in ("reward_both", "reward_factor_1") or reward_state not in (0, 1, 2):
        raise ValueError("unknown Wing reward condition or state")
    kernels = controlled_kernels(**{key: parameters[key] for key in ("alpha", "x", "strength")})
    transition = kernels.sum(axis=1)
    values = np.einsum("nfs,as->nfa", beliefs, transition[:, :, reward_state])
    first = np.repeat(values[:, 0], 3, axis=1)
    second = np.tile(values[:, 1], (1, 3))
    return (first + second) / 2 if condition == "reward_both" else first


def analyze_control_samples(
    train: ControlData,
    test: ControlData,
    *,
    parameters: dict[str, Any],
    condition: str,
    seed: int = 42,
    n_resamples: int = 200,
    n_null_repeats: int = 5,
    shifted: tuple[ControlData, ControlData] | None = None,
) -> dict[str, Any]:
    if n_null_repeats < 1:
        raise ValueError("at least one null repetition is required")
    controlled = parameters["strength"] is not None
    train_b, train_p = replay_beliefs(train, **parameters)
    test_b, test_p = replay_beliefs(test, **parameters)
    alignment = max(float(np.max(np.abs(train_b - train.beliefs))), float(np.max(np.abs(test_b - test.beliefs))))
    if alignment > 1e-9:
        raise ValueError(f"replayed beliefs disagree with environment diagnostics: {alignment}")
    train_b, test_b = train_b[train.mask], test_b[test.mask]
    train_p, test_p = train_p[train.mask], test_p[test.mask]
    train_h, test_h = train.activations[train.mask], test.activations[test.mask]
    train_groups, test_groups = train.episode_ids[train.mask], test.episode_ids[test.mask]
    train_policy = train.policy_probabilities[train.mask]
    test_policy = test.policy_probabilities[test.mask]
    train_joint_p = (train_p[:, 0, :, None] * train_p[:, 1, None, :]).reshape(-1, 4)
    test_joint_p = (test_p[:, 0, :, None] * test_p[:, 1, None, :]).reshape(-1, 4)
    nuisance = {
        "joint_ntp": (train_joint_p, test_joint_p),
        "log_joint_ntp": (_log(train_joint_p), _log(test_joint_p)),
        "policy_probabilities": (train_policy, test_policy),
        "centered_log_policy": (
            _log(train_policy) - _log(train_policy).mean(axis=1, keepdims=True),
            _log(test_policy) - _log(test_policy).mean(axis=1, keepdims=True),
        ),
    }
    if controlled:
        reward_state = train.metadata.get("env_config", {}).get("task", {}).get("kwargs", {}).get("reward_state", 0)
        test_reward_state = test.metadata.get("env_config", {}).get("task", {}).get("kwargs", {}).get("reward_state", 0)
        if reward_state != test_reward_state:
            raise ValueError("train and test reward states must match")
        nuisance["expected_immediate_reward_all_actions"] = (
            _reward_features(train_b, parameters, condition, reward_state),
            _reward_features(test_b, parameters, condition, reward_state),
        )
    suffixes = {}
    keys = {}
    for length in SUFFIX_LENGTHS:
        suffixes[length] = (
            replay_beliefs(train, **parameters, suffix_length=length)[0][train.mask],
            replay_beliefs(test, **parameters, suffix_length=length)[0][test.mask],
        )
        if length in (1, 2, 4):
            keys[length] = tuple(
                suffix_keys(_symbols(data, controlled), data.episode_ids, data.episode_steps, length)[data.mask]
                for data in (train, test)
            )
    selected, adversarial = _select_alternatives(train, parameters, seed=seed)
    alternative_targets = {}
    for candidate in selected:
        alt_train, p_train = replay_beliefs(train, **candidate["parameters"])
        alt_test, p_test = replay_beliefs(test, **candidate["parameters"])
        alternative_targets[candidate["id"]] = (alt_train[train.mask], alt_test[test.mask])
        candidate["test_next_observation_kl"] = _mean_kl(test_p, p_test[test.mask])
        candidate["test_posterior_mse"] = float(np.mean((test_b - alt_test[test.mask]) ** 2))
    shifted_targets = None
    if shifted is not None:
        shifted_targets = tuple(replay_beliefs(data, **parameters)[0][data.mask] for data in shifted)
    report = {
        "schema_version": 2,
        "metadata": {
            "seed": seed,
            "condition": condition,
            "reward_state": reward_state if controlled else None,
            "parameters": parameters,
            "train_collection": dict(train.metadata),
            "test_collection": dict(test.metadata),
            "representation": "each_encoder_block_current_position_residual_before_final_layer_norm",
            "n_fit": len(train_b),
            "n_test": len(test_b),
            "train_episodes": int(len(np.unique(train_groups))),
            "test_episodes": int(len(np.unique(test_groups))),
            "fit_protocol": "training-only whole-episode SVD cross-validation; independent test rollouts",
            "uncertainty": "paired episode bootstrap with fixed fitted probes; not training-seed uncertainty",
            "n_bootstrap_resamples": n_resamples,
            "n_null_repeats": n_null_repeats,
            "sampling_distribution": "learned_stochastic_policy_process_weighted",
            "belief_conditioning": "visible tokens and preceding executed actions only; never rewards or hidden states",
            "belief_timing": "current decision-time arrival belief",
            "ntp_timing": "next emitted token" if controlled else "currently hidden token being guessed, from delayed source posterior",
            "rl_policy_is_not_ntp": True,
            "ntp_baselines": "ntp/log_ntp use the target factor marginal; joint_ntp/log_joint_ntp use the full four-token distribution",
            "all_action_ntp": "Wing controls rotate destinations after emission, so immediate token probabilities coincide across actions" if controlled else "actions do not affect the process",
            "suffix_lengths": list(SUFFIX_LENGTHS),
            "suffix_prior": "model reset prior; uniform, not an assumed policy-stationary prior",
            "log_floor": LOG_FLOOR,
            "null_interpretation": "descriptive controls, not exchangeability-based significance tests",
            "test_layer_selection": "none; every layer reported",
            "matched_pair_protocol": "disjoint held-out pairs in different episodes with identical visible suffix; no belief-based selection",
        },
        "replay_max_abs_error": alignment,
        "adversarial_search": adversarial,
        "factors": {},
        "scope_warning": "Linear accessibility is not causal use. Small bootstrap intervals do not establish model-seed generality; multiple layer/control comparisons are descriptive.",
    }
    for factor in range(2):
        name = f"factor_{factor + 1}"
        y_train, y_test = train_b[:, factor], test_b[:, factor]
        features = {
            "ntp": (train_p[:, factor], test_p[:, factor]),
            "log_ntp": (_log(train_p[:, factor]), _log(test_p[:, factor])),
            **nuisance,
            **{f"suffix_{length}_belief": (values[0][:, factor], values[1][:, factor]) for length, values in suffixes.items()},
        }
        baseline_predictions = {"train_mean": np.broadcast_to(y_train.mean(axis=0), y_test.shape)}
        baselines = {"train_mean": score_prediction(baseline_predictions["train_mean"], y_test)}
        for label, (x_train, x_test) in features.items():
            predicted, fit = _fit(x_train, y_train, x_test, train_groups, seed + factor)
            baseline_predictions[label] = predicted
            baselines[label] = {**score_prediction(predicted, y_test), "fit": fit}
        layers = {}
        for layer in range(train_h.shape[1]):
            x_train, x_test = train_h[:, layer], test_h[:, layer]
            predicted, fit = _fit(x_train, y_train, x_test, train_groups, seed + factor)
            layer_report = {
                "belief_probe": {**score_prediction(predicted, y_test), "fit": fit},
                "delta_contrast": score_prediction(
                    ((predicted[:, 0] - predicted[:, 2]) / np.sqrt(2))[:, None],
                    ((y_test[:, 0] - y_test[:, 2]) / np.sqrt(2))[:, None],
                ),
                "surplus_over_baselines": {
                    label: paired_comparison(predicted, baseline, y_test, test_groups, seed=seed, n_resamples=n_resamples)
                    for label, baseline in baseline_predictions.items()
                },
                "output_probes": {},
                "nulls": {},
                "same_suffix_pairs": {},
                "alternative_belief_probes": {},
            }
            for label in ("ntp", "log_ntp"):
                q_train, q_test = features[label]
                decoded, output_fit = _fit(x_train, q_train, x_test, train_groups, seed + factor)
                layer_report["output_probes"][label] = {**score_prediction(decoded, q_test), "fit": output_fit}
            for length, (_, test_keys) in keys.items():
                pairs = _matched_pairs(test_keys, test_groups)
                layer_report["same_suffix_pairs"][str(length)] = {
                    "n_pairs": len(pairs),
                    "belief_difference": score_prediction(
                        predicted[pairs[:, 0]] - predicted[pairs[:, 1]],
                        y_test[pairs[:, 0]] - y_test[pairs[:, 1]],
                    ) if len(pairs) else None,
                }
            for repeat in range(n_null_repeats):
                null_seed = seed + 1000 + repeat
                rng = np.random.default_rng(null_seed)
                shuffled, shuffled_fit = _fit(x_train, y_train[rng.permutation(len(y_train))], x_test, train_groups, seed + factor)
                null_results = {
                    "shuffled_training_labels": {**score_prediction(shuffled, y_test), "fit": shuffled_fit},
                }
                random_train = rng.dirichlet(np.ones(3), size=len(y_train))
                random_test = rng.dirichlet(np.ones(3), size=len(y_test))
                random_pred, random_fit = _fit(x_train, random_train, x_test, train_groups, seed + factor)
                null_results["dirichlet_labels"] = {**score_prediction(random_pred, random_test), "fit": random_fit}
                gaussian = gaussian_feature_null(x_train, len(x_test), seed=null_seed)
                matched = matched_feature_null(x_train, x_test, *keys[1], seed=null_seed)
                ntp_matched = matched_feature_null(
                    x_train, x_test,
                    np.floor(train_p[:, factor] / 0.05).astype(int),
                    np.floor(test_p[:, factor] / 0.05).astype(int),
                    seed=null_seed,
                )
                for label, (null_train, null_test, metadata) in (
                    ("gaussian_covariance_matched", gaussian),
                    ("visible_suffix_matched_activations", matched),
                    ("ntp_bin_matched_activations", ntp_matched),
                ):
                    null_pred, null_fit = _fit(null_train, y_train, null_test, train_groups, seed + factor)
                    null_results[label] = {**score_prediction(null_pred, y_test), "fit": null_fit, "sampling": metadata}
                layer_report["nulls"][str(repeat)] = null_results
            for candidate in selected:
                alt_train, alt_test = alternative_targets[candidate["id"]]
                alt_pred, alt_fit = _fit(x_train, alt_train[:, factor], x_test, train_groups, seed + factor)
                metrics = score_prediction(alt_pred, alt_test[:, factor])
                preference = paired_target_comparison(
                    predicted, y_test, alt_pred, alt_test[:, factor], test_groups,
                    seed=seed, n_resamples=n_resamples,
                )
                layer_report["alternative_belief_probes"][candidate["id"]] = {
                    **metrics, "fit": alt_fit,
                    "true_belief_r2_minus_alternative_r2": preference["r_squared_difference"],
                    "paired_target_comparison": preference,
                }
            if shifted_targets is not None:
                shifted_train, shifted_test = shifted
                sx_train = shifted_train.activations[shifted_train.mask, layer]
                sx_test = shifted_test.activations[shifted_test.mask, layer]
                sy_train, sy_test = shifted_targets[0][:, factor], shifted_targets[1][:, factor]
                transfer_pred, _ = _fit(x_train, y_train, sx_test, train_groups, seed + factor)
                refit_pred, refit_fit = _fit(sx_train, sy_train, sx_test, shifted_train.episode_ids[shifted_train.mask], seed + factor)
                layer_report["shuffled_history_stress"] = {
                    "frozen_original_probe": score_prediction(transfer_pred, sy_test),
                    "refitted_on_shuffled_histories": {**score_prediction(refit_pred, sy_test), "fit": refit_fit},
                    "interpretation": "offline observation/action-record reordering with activations and exact beliefs recomputed; distribution shift, not a null expected to score zero",
                }
            layers[f"layer_{layer + 1}"] = layer_report
        report["factors"][name] = {"baselines": baselines, "layers": layers}
    return report


def analyze_module_controls(module, *, env_config, condition, seed=42, n_steps=20_000, device=None, smoke=False):
    from experiments.wing_two_factor_explore_cycle_1.control_data import replay_activations, shuffled_history

    device = torch.device("cpu") if device is None else device
    parameters = dict(env_config["model"]["kwargs"]["factors"][0]["kwargs"])
    parameters["strength"] = env_config["task"].get("kwargs", {}).get("strength")
    seeds = np.random.SeedSequence(seed).spawn(4)
    train, test = (
        collect_control_data(module, env_config=env_config, n_steps=n_steps,
                             seed=int(stream.generate_state(1)[0]), device=device)
        for stream in seeds[:2]
    )
    shifted = []
    for data, stream in zip((train, test), seeds[2:]):
        shuffled = shuffled_history(data, seed=int(stream.generate_state(1)[0]))
        activations, policy = replay_activations(module, shuffled, device=device)
        shifted.append(replace(shuffled, activations=activations, policy_probabilities=policy))
    return analyze_control_samples(
        train, test, parameters=parameters, condition=condition, seed=seed,
        n_resamples=50 if smoke else 200, n_null_repeats=2 if smoke else 5,
        shifted=tuple(shifted),
    )


def summarize_report(report):
    factors = {}
    for name, factor in report["factors"].items():
        last_name = max(factor["layers"], key=lambda label: int(label.split("_")[-1]))
        last = factor["layers"][last_name]
        factors[name] = {
            "layer": last_name,
            "belief_r_squared": last["belief_probe"]["r_squared"],
            "delta_contrast_r_squared": last["delta_contrast"]["r_squared"],
            "baseline_r_squared": {key: value["r_squared"] for key, value in factor["baselines"].items()},
            "alternatives": {
                key: {
                    "r_squared": value["r_squared"],
                    "true_minus_alternative_r_squared": value["true_belief_r2_minus_alternative_r2"],
                    "paired_95_percent_interval": value["paired_target_comparison"]["r_squared_difference_ci"],
                }
                for key, value in last["alternative_belief_probes"].items()
            },
            "mean_null_r_squared": {
                key: float(np.mean([null[key]["r_squared"] for null in last["nulls"].values()]))
                for key in next(iter(last["nulls"].values()))
                if all(null[key]["r_squared"] is not None for null in last["nulls"].values())
            },
            "shuffled_history_r_squared": {
                key: last["shuffled_history_stress"][key]["r_squared"]
                for key in ("frozen_original_probe", "refitted_on_shuffled_histories")
            } if "shuffled_history_stress" in last else None,
        }
    return {
        "schema_version": report["schema_version"],
        "condition": report["metadata"]["condition"],
        "n_fit": report["metadata"]["n_fit"],
        "n_test": report["metadata"]["n_test"],
        "train_episodes": report["metadata"]["train_episodes"],
        "test_episodes": report["metadata"]["test_episodes"],
        "layer_selection": "last layer, not best test layer",
        "factors": factors,
        "scope_warning": report["scope_warning"],
    }


def _provenance(module_path):
    from analysis.probes import controls
    from experiments.wing_two_factor_explore_cycle_1 import control_data

    sources = [Path(__file__), Path(controls.__file__), Path(control_data.__file__)]
    repositories = [Path(__file__).resolve().parents[2], Path(controls.__file__).resolve().parents[2]]
    revisions = {}
    for root in repositories:
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        revisions[str(root)] = {"commit": revision, "dirty": dirty}
    return {
        "repositories": revisions,
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "module_checkpoint_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(module_path.iterdir()) if path.is_file()
        },
        "framework_versions": {"python": platform.python_version(), **{name: version(name) for name in ("torch", "numpy", "ray")}},
        "command": list(sys.argv),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Non-interventional Wing belief-probe controls on an existing checkpoint")
    parser.add_argument("--study", choices=("token_guess", "explore_cycle_2"), required=True)
    parser.add_argument("--condition", choices=("reward_both", "reward_factor_1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    if args.output.exists() or summary_path.exists():
        parser.error("output or summary already exists; choose a new result path")
    if args.study == "token_guess":
        if args.condition is not None:
            parser.error("--condition is only valid for explore_cycle_2")
        from experiments.wing_token_guess_cycle_1.process import environment_config
        config, condition = environment_config(), "token_guess"
    else:
        if args.condition is None:
            parser.error("explore_cycle_2 requires --condition")
        from experiments.wing_two_factor_explore_cycle_2.process import environment_config
        config, condition = environment_config(args.condition), args.condition
    from ray.rllib.core.rl_module.rl_module import RLModule

    checkpoint = args.checkpoint.resolve()
    nested = checkpoint / "learner_group" / "learner" / "rl_module" / "default_policy"
    module_path = nested if nested.is_dir() else checkpoint
    provenance = _provenance(module_path)
    torch.set_num_threads(1)
    module = RLModule.from_checkpoint(str(module_path))
    report = analyze_module_controls(
        module, env_config=config, condition=condition, seed=args.seed,
        n_steps=256 if args.smoke else args.steps, smoke=args.smoke,
    )
    report["metadata"].update({"checkpoint": str(checkpoint), "environment_config": config, "provenance": provenance})
    summary = summarize_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((args.output, report), (summary_path, summary)):
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(f"Wrote {args.output} and {summary_path}")
    for name, values in summary["factors"].items():
        print(f"{name}: last-layer belief R2={values['belief_r_squared']}; NTP={values['baseline_r_squared']['ntp']}; log NTP={values['baseline_r_squared']['log_ntp']}")


if __name__ == "__main__":
    main()
