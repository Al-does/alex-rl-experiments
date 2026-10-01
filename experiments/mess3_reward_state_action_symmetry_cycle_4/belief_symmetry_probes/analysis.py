"""Analysis-only scalar belief probes shared by cycles 4 and 5."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from analysis.checkpoints import load_algorithm
from analysis.probes import (
    cluster_bootstrap_statistics,
    conditional_mse_metrics,
    fit_affine_probe,
    global_mse_metrics,
    held_out_permutation_null,
    mean_squared_error,
    percentile_interval,
    predictive_belief_update,
    probe_predict,
    r2_score,
)
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    branch_keys,
    collect_probe_data,
    make_transducer_target,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

RIDGE = 1e-6
N_ENVS = 16
EPISODE_LENGTH = 1024
MIN_GROUP_SIZE = 50
PERMUTATION_SAMPLE_CAP = 4096
STREAMS = {
    "train": (510,),
    "test": (511,),
    "bootstrap_symmetric_b2": (520,),
    "bootstrap_antisymmetric_b0_minus_b1": (521,),
    "bootstrap_coarse_b2": (522,),
    "bootstrap_full_belief": (523,),
    "permutation_symmetric_b2": (530,),
    "permutation_antisymmetric_b0_minus_b1": (531,),
    "permutation_coarse_b2": (532,),
    "permutation_full_belief": (533,),
    "permutation_sample_symmetric_b2": (540,),
    "permutation_sample_antisymmetric_b0_minus_b1": (541,),
    "permutation_sample_coarse_b2": (542,),
    "permutation_sample_full_belief": (543,),
}
TARGET_NAMES = (
    "symmetric_b2",
    "antisymmetric_b0_minus_b1",
    "coarse_b2",
    "full_belief",
)


def decompose_belief(beliefs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the symmetric and antisymmetric scalar coordinates."""
    values = np.asarray(beliefs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("beliefs must have shape (N, 3)")
    return values[:, 2:3], values[:, 0:1] - values[:, 1:2]


def reconstruct_belief(
    symmetric_b2: np.ndarray,
    antisymmetric_b0_minus_b1: np.ndarray,
) -> np.ndarray:
    """Invert the two-coordinate decomposition on the probability simplex."""
    b2 = np.asarray(symmetric_b2, dtype=np.float64).reshape(-1, 1)
    difference = np.asarray(
        antisymmetric_b0_minus_b1, dtype=np.float64
    ).reshape(-1, 1)
    if len(b2) != len(difference):
        raise ValueError("coordinate arrays must have equal length")
    remaining = 1.0 - b2
    return np.concatenate(
        ((remaining + difference) / 2.0, (remaining - difference) / 2.0, b2),
        axis=1,
    )


def _device(context: RunContext) -> str:
    profile = context.hardware or PROFILES["cpu"]
    return "cuda" if profile.learner_device == "cuda" and torch.cuda.is_available() else "cpu"


def _install_checkpoint_import_aliases(cycle: int) -> None:
    """Keep checkpoints loadable after the cycle-5 package rename."""
    if cycle != 5:
        return
    old = "experiments.mess3_reward_state_action_asymmetry_cycle_5"
    new = "experiments.mess3_reward_state_action_symmetry_cycle_5"
    sys.modules.setdefault(old, importlib.import_module(new))
    sys.modules.setdefault(
        f"{old}.task",
        importlib.import_module(f"{new}.task"),
    )


def _episode_clusters(data: ProbeData) -> np.ndarray:
    clusters = np.empty(len(data.episode_steps), dtype=np.int64)
    next_cluster = 0
    for env_index in np.unique(data.env_indices):
        members = np.flatnonzero(data.env_indices == env_index)
        current = next_cluster
        for offset, index in enumerate(members):
            if offset and data.episode_steps[index] == 0:
                current += 1
            clusters[index] = current
        next_cluster = current + 1
    return clusters


def _coarse_spec(environment: Any) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Construct and validate the A={0,1}, B={2} lumped HMM."""
    emission = np.asarray(environment.model.emission_matrix, dtype=np.float64)
    coarse_emission = np.column_stack((emission[:, :2].sum(axis=1), emission[:, 2]))
    if not np.allclose(coarse_emission[0], coarse_emission[1], atol=1e-12):
        raise ValueError("states 0 and 1 do not have equal coarsened emissions")
    lumped_emission = np.stack((coarse_emission[0], coarse_emission[2]))
    expected = np.asarray([[0.925, 0.075], [0.15, 0.85]], dtype=np.float64)
    if not np.allclose(lumped_emission, expected, atol=1e-12):
        raise ValueError(f"unexpected coarse emission: {lumped_emission}")

    transitions: dict[int, np.ndarray] = {}
    for action in range(environment.action_space.n):
        full = np.asarray(
            environment.task.transition_matrix_for_action(action), dtype=np.float64
        )
        destination_lumps = np.column_stack((full[:, :2].sum(axis=1), full[:, 2]))
        if not np.allclose(destination_lumps[0], destination_lumps[1], atol=1e-12):
            raise ValueError(f"action {action} is not strongly lumpable over states 0/1")
        # The A row is one source row's destination-lump probabilities. Never
        # sum source rows; that would produce a non-stochastic transition.
        lumped = np.stack((destination_lumps[0], destination_lumps[2]))
        if not np.allclose(lumped.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError(f"invalid lumped transition for action {action}")
        transitions[action] = lumped
    initial = np.asarray(environment.model.initial_distribution, dtype=np.float64)
    coarse_initial = np.asarray([initial[:2].sum(), initial[2]], dtype=np.float64)
    return coarse_initial, lumped_emission, transitions


def _coarse_targets(
    data: ProbeData,
    *,
    initial: np.ndarray,
    emission: np.ndarray,
    transitions: dict[int, np.ndarray],
) -> np.ndarray:
    beliefs = np.repeat(initial[None, :], N_ENVS, axis=0)
    result = np.empty((len(data.episode_steps), 1), dtype=np.float64)
    actions = np.asarray(data.actions, dtype=np.int64).reshape(-1)
    previous_actions = np.zeros(N_ENVS, dtype=np.int64)
    for index, (env_index, step, token, action) in enumerate(
        zip(data.env_indices, data.episode_steps, data.tokens, actions, strict=True)
    ):
        env_index = int(env_index)
        coarse_token = 1 if int(token) == 2 else 0
        measurement = np.diag(emission[:, coarse_token])
        if int(step) == 0:
            beliefs[env_index] = predictive_belief_update(initial, measurement)
        else:
            # ProbeData.actions[index] is selected from the current observation;
            # the transition represented by that observation was caused by the
            # preceding retained action for this environment.
            beliefs[env_index] = predictive_belief_update(
                beliefs[env_index],
                transitions[int(previous_actions[env_index])] @ measurement,
            )
        result[index, 0] = beliefs[env_index, 1]
        previous_actions[env_index] = int(action)
    return result


def _subset(data: ProbeData, indices: np.ndarray) -> ProbeData:
    kwargs: dict[str, Any] = {}
    for field in ProbeData.__dataclass_fields__:
        value = getattr(data, field)
        kwargs[field] = value[indices] if value is not None else None
    return ProbeData(**kwargs)


def _collect_with_history(
    *,
    n_steps: int,
    warmup: int,
    seed: np.random.SeedSequence,
    common: dict[str, Any],
    coarse_spec: tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]] | None,
) -> tuple[ProbeData, np.ndarray | None]:
    # Keep episode prefixes long enough to advance the separate coarse filter,
    # then apply exactly the standard per-episode warmup to scored rows.
    episode_payload = max(1, EPISODE_LENGTH - warmup)
    budget = (
        n_steps
        + int(np.ceil(n_steps / episode_payload)) * warmup
        # The first episode length is randomized independently in each
        # environment. A full vector-episode margin covers every possible
        # truncated prefix while preserving the first n process-weighted rows.
        + N_ENVS * EPISODE_LENGTH
    )
    raw = collect_probe_data(
        n_steps=budget, seed=seed, warmup=0, **common
    )
    eligible = np.flatnonzero(raw.episode_steps >= warmup)
    if len(eligible) < n_steps:
        raise RuntimeError("history collection budget did not yield enough scored rows")
    indices = eligible[:n_steps]
    coarse = None
    if coarse_spec is not None:
        initial, emission, transitions = coarse_spec
        coarse = _coarse_targets(
            raw, initial=initial, emission=emission, transitions=transitions
        )[indices]
    return _subset(raw, indices), coarse


def _permutation_metrics(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    *,
    sample_seed: int,
    permutation_seed: int,
    n_permutations: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(sample_seed)
    train_idx = rng.choice(
        len(train_targets), min(PERMUTATION_SAMPLE_CAP, len(train_targets)), replace=False
    )
    test_idx = rng.choice(
        len(test_targets), min(PERMUTATION_SAMPLE_CAP, len(test_targets)), replace=False
    )
    x_train, y_train = train_features[train_idx], train_targets[train_idx]
    x_test, y_test = test_features[test_idx], test_targets[test_idx]

    def fit_predict(targets: np.ndarray) -> np.ndarray:
        weight, bias = fit_affine_probe(x_train, targets, ridge=RIDGE)
        return probe_predict(weight, bias, x_test)

    real = mean_squared_error(fit_predict(y_train), y_test)
    null = held_out_permutation_null(
        y_train, fit_predict, y_test, n_permutations=n_permutations, seed=permutation_seed
    )
    p05, p50, p95 = np.quantile(null, [0.05, 0.5, 0.95])
    return {
        "permutation_real_mse": real,
        "permutation_null_mse_p05": float(p05),
        "permutation_null_mse_p50": float(p50),
        "permutation_null_mse_p95": float(p95),
        "permutation_null_p_value_lower_tail": float(
            (1 + np.count_nonzero(null <= real)) / (len(null) + 1)
        ),
        "permutation_null_n": int(n_permutations),
    }


def _evaluate_target(
    name: str,
    train: ProbeData,
    test: ProbeData,
    train_target: np.ndarray,
    test_target: np.ndarray,
    streams: dict[str, np.random.SeedSequence],
    *,
    n_resamples: int,
) -> dict[str, Any]:
    train_target = np.asarray(train_target, dtype=np.float64)
    test_target = np.asarray(test_target, dtype=np.float64)
    if name == "full_belief":
        if train_target.ndim != 2 or train_target.shape[1] != 3:
            raise ValueError("full_belief requires belief targets with shape (N, 3)")
    else:
        train_target = train_target.reshape(-1, 1)
        test_target = test_target.reshape(-1, 1)
    variance = float(np.square(test_target - test_target.mean(axis=0)).mean())
    if variance <= np.finfo(np.float64).eps:
        return {"status": "degenerate", "target_variance": variance}
    weight, bias = fit_affine_probe(train.activations, train_target, ridge=RIDGE)
    predicted = probe_predict(weight, bias, test.activations)
    bootstrap = cluster_bootstrap_statistics(
        _episode_clusters(test),
        lambda indices: mean_squared_error(predicted[indices], test_target[indices]),
        n_resamples=n_resamples,
        seed=seed_sequence_to_int(streams[f"bootstrap_{name}"], bits=32),
    )
    ci_low, ci_high = percentile_interval(bootstrap)
    return {
        "status": "fitted",
        "definition": {
            "symmetric_b2": "exact full-filter P(state=2)",
            "antisymmetric_b0_minus_b1": "exact full-filter P(state=0)-P(state=1)",
            "coarse_b2": "separate exact lumped-filter P(B={state2})",
            "full_belief": "exact full-filter 3-state belief vector",
        }[name],
        **global_mse_metrics(predicted, test_target),
        "r_squared": r2_score(predicted, test_target),
        "R2": r2_score(predicted, test_target),
        **conditional_mse_metrics(
            predicted,
            test_target,
            branch_keys(test, depth=2),
            min_group_size=MIN_GROUP_SIZE,
        ),
        "mse_ci_95_low": ci_low,
        "mse_ci_95_high": ci_high,
        "bootstrap_n": n_resamples,
        "bootstrap_cluster": "environment_episode",
        **_permutation_metrics(
            train.activations,
            train_target,
            test.activations,
            test_target,
            sample_seed=seed_sequence_to_int(
                streams[f"permutation_sample_{name}"], bits=32
            ),
            permutation_seed=seed_sequence_to_int(
                streams[f"permutation_{name}"], bits=32
            ),
            n_permutations=n_resamples,
        ),
    }


def probe_checkpoint(
    context: RunContext,
    checkpoint: Path,
    *,
    cycle: int,
    variant: int,
    label: str,
    target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("belief symmetry probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, STREAMS)
    steps = 4096 if context.smoke else None
    train_steps, test_steps = (steps, steps) if steps else (60_000, 80_000)
    warmup = 4 if context.smoke else 64
    n_resamples = 100 if context.smoke else 1000

    requested_targets = (
        tuple(target_names)
        if target_names is not None
        else TARGET_NAMES[:2] + (("coarse_b2",) if variant in (1, 2) else ())
    )
    unknown_targets = set(requested_targets) - set(TARGET_NAMES)
    if unknown_targets:
        raise ValueError(f"unknown probe targets: {sorted(unknown_targets)}")
    if "coarse_b2" in requested_targets and variant not in (1, 2):
        raise ValueError("coarse_b2 is defined only for variants 1 and 2")

    _install_checkpoint_import_aliases(cycle)
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        env_config["diagnostics"] = {
            "state": True, "belief": True, "tokens": True, "transitions": True
        }

        def make_environment():
            return env_class(env_config)

        environment = make_environment()
        try:
            initial, action_outcome, initial_outcome = make_transducer_target(environment)
            coarse_spec = (
                _coarse_spec(environment)
                if "coarse_b2" in requested_targets
                else None
            )
        finally:
            environment.close()
        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": _device(context),
            "n_envs": N_ENVS,
            "initial_belief": initial,
            "action_outcome_operator": action_outcome,
            "initial_outcome_operator": initial_outcome,
        }
        train, train_coarse = _collect_with_history(
            n_steps=train_steps,
            warmup=warmup,
            seed=streams["train"],
            common=common,
            coarse_spec=coarse_spec,
        )
        test, test_coarse = _collect_with_history(
            n_steps=test_steps,
            warmup=warmup,
            seed=streams["test"],
            common=common,
            coarse_spec=coarse_spec,
        )

    consistency = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in (train, test)
    )
    if consistency > 1e-10:
        raise AssertionError(f"full target disagrees with diagnostic belief: {consistency:.3e}")
    train_symmetric, train_antisymmetric = decompose_belief(train.beliefs)
    test_symmetric, test_antisymmetric = decompose_belief(test.beliefs)
    targets = {}
    if "symmetric_b2" in requested_targets:
        targets["symmetric_b2"] = (train_symmetric, test_symmetric)
    if "antisymmetric_b0_minus_b1" in requested_targets:
        targets["antisymmetric_b0_minus_b1"] = (
            train_antisymmetric,
            test_antisymmetric,
        )
    if "coarse_b2" in requested_targets:
        assert train_coarse is not None and test_coarse is not None
        coarse_projection_differences = np.concatenate(
            (
                train_coarse[:, 0] - train_symmetric[:, 0],
                test_coarse[:, 0] - test_symmetric[:, 0],
            )
        )
        coarse_projection_mse = float(
            np.mean(np.square(coarse_projection_differences))
        )
        coarse_projection_max_abs_difference = float(
            np.max(np.abs(coarse_projection_differences))
        )
        targets["coarse_b2"] = (train_coarse, test_coarse)
    else:
        coarse_projection_mse = None
        coarse_projection_max_abs_difference = None
    if "full_belief" in requested_targets:
        targets["full_belief"] = (train.beliefs, test.beliefs)

    return {
        "checkpoint": label,
        "protocol": {
            "representation": "post_final_layer_norm",
            "sampling_distribution": "greedy_process_weighted_rollout",
            "train_steps": train_steps,
            "test_steps": test_steps,
            "n_envs": N_ENVS,
            "warmup": warmup,
            "ridge": RIDGE,
            "branch_depth": 2,
            "target_dtype": "float64",
            "seed_streams": {name: list(key) for name, key in STREAMS.items()},
        },
        "target_consistency_max_abs": consistency,
        "coarse_projection_mse": coarse_projection_mse,
        "coarse_projection_max_abs_difference": (
            coarse_projection_max_abs_difference
        ),
        "targets": {
            name: _evaluate_target(
                name, train, test, train_target, test_target, streams,
                n_resamples=n_resamples,
            )
            for name, (train_target, test_target) in targets.items()
        },
    }


def _source_provenance(bundle: Path) -> dict[str, Any]:
    path = bundle / "source_provenance.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"source_run_id": bundle.name, "bundle": str(bundle.resolve())}


def _checkpoint_requests(bundle: Path) -> list[dict[str, Any]]:
    manifest_path = bundle / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return [
            {
                "label": "initial",
                "training_iteration": 0,
                "path": "initial_checkpoint",
            },
            {
                "label": "final",
                "training_iteration": None,
                "path": "final_checkpoint",
            },
        ]
    payload = json.loads(manifest_path.read_text())
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"{manifest_path} contains no checkpoints")
    return checkpoints


def _training_probe_curve_path(*, cycle: int, variant: int, seed: int) -> Path:
    package = f"mess3_reward_state_action_symmetry_cycle_{cycle}"
    return (
        Path(__file__).resolve().parents[1].parent
        / package
        / f"variant_{variant}"
        / "results"
        / f"mess3-rsa-c{cycle}-v{variant}-seed{seed}"
        / "checkpoint_probe_curve.json"
    )


def _training_full_belief_mse_by_label(
    *,
    cycle: int,
    variant: int,
    seed: int,
) -> dict[str, float]:
    """Reuse log-spaced training full-belief probes instead of re-running them."""

    curve_path = _training_probe_curve_path(
        cycle=cycle,
        variant=variant,
        seed=seed,
    )
    if not curve_path.is_file():
        return {}
    payload = json.loads(curve_path.read_text())
    lookup: dict[str, float] = {}
    for point in payload.get("checkpoints", []):
        step = int(point["agent_steps"])
        label = "initial" if step == 0 else point.get("checkpoint_name")
        if not isinstance(label, str):
            continue
        probe = point.get("probe", {})
        if probe.get("target") != "exact_predictive_bayesian_belief":
            continue
        lookup[label] = float(point["mse"])
    return lookup


def _imported_full_belief_checkpoint(
    *,
    label: str,
    training_iteration: int | None,
    mse: float,
) -> dict[str, Any]:
    return {
        "checkpoint": label,
        "training_iteration": training_iteration,
        "imported_from": "training_checkpoint_probe_curve",
        "targets": {
            "full_belief": {
                "status": "imported",
                "definition": "exact full-filter 3-state belief vector",
                "mse": mse,
                "source_probe_target": "exact_predictive_bayesian_belief",
            }
        },
    }


def run_probe_condition(context: RunContext, *, cycle: int, variant: int) -> dict[str, Any]:
    bundle = context.resume_from
    if bundle is None:
        raise ValueError("resume_from must name a bundle containing initial_checkpoint/ and final_checkpoint/")
    bundle = Path(bundle)
    source = _source_provenance(bundle)
    requested_target = source.get("requested_target")
    target_names = (requested_target,) if requested_target else None
    checkpoint_requests = _checkpoint_requests(bundle)
    for request in checkpoint_requests:
        checkpoint = bundle / request["path"]
        if not checkpoint.is_dir() or not any(checkpoint.rglob("*")):
            raise FileNotFoundError(f"missing checkpoint bundle member: {checkpoint}")
    imported_full_belief = (
        _training_full_belief_mse_by_label(
            cycle=cycle,
            variant=variant,
            seed=context.seed,
        )
        if requested_target == "full_belief"
        else {}
    )
    context.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "study": "belief_symmetry_probes",
        "cycle": cycle,
        "variant": variant,
        "seed": context.seed,
        "source": source,
        "target_definitions": {
            "symmetric_b2": "b2",
            "antisymmetric_b0_minus_b1": "b0-b1",
            **({"coarse_b2": "separate A={0,1}, B={2} lumped Bayes filter"} if variant in (1, 2) else {}),
            **({"full_belief": "exact 3-state predictive Bayes filter belief"}),
        },
        "filter_definitions": {
            "full": "delay-0 initial measurement; later transition@measurement using action-dependent transitions",
            **({"coarse": "tokens 0/1 coarsened to not-2; destination-lump rows, never summed source rows"} if variant in (1, 2) else {}),
        },
        "random_weight_baseline_interpretation": (
            "The restored initial checkpoint estimates the affine random-network floor; "
            "it is a baseline, not evidence that an untrained network computes belief."
        ),
        "requested_target": requested_target,
        "checkpoint_schedule": [
            {
                "label": request["label"],
                "training_iteration": request.get("training_iteration"),
            }
            for request in checkpoint_requests
        ],
        "checkpoints": {},
    }
    for request in checkpoint_requests:
        checkpoint = bundle / request["path"]
        label = request["label"]
        if label in imported_full_belief:
            result = _imported_full_belief_checkpoint(
                label=label,
                training_iteration=request.get("training_iteration"),
                mse=imported_full_belief[label],
            )
        else:
            result = probe_checkpoint(
                replace(context, resume_from=checkpoint),
                checkpoint,
                cycle=cycle,
                variant=variant,
                label=label,
                target_names=target_names,
            )
            result["training_iteration"] = request.get("training_iteration")
        summary["checkpoints"][label] = result
        # Persist after every checkpoint so a preempted remote job retains
        # compact progress and can be diagnosed without checkpoint artifacts.
        (context.results_dir / "condition_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    (context.results_dir / "condition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
