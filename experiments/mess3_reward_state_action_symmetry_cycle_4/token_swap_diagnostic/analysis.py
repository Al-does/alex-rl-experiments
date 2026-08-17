"""Paired token-swap intervention for cycle 4/5 variant-2 belief probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.checkpoints import load_algorithm
from analysis.probes import (
    cluster_bootstrap_statistics,
    fit_affine_probe,
    global_mse_metrics,
    mean_squared_error,
    percentile_interval,
    probe_predict,
)
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.analysis import (
    _install_checkpoint_import_aliases,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

PROBE_RIDGE = 1e-6
N_ENVS = 16
FULL_TRAIN_STEPS = 60_000
FULL_TEST_STEPS = 80_000
SMOKE_STEPS = 4_096
FULL_RESAMPLES = 1_000
SMOKE_RESAMPLES = 100
STATE_TOKEN_PERMUTATION = np.asarray([1, 0, 2], dtype=np.int64)
_STREAM_KEYS = {
    "probe_train": (610,),
    "probe_test": (611,),
    "bootstrap": (612,),
}


def swap_state_0_1_tokens(observations: np.ndarray) -> np.ndarray:
    """Exchange the state-0/state-1 token channels without mutating the input."""

    values = np.asarray(observations)
    if values.ndim < 1 or values.shape[-1] < 3:
        raise ValueError("observations must end with the three token channels")
    swapped = np.array(values, copy=True)
    swapped[..., 0] = values[..., 1]
    swapped[..., 1] = values[..., 0]
    return swapped


def _device(context: RunContext) -> str:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_checkpoint(source: Path) -> Path:
    """Accept either a direct Algorithm checkpoint or a probe source bundle."""

    source = Path(source)
    bundled = source / "final_checkpoint"
    checkpoint = bundled if bundled.is_dir() else source
    if not checkpoint.is_dir() or not any(checkpoint.rglob("*")):
        raise FileNotFoundError(f"checkpoint is empty or missing: {checkpoint}")
    return checkpoint


def _validate_intervention_environment(environment: Any, *, cycle: int) -> None:
    """Require the exact state/token symmetry that defines the counterfactual."""

    if cycle not in (4, 5):
        raise ValueError("token-swap diagnostics support only cycles 4 and 5")
    if getattr(environment.task, "variant", None) != 2:
        raise ValueError("token-swap diagnostics require action-symmetry variant 2")

    observation = environment.config.observation
    if (
        observation.token is None
        or observation.token.offset != 0
        or observation.token.depth != 1
        or observation.action is None
        or observation.action.offset != 0
        or observation.action.depth != 1
        or observation.belief
        or observation.hidden_state
        or environment.observation_space.shape != (6,)
    ):
        raise ValueError(
            "diagnostic requires one current three-way token and one previous "
            "three-way action block"
        )

    permutation = STATE_TOKEN_PERMUTATION
    initial = np.asarray(environment.model.initial_distribution, dtype=np.float64)
    emission = np.asarray(environment.model.emission_matrix, dtype=np.float64)
    if not np.allclose(initial[permutation], initial, atol=1e-12):
        raise ValueError("initial state distribution is not invariant to state 0/1")
    if not np.allclose(
        emission[np.ix_(permutation, permutation)],
        emission,
        atol=1e-12,
    ):
        raise ValueError("emissions are not equivariant to the state/token swap")
    for action in range(environment.action_space.n):
        transition = np.asarray(
            environment.task.transition_matrix_for_action(action),
            dtype=np.float64,
        )
        if not np.allclose(
            transition[np.ix_(permutation, permutation)],
            transition,
            atol=1e-12,
        ):
            raise ValueError(
                f"action {action} transition is not equivariant to state 0/1"
            )


def _contiguous_segments(data: ProbeData) -> list[np.ndarray]:
    """Group retained rows into contiguous within-environment episode suffixes."""

    segments: list[np.ndarray] = []
    for env_index in np.unique(data.env_indices):
        members = np.flatnonzero(data.env_indices == env_index)
        start = 0
        for offset in range(1, len(members)):
            previous, current = members[offset - 1 : offset + 1]
            if data.episode_steps[current] != data.episode_steps[previous] + 1:
                segments.append(members[start:offset])
                start = offset
        if len(members):
            segments.append(members[start:])
    return [segment for segment in segments if len(segment)]


def _reconstruct_observations(
    data: ProbeData,
    segments: list[np.ndarray],
) -> np.ndarray:
    """Rebuild the six policy-visible features from public rollout records."""

    observations = np.zeros((len(data.tokens), 6), dtype=np.float32)
    valid_tokens = data.tokens >= 0
    token_rows = np.flatnonzero(valid_tokens)
    observations[token_rows, data.tokens[valid_tokens]] = 1.0
    actions = np.asarray(data.actions, dtype=np.int64).reshape(-1)
    for segment in segments:
        if len(segment) < 2:
            continue
        current = segment[1:]
        previous_actions = actions[segment[:-1]]
        observations[current, 3 + previous_actions] = 1.0
    return observations


def _initial_state(module: Any, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * np.asarray(value).ndim))
        for key, value in module.get_initial_state().items()
    }


@torch.no_grad()
def paired_token_swap_activations(
    module: Any,
    data: ProbeData,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Encode factual/swapped versions of each fixed token-action history."""

    if not module.is_stateful() or not hasattr(module, "sequence_lookback"):
        raise TypeError("token-swap diagnostic requires a stateful transformer")
    segments = _contiguous_segments(data)
    observations = _reconstruct_observations(data, segments)
    max_length = max(map(len, segments))
    padded = np.zeros(
        (len(segments), max_length, observations.shape[1]),
        dtype=np.float32,
    )
    lengths = np.asarray([len(segment) for segment in segments], dtype=np.int64)
    for index, segment in enumerate(segments):
        padded[index, : len(segment)] = observations[segment]
    paired = np.concatenate((padded, swap_state_0_1_tokens(padded)), axis=0)

    torch_device = torch.device(device)
    module = module.to(torch_device).eval()
    state = _initial_state(module, len(paired), torch_device)
    factual = np.empty_like(data.activations, dtype=np.float64)
    swapped = np.empty_like(data.activations, dtype=np.float64)
    eligible: list[int] = []
    lookback = int(module.sequence_lookback)

    for step in range(max_length):
        inputs = torch.from_numpy(paired[:, step]).to(torch_device)
        embedding, state = module.encode_step(inputs, state)
        encoded = embedding.detach().cpu().numpy()
        active_segments = np.flatnonzero(lengths > step)
        for segment_index in active_segments:
            row = int(segments[segment_index][step])
            factual[row] = encoded[segment_index]
            swapped[row] = encoded[len(segments) + segment_index]
            # Earlier retained rows begin with an incomplete hidden context.
            if step >= lookback:
                eligible.append(row)

    indices = np.asarray(sorted(eligible), dtype=np.int64)
    if not len(indices):
        raise RuntimeError(
            f"no histories exceeded the transformer's {lookback}-step lookback"
        )
    reconstruction_error = float(
        np.max(np.abs(factual[indices] - data.activations[indices]))
    )
    if reconstruction_error > 2e-5:
        raise AssertionError(
            "reconstructed factual histories do not match rollout activations: "
            f"{reconstruction_error:.3e}"
        )
    return factual[indices], swapped[indices], indices, reconstruction_error


def _episode_clusters(data: ProbeData, indices: np.ndarray) -> np.ndarray:
    clusters = np.empty(len(indices), dtype=np.int64)
    next_cluster = 0
    previous_by_env: dict[int, tuple[int, int]] = {}
    for output_index, source_index in enumerate(indices):
        env_index = int(data.env_indices[source_index])
        episode_step = int(data.episode_steps[source_index])
        previous = previous_by_env.get(env_index)
        if previous is None or episode_step != previous[0] + 1:
            cluster = next_cluster
            next_cluster += 1
        else:
            cluster = previous[1]
        clusters[output_index] = cluster
        previous_by_env[env_index] = (episode_step, cluster)
    return clusters


def _shift_metrics(
    decoded_shift: np.ndarray,
    exact_shift: np.ndarray,
) -> dict[str, float | int]:
    decoded_norm = np.linalg.norm(decoded_shift, axis=1)
    exact_norm = np.linalg.norm(exact_shift, axis=1)
    valid = (decoded_norm > 1e-12) & (exact_norm > 1e-12)
    cosine = np.sum(decoded_shift[valid] * exact_shift[valid], axis=1) / (
        decoded_norm[valid] * exact_norm[valid]
    )
    return {
        "shift_mse": mean_squared_error(decoded_shift, exact_shift),
        "shift_cosine_mean": float(cosine.mean()) if len(cosine) else float("nan"),
        "shift_norm_ratio_mean": (
            float(np.mean(decoded_norm[valid] / exact_norm[valid]))
            if len(cosine)
            else float("nan")
        ),
        "decoded_shift_rms": float(np.sqrt(np.mean(np.square(decoded_shift)))),
        "exact_shift_rms": float(np.sqrt(np.mean(np.square(exact_shift)))),
        "n_nonzero": int(np.count_nonzero(valid)),
    }


def evaluate_token_swap(
    *,
    factual_activations: np.ndarray,
    swapped_activations: np.ndarray,
    factual_targets: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    """Score a frozen factual decoder under the paired token intervention."""

    factual_targets = np.asarray(factual_targets, dtype=np.float64)
    counterfactual_targets = factual_targets[:, STATE_TOKEN_PERMUTATION]
    factual_predictions = probe_predict(weight, bias, factual_activations)
    counterfactual_predictions = probe_predict(weight, bias, swapped_activations)
    equivariant_predictions = factual_predictions[:, STATE_TOKEN_PERMUTATION]
    factual_metrics = global_mse_metrics(factual_predictions, factual_targets)
    counterfactual_metrics = global_mse_metrics(
        counterfactual_predictions,
        counterfactual_targets,
    )
    decoded_shift = counterfactual_predictions - factual_predictions
    exact_shift = counterfactual_targets - factual_targets
    factual_antisymmetric = factual_predictions[:, 0] - factual_predictions[:, 1]
    swapped_antisymmetric = (
        counterfactual_predictions[:, 0] - counterfactual_predictions[:, 1]
    )
    return {
        "factual": factual_metrics,
        "counterfactual": counterfactual_metrics,
        "counterfactual_minus_factual_mse": float(
            counterfactual_metrics["mse"] - factual_metrics["mse"]
        ),
        "counterfactual_over_factual_mse": float(
            counterfactual_metrics["mse"] / factual_metrics["mse"]
        )
        if factual_metrics["mse"] > 0.0
        else float("nan"),
        "equivariance_mse": mean_squared_error(
            counterfactual_predictions,
            equivariant_predictions,
        ),
        "state_2_invariance_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        counterfactual_predictions[:, 2]
                        - factual_predictions[:, 2]
                    )
                )
            )
        ),
        "antisymmetric_sign_reversal_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(swapped_antisymmetric + factual_antisymmetric)
                )
            )
        ),
        "activation_shift_rms": float(
            np.sqrt(np.mean(np.square(swapped_activations - factual_activations)))
        ),
        **_shift_metrics(decoded_shift, exact_shift),
        "_factual_predictions": factual_predictions,
        "_counterfactual_predictions": counterfactual_predictions,
        "_counterfactual_targets": counterfactual_targets,
    }


def probe_checkpoint(
    context: RunContext,
    checkpoint: Path,
    *,
    cycle: int,
) -> dict[str, Any]:
    """Fit on factual rollouts, then swap tokens in fixed held-out histories."""

    if context.seed is None:
        raise ValueError("token-swap diagnostic requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = SMOKE_STEPS if context.smoke else FULL_TRAIN_STEPS
    test_steps = SMOKE_STEPS if context.smoke else FULL_TEST_STEPS
    warmup = 4 if context.smoke else 64
    n_resamples = SMOKE_RESAMPLES if context.smoke else FULL_RESAMPLES

    _install_checkpoint_import_aliases(cycle)
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        env_config["diagnostics"] = {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        }

        def make_environment():
            return env_class(env_config)

        environment = make_environment()
        try:
            _validate_intervention_environment(environment, cycle=cycle)
            transducer_target = make_transducer_target(environment)
        finally:
            environment.close()
        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": _device(context),
            "warmup": warmup,
            "n_envs": N_ENVS,
            "initial_belief": transducer_target[0],
            "action_outcome_operator": transducer_target[1],
            "initial_outcome_operator": transducer_target[2],
        }
        train = collect_probe_data(
            n_steps=train_steps,
            seed=streams["probe_train"],
            **common,
        )
        test = collect_probe_data(
            n_steps=test_steps,
            seed=streams["probe_test"],
            **common,
        )
        weight, bias = fit_affine_probe(
            train.activations,
            train.beliefs,
            ridge=PROBE_RIDGE,
        )
        factual_activations, swapped_activations, indices, reconstruction_error = (
            paired_token_swap_activations(
                module,
                test,
                device=_device(context),
            )
        )

    target_error = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "Bayesian target is misaligned with environment diagnostics: "
            f"{target_error:.3e}"
        )
    evaluation = evaluate_token_swap(
        factual_activations=factual_activations,
        swapped_activations=swapped_activations,
        factual_targets=test.beliefs[indices],
        weight=weight,
        bias=bias,
    )
    factual_predictions = evaluation.pop("_factual_predictions")
    counterfactual_predictions = evaluation.pop("_counterfactual_predictions")
    counterfactual_targets = evaluation.pop("_counterfactual_targets")
    clusters = _episode_clusters(test, indices)
    bootstrap = cluster_bootstrap_statistics(
        clusters,
        lambda selected: (
            mean_squared_error(
                counterfactual_predictions[selected],
                counterfactual_targets[selected],
            )
            - mean_squared_error(
                factual_predictions[selected],
                test.beliefs[indices][selected],
            )
        ),
        n_resamples=n_resamples,
        seed=seed_sequence_to_int(streams["bootstrap"], bits=32),
    )
    delta_low, delta_high = percentile_interval(bootstrap)
    evaluation["counterfactual_minus_factual_mse_ci_95"] = [
        delta_low,
        delta_high,
    ]

    return {
        "schema_version": 1,
        "study": "state_0_1_token_swap_diagnostic",
        "cycle": cycle,
        "variant": 2,
        "seed": context.seed,
        "checkpoint": str(checkpoint.resolve()),
        "hypothesis": (
            "A state-0/state-1-equivariant belief representation should exchange "
            "decoded b0/b1, preserve decoded b2, and retain factual probe MSE."
        ),
        "intervention": (
            "Exchange token-0/token-1 channels at every position in each held-out "
            "history while holding its factual previous-action sequence fixed."
        ),
        "target_intervention": "exchange exact Bayesian b0/b1; preserve b2",
        "decoder": "frozen factual held-out affine least-squares probe",
        "representation": "post_final_layer_norm",
        "sampling_distribution": "greedy_process_weighted_factual_rollout",
        "policy_feedback": "disabled_during_paired_history_replay",
        "train_steps": train_steps,
        "test_steps": test_steps,
        "n_evaluated": int(len(indices)),
        "n_envs": N_ENVS,
        "warmup": warmup,
        "probe_ridge": PROBE_RIDGE,
        "bootstrap_n": n_resamples,
        "bootstrap_cluster": "environment_episode",
        "target_consistency_max_abs": target_error,
        "activation_reconstruction_max_abs": reconstruction_error,
        "metrics": evaluation,
        "interpretation": (
            "This tests a controlled representational response, not closed-loop "
            "policy use: policy actions and environment transitions are not rerun "
            "after the token intervention."
        ),
    }


def run_token_swap_diagnostic(
    context: RunContext,
    *,
    cycle: int,
) -> dict[str, Any]:
    if context.resume_from is None:
        raise ValueError(
            "resume_from must name an Algorithm checkpoint or a bundle with "
            "final_checkpoint/"
        )
    checkpoint = _resolve_checkpoint(Path(context.resume_from))
    summary = probe_checkpoint(context, checkpoint, cycle=cycle)
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "token_swap_diagnostic.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
