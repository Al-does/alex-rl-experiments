"""Belief probes and causal action evaluations for feedback cycle 1."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checkpoints import load_algorithm
from analysis.probes import (
    cluster_bootstrap_statistics,
    conditional_mse_metrics,
    fit_affine_probe,
    global_mse_metrics,
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
from experiments.mess3_token_guess_cycle_2.analysis import (
    ProbeResult,
    _device,
    _episode_clusters,
    _permutation_null_metrics,
    _simplex_display,
    plot_init_final,
    plot_probe_pair,
)
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int


PROBE_RIDGE = 1e-6
MIN_GROUP_SIZE = 50
N_ENVS = 16
FULL_RESAMPLES = 1_000
SMOKE_RESAMPLES = 100
FULL_TEST_STEPS = 80_000
PLOT_SAMPLE_SIZE = 80_000
CONTEXT_LENGTH = 10
_STREAM_KEYS = {
    "probe_train": (300,),
    "probe_test": (301,),
    "plot_sample": (302,),
    "bootstrap": (303,),
    "permutation": (304,),
    "permutation_sample": (305,),
    "shuffle": (306,),
}


def _action_corruptor(
    mode: str,
    *,
    seed: int | np.random.SeedSequence | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a transform for the three-dimensional previous-action block."""

    if mode not in {"mask", "shuffle"}:
        raise ValueError(f"unsupported action corruption {mode!r}")
    rng = np.random.default_rng(seed)

    def transform(observations: np.ndarray) -> np.ndarray:
        changed = np.array(observations, copy=True)
        action_block = changed[:, 3:6]
        if mode == "mask":
            action_block[:] = 0.0
        else:
            action_block[:] = action_block[rng.permutation(len(action_block))]
        return changed

    return transform


def _probe_metrics(
    predicted: np.ndarray,
    data: ProbeData,
) -> dict[str, float | int]:
    return {
        **global_mse_metrics(predicted, data.beliefs),
        **conditional_mse_metrics(
            predicted,
            data.beliefs,
            branch_keys(data, depth=2),
            min_group_size=MIN_GROUP_SIZE,
        ),
        "r_squared": r2_score(predicted, data.beliefs),
    }


def _previous_indices(data: ProbeData) -> np.ndarray:
    previous = np.full(len(data.beliefs), -1, dtype=np.int64)
    last_by_env: dict[int, int] = {}
    for index, (env_index, episode_step) in enumerate(
        zip(data.env_indices, data.episode_steps, strict=True)
    ):
        prior = last_by_env.get(int(env_index))
        if (
            prior is not None
            and int(data.episode_steps[prior]) + 1 == int(episode_step)
        ):
            previous[index] = prior
        last_by_env[int(env_index)] = index
    return previous


def _counterfactual_contexts(
    data: ProbeData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct factual context windows and their current sample indices."""

    previous = _previous_indices(data)
    observations = np.zeros((len(data.beliefs), 6), dtype=np.float32)
    valid_token = data.tokens >= 0
    observations[np.flatnonzero(valid_token), data.tokens[valid_token]] = 1.0
    valid_previous = previous >= 0
    previous_actions = np.asarray(data.actions).reshape(-1).astype(np.int64)
    rows = np.flatnonzero(valid_previous)
    observations[rows, 3 + previous_actions[previous[rows]]] = 1.0

    contexts: list[np.ndarray] = []
    sample_indices: list[int] = []
    predecessor_indices: list[int] = []
    histories: dict[int, list[int]] = {}
    for index, (env_index, episode_step) in enumerate(
        zip(data.env_indices, data.episode_steps, strict=True)
    ):
        history = histories.setdefault(int(env_index), [])
        if history and int(data.episode_steps[history[-1]]) + 1 != int(episode_step):
            history.clear()
        history.append(index)
        if len(history) >= CONTEXT_LENGTH + 1:
            contexts.append(observations[history[-CONTEXT_LENGTH:]])
            sample_indices.append(index)
            predecessor_indices.append(history[-2])
    return (
        np.asarray(contexts, dtype=np.float32),
        np.asarray(sample_indices, dtype=np.int64),
        np.asarray(predecessor_indices, dtype=np.int64),
    )


@torch.no_grad()
def _encode_contexts(
    module: Any,
    contexts: np.ndarray,
    *,
    device: str,
    batch_size: int = 2_048,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    torch_device = torch.device(device)
    for start in range(0, len(contexts), batch_size):
        batch = torch.from_numpy(contexts[start : start + batch_size]).to(
            torch_device
        )
        state = {
            "ctx": batch[:, :-1, :],
            "len": torch.full(
                (len(batch), 1),
                CONTEXT_LENGTH - 1,
                dtype=torch.float32,
                device=torch_device,
            ),
        }
        embedding, _ = module.encode_step(batch[:, -1, :], state)
        encoded.append(embedding.cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.float64, copy=False)


def _shift_metrics(
    decoded: np.ndarray,
    exact: np.ndarray,
) -> dict[str, float | int]:
    decoded_norm = np.linalg.norm(decoded, axis=1)
    exact_norm = np.linalg.norm(exact, axis=1)
    valid = (decoded_norm > 1e-12) & (exact_norm > 1e-12)
    cosine = np.sum(decoded[valid] * exact[valid], axis=1) / (
        decoded_norm[valid] * exact_norm[valid]
    )
    return {
        "shift_mse": mean_squared_error(decoded, exact),
        "shift_cosine_mean": float(cosine.mean()) if len(cosine) else float("nan"),
        "shift_norm_ratio_mean": float(
            np.mean(decoded_norm[valid] / exact_norm[valid])
        )
        if len(cosine)
        else float("nan"),
        "exact_shift_rms": float(np.sqrt(np.mean(np.square(exact)))),
        "decoded_shift_rms": float(np.sqrt(np.mean(np.square(decoded)))),
        "n_evaluated": int(len(exact)),
        "n_nonzero": int(np.count_nonzero(valid)),
    }


def _counterfactual_action_evaluation(
    module: Any,
    data: ProbeData,
    *,
    weight: np.ndarray,
    bias: np.ndarray,
    transitions: np.ndarray,
    emission: np.ndarray,
    device: str,
) -> dict[str, Any]:
    contexts, sample_indices, predecessor_indices = _counterfactual_contexts(data)
    factual_embeddings = _encode_contexts(module, contexts, device=device)
    reconstruction_error = float(
        np.max(np.abs(factual_embeddings - data.activations[sample_indices]))
    )
    if reconstruction_error > 2e-5:
        raise AssertionError(
            "reconstructed transformer contexts do not match rollout activations: "
            f"{reconstruction_error:.3e}"
        )

    factual_actions = (
        np.asarray(data.actions).reshape(-1)[predecessor_indices].astype(np.int64)
    )
    alternatives = np.asarray(
        [
            alternative
            for action in factual_actions
            for alternative in range(3)
            if alternative != action
        ],
        dtype=np.int64,
    )
    repeated_contexts = np.repeat(contexts, 2, axis=0)
    repeated_contexts[:, -1, 3:6] = 0.0
    repeated_contexts[np.arange(len(alternatives)), -1, 3 + alternatives] = 1.0
    counterfactual_embeddings = _encode_contexts(
        module,
        repeated_contexts,
        device=device,
    )

    before = np.repeat(data.beliefs[predecessor_indices], 2, axis=0)
    tokens = np.repeat(data.tokens[sample_indices], 2)
    measured = before * emission[:, tokens].T
    measured /= measured.sum(axis=1, keepdims=True)
    exact_counterfactual = np.einsum(
        "ni,nij->nj",
        measured,
        transitions[alternatives],
    )
    exact_factual = np.repeat(data.beliefs[sample_indices], 2, axis=0)
    exact_shift = exact_counterfactual - exact_factual
    decoded_factual = np.repeat(
        probe_predict(weight, bias, factual_embeddings),
        2,
        axis=0,
    )
    decoded_counterfactual = probe_predict(
        weight,
        bias,
        counterfactual_embeddings,
    )
    decoded_shift = decoded_counterfactual - decoded_factual

    recomputed_factual = np.einsum(
        "ni,nij->nj",
        measured,
        transitions[np.repeat(factual_actions, 2)],
    )
    target_error = float(np.max(np.abs(recomputed_factual - exact_factual)))
    if target_error > 1e-10:
        raise AssertionError(
            f"counterfactual target alignment failed: {target_error:.3e}"
        )

    repeated_factual_actions = np.repeat(factual_actions, 2)
    by_factual_action = {
        str(action): _shift_metrics(
            decoded_shift[repeated_factual_actions == action],
            exact_shift[repeated_factual_actions == action],
        )
        for action in range(3)
    }
    return {
        "schema_version": 1,
        "intervention": (
            "replace the previous executed action in the current observation "
            "while holding the preceding token/action context fixed"
        ),
        "target": "exact_delay_one_action_conditioned_bayesian_belief_shift",
        "representation": "post_final_layer_norm",
        "alternatives": "all_other_actions",
        **_shift_metrics(decoded_shift, exact_shift),
        "by_factual_action": by_factual_action,
        "activation_reconstruction_max_abs": reconstruction_error,
        "target_reconstruction_max_abs": target_error,
        "interpretation": (
            "Tests whether local representational sensitivity agrees with the "
            "Bayesian effect of the previous action."
        ),
    }


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    agent_steps: int | None = None,
    run_causal_evaluations: bool = False,
    train_steps: int | None = None,
    test_steps: int | None = None,
) -> ProbeResult:
    """Fit the standard probe and optionally run final causal evaluations."""

    if context.seed is None:
        raise ValueError("feedback-cycle probing requires a resolved seed")
    context.results_dir.mkdir(parents=True, exist_ok=True)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = train_steps or (4_096 if context.smoke else 60_000)
    test_steps = test_steps or (4_096 if context.smoke else FULL_TEST_STEPS)
    warmup = 4 if context.smoke else 64
    n_resamples = SMOKE_RESAMPLES if context.smoke else FULL_RESAMPLES

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        environment_class = algorithm.config.env
        environment_config = dict(algorithm.config.env_config)
        environment_config["diagnostics"] = {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        }

        def make_environment():
            return environment_class(environment_config)

        environment = make_environment()
        try:
            initial_belief, outcome_operator, initial_operator = (
                make_transducer_target(environment)
            )
            transitions = np.stack(
                [
                    environment.task.transition_matrix_for_action(action)
                    for action in range(3)
                ]
            )
            emission = np.asarray(
                environment.model.emission_matrix,
                dtype=np.float64,
            )
        finally:
            environment.close()
        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": _device(context),
            "warmup": warmup,
            "n_envs": N_ENVS,
            "initial_belief": initial_belief,
            "action_outcome_operator": outcome_operator,
            "initial_outcome_operator": initial_operator,
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
        causal: dict[str, Any] | None = None
        if run_causal_evaluations:
            ablations: dict[str, Any] = {}
            for mode in ("mask", "shuffle"):
                transform_seed = (
                    seed_sequence_to_int(streams["shuffle"], bits=32)
                    if mode == "shuffle"
                    else None
                )
                ablated = collect_probe_data(
                    n_steps=test_steps,
                    seed=streams["probe_test"],
                    observation_transform=_action_corruptor(
                        mode,
                        seed=transform_seed,
                    ),
                    **common,
                )
                ablated_predicted = probe_predict(
                    weight,
                    bias,
                    ablated.activations,
                )
                ablations[mode] = {
                    "token_accuracy": float(ablated.rewards.mean()),
                    **_probe_metrics(ablated_predicted, ablated),
                }
            baseline_predicted = probe_predict(weight, bias, test.activations)
            baseline = {
                "token_accuracy": float(test.rewards.mean()),
                **_probe_metrics(baseline_predicted, test),
            }
            for payload in ablations.values():
                payload["delta_token_accuracy"] = (
                    float(payload["token_accuracy"])
                    - float(baseline["token_accuracy"])
                )
                payload["mse_inflation"] = (
                    float(payload["mse"]) / float(baseline["mse"])
                )
            causal = {
                "schema_version": 1,
                "action_input_ablation": {
                    "baseline": baseline,
                    "corruptions": ablations,
                    "policy_mode": "greedy_closed_loop",
                    "interpretation": (
                        "Masking or shuffling changes only policy-visible "
                        "previous-action features; transitions use executed actions."
                    ),
                },
                "counterfactual_belief_shift": _counterfactual_action_evaluation(
                    module,
                    test,
                    weight=weight,
                    bias=bias,
                    transitions=transitions,
                    emission=emission,
                    device=_device(context),
                ),
            }

    target_error = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "Bayesian target is misaligned with environment diagnostics: "
            f"{target_error:.3e}"
        )
    predicted = probe_predict(weight, bias, test.activations)
    metrics: dict[str, Any] = {
        "condition": condition,
        "checkpoint_step": agent_steps,
        "is_untrained": agent_steps == 0,
        "target": "exact_predictive_bayesian_belief",
        "probe": "held_out_affine_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "representation": "post_final_layer_norm",
        "sampling_distribution": "process_weighted_rollout",
        "policy_mode": "greedy",
        "warmup": warmup,
        "n_envs": N_ENVS,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "branch_depth": 2,
        "min_group_size": MIN_GROUP_SIZE,
        **_probe_metrics(predicted, test),
    }
    bootstrap = cluster_bootstrap_statistics(
        _episode_clusters(test),
        lambda indices: mean_squared_error(
            predicted[indices],
            test.beliefs[indices],
        ),
        n_resamples=n_resamples,
        seed=seed_sequence_to_int(streams["bootstrap"], bits=32),
    )
    metrics["mse_ci_95_low"], metrics["mse_ci_95_high"] = percentile_interval(
        bootstrap
    )
    metrics.update(
        {
            "bootstrap_n": n_resamples,
            "bootstrap_cluster": "environment_episode",
            **_permutation_null_metrics(
                train,
                test,
                n_permutations=n_resamples,
                sample_seed=seed_sequence_to_int(
                    streams["permutation_sample"],
                    bits=32,
                ),
                permutation_seed=seed_sequence_to_int(
                    streams["permutation"],
                    bits=32,
                ),
            ),
            "token_accuracy_greedy": float(test.rewards.mean()),
            "bayesian_optimal_accuracy_on_rollout": float(
                np.max(test.beliefs @ emission, axis=1).mean()
            ),
            "n_fit": len(train.beliefs),
            "n_test": len(test.beliefs),
            "target_consistency_max_abs": target_error,
            "interpretation": (
                "Affine decodability alone does not establish causal policy use."
            ),
        }
    )
    if agent_steps == 0:
        metrics["untrained_mse"] = metrics["mse"]
    if causal is not None:
        metrics["causal_evaluations"] = causal
        (context.results_dir / "causal_evaluations.json").write_text(
            json.dumps(causal, indent=2) + "\n"
        )

    sample_size = min(PLOT_SAMPLE_SIZE, len(test.beliefs))
    sample_rng = np.random.default_rng(streams["plot_sample"])
    indices = sample_rng.choice(len(test.beliefs), sample_size, replace=False)
    result = ProbeResult(
        metrics=metrics,
        target_display=test.beliefs[indices],
        decoded_display=_simplex_display(predicted[indices]),
    )
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    plot_probe_pair(
        result,
        title=condition.replace("_", " "),
        path=context.results_dir / "belief_simplex.png",
    )
    return result


def plot_probe_trajectory(
    checkpoints: list[Mapping[str, Any]],
    *,
    path: Path,
) -> None:
    """Plot normalized MSE, achieved accuracy, and on-rollout Bayes accuracy."""

    steps = np.asarray([point["agent_steps"] for point in checkpoints])
    mse_ratio = np.asarray([point["global_mse_ratio"] for point in checkpoints])
    accuracy = 100.0 * np.asarray(
        [point["token_accuracy_greedy"] for point in checkpoints]
    )
    bayes = 100.0 * np.asarray(
        [point["bayesian_optimal_accuracy_on_rollout"] for point in checkpoints]
    )
    figure, left = plt.subplots(figsize=(8.2, 4.8))
    right = left.twinx()
    left.plot(steps, mse_ratio, marker="o", color="#355c9a", label="MSE / variance")
    right.plot(steps, accuracy, marker="s", color="#c45135", label="Agent accuracy")
    right.plot(
        steps,
        bayes,
        marker="^",
        linestyle="--",
        color="#222222",
        label="On-rollout Bayes accuracy",
    )
    left.set_xlabel("Environment steps")
    left.set_ylabel("Normalized probe MSE", color="#355c9a")
    right.set_ylabel("Token accuracy (%)", color="#c45135")
    left.set_title("MESS3 feedback cycle 1")
    left.grid(alpha=0.2)
    left_handles, left_labels = left.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    left.legend(left_handles + right_handles, left_labels + right_labels)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


__all__ = [
    "ProbeResult",
    "plot_init_final",
    "plot_probe_pair",
    "plot_probe_trajectory",
    "probe_checkpoint",
]
