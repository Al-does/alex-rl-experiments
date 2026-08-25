"""Held-out belief probes and factor geometry for independent MESS3 factors."""

from __future__ import annotations

from dataclasses import dataclass
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
    correlation_residual_metrics,
    fit_correlation_residual_probe,
    fit_affine_probe,
    fit_product_constrained_joint_probe,
    global_mse_metrics,
    joint_readout_excess_subspace,
    percentile_interval,
    probe_predict,
    product_constrained_joint_metrics,
    r2_score,
    regression_factor_geometry,
    representation_dimension_predictions,
)
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.hmm import factor_marginals
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences


FACTOR_SIZE = 3
PROBE_RIDGE = 1e-6
N_ENVS = 8
FULL_TRAIN_STEPS = 30_000
FULL_TEST_STEPS = 40_000
SMOKE_STEPS = 1_024
_STREAM_KEYS = {
    "probe_train": (610,),
    "probe_test": (611,),
    "pcjr_bootstrap": (612,),
}


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    observations: np.ndarray
    joint_belief: np.ndarray
    factor_beliefs: tuple[np.ndarray, ...]
    tokens: np.ndarray
    env_indices: np.ndarray
    episode_steps: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


def _initial_state(module: Any, batch_size: int, device: torch.device):
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        .to(device)
        for key, value in module.get_initial_state().items()
    }


@torch.inference_mode()
def collect_probe_data(
    module: Any,
    env_factory,
    *,
    n_steps: int,
    seed,
    device: str | torch.device,
    warmup: int,
    n_factors: int,
) -> ProbeData:
    """Collect paper-comparable residuals and exact aligned belief targets."""

    device = torch.device(device)
    module = module.to(device).eval()
    stateful = module.is_stateful()

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices: np.ndarray):
        fresh = _initial_state(module, len(indices), device)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index_tensor, fresh[key])
        return state

    def step_adapter(
        observations: np.ndarray,
        state: Any,
        randomness: PolicyRandomness,
        action_spaces: Any,
    ):
        del action_spaces
        observation_tensor = torch.from_numpy(observations).float().to(device)
        embedding, state = module.encode_step_pre_final_norm(
            observation_tensor,
            state,
        )
        actions = randomness.numpy.integers(
            0,
            FACTOR_SIZE**n_factors,
            size=len(observations),
        )
        return actions, state, embedding.cpu().numpy()

    def target_adapter(observations, infos, episode_steps):
        del observations
        joint = np.stack([info["belief_current"] for info in infos])
        factors = factor_marginals(joint, (FACTOR_SIZE,) * n_factors)
        return {
            "joint_belief": joint,
            **{
                f"factor_{index}": belief
                for index, belief in enumerate(factors)
            },
            "token": np.asarray(
                [info["visible_token_current"] for info in infos],
                dtype=np.int64,
            ),
            "env_index": np.arange(len(infos), dtype=np.int64),
            "episode_step": np.asarray(episode_steps, dtype=np.int64),
        }

    collected = collect_batched_rollout_data(
        env_factory,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=N_ENVS,
        initial_state=initial_state if stateful else None,
        reset_state=reset_state if stateful else None,
        warmup=warmup,
        store_observations=True,
    )
    if collected.observations is None:
        raise AssertionError("probe collection requested policy observations")
    return ProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        observations=np.asarray(collected.observations, dtype=np.float64),
        joint_belief=np.asarray(
            collected.targets["joint_belief"],
            dtype=np.float64,
        ),
        factor_beliefs=tuple(
            np.asarray(collected.targets[f"factor_{index}"], dtype=np.float64)
            for index in range(n_factors)
        ),
        tokens=np.asarray(collected.targets["token"], dtype=np.int64),
        env_indices=np.asarray(
            collected.targets["env_index"],
            dtype=np.int64,
        ),
        episode_steps=np.asarray(
            collected.targets["episode_step"],
            dtype=np.int64,
        ),
        actions=np.asarray(collected.actions, dtype=np.int64).reshape(-1),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
    )


def _fit_target(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    *,
    groups: np.ndarray,
) -> dict[str, float | int]:
    weight, bias = fit_affine_probe(
        train_features,
        train_target,
        ridge=PROBE_RIDGE,
    )
    predicted = probe_predict(weight, bias, test_features)
    return {
        **global_mse_metrics(predicted, test_target),
        "r_squared": r2_score(predicted, test_target),
        **conditional_mse_metrics(
            predicted,
            test_target,
            groups,
            min_group_size=20,
        ),
    }


def _episode_clusters(data: ProbeData) -> np.ndarray:
    """Return stable cluster IDs for complete environment episodes."""

    clusters = np.empty(len(data.episode_steps), dtype=np.int64)
    next_cluster = 0
    for env_index in np.unique(data.env_indices):
        members = np.flatnonzero(data.env_indices == env_index)
        current = next_cluster
        first = True
        for index in members:
            if not first and data.episode_steps[index] == 0:
                current += 1
            clusters[index] = current
            first = False
        next_cluster = current + 1
    return clusters


def _pcjr_bootstrap(
    *,
    direct_prediction: np.ndarray,
    product_prediction: np.ndarray,
    target: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired PCJR error differences over complete episodes."""

    differences = (
        np.square(product_prediction - target).mean(axis=1)
        - np.square(direct_prediction - target).mean(axis=1)
    )
    estimates = cluster_bootstrap_statistics(
        clusters,
        lambda indices: float(differences[indices].mean()),
        n_resamples=n_resamples,
        seed=seed,
    )
    low, high = percentile_interval(estimates)
    return {
        "paired_error_difference": float(differences.mean()),
        "paired_episode_bootstrap_ci95_low": low,
        "paired_episode_bootstrap_ci95_high": high,
        "paired_episode_bootstrap_n": n_resamples,
        "reading": (
            "positive means the direct joint probe has lower held-out MSE; "
            "negative means product-constrained reconstruction is better"
        ),
    }


def _plot_cev(
    metrics: dict[str, Any],
    *,
    path: Path,
    n_factors: int,
) -> None:
    pca = metrics["geometry"]["activation_pca"]
    cumulative = np.asarray(pca["cumulative_explained_variance"])
    predictions = metrics["dimension_predictions"]
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
        marker=".",
        label="Transformer residual",
    )
    axis.axhline(0.95, color="#333333", linestyle="--", linewidth=1.0)
    axis.axvline(
        predictions["factored"],
        color="#5a9f68",
        linestyle=":",
        label=f"Factored prediction ({predictions['factored']})",
    )
    axis.axvline(
        predictions["joint"],
        color="#b05a4a",
        linestyle=":",
        label=f"Joint prediction ({predictions['joint']})",
    )
    axis.set_xlabel("Principal components")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0.0, 1.01)
    axis.set_title(
        f"{n_factors}-MESS3 belief geometry — step {metrics['checkpoint_step']}"
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    agent_steps: int,
    n_factors: int,
) -> dict[str, Any]:
    """Fit held-out joint/factor probes and run paper-inspired geometry tests."""

    if context.seed is None:
        raise ValueError("factored MESS3 probing requires a resolved seed")
    context.results_dir.mkdir(parents=True, exist_ok=True)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    n_steps = SMOKE_STEPS if context.smoke else None
    train_steps = n_steps or FULL_TRAIN_STEPS
    test_steps = n_steps or FULL_TEST_STEPS
    warmup = 4 if context.smoke else 64
    profile = context.hardware or PROFILES["cpu"]
    device = (
        "cuda"
        if profile.learner_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        env_config["diagnostics"] = {
            "belief": True,
            "tokens": True,
        }

        def make_environment():
            return env_class(env_config)

        common = {
            "module": module,
            "env_factory": make_environment,
            "device": device,
            "warmup": warmup,
            "n_factors": n_factors,
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

    factor_targets = {
        f"factor_{index}": train.factor_beliefs[index]
        for index in range(n_factors)
    }
    geometry = regression_factor_geometry(
        train.activations,
        factor_targets,
        ridge=PROBE_RIDGE,
        target_ranks={name: FACTOR_SIZE - 1 for name in factor_targets},
    )
    targets = {
        "joint_belief": _fit_target(
            train.activations,
            test.activations,
            train.joint_belief,
            test.joint_belief,
            groups=test.tokens,
        ),
        **{
            f"factor_{index}": _fit_target(
                train.activations,
                test.activations,
                train.factor_beliefs[index],
                test.factor_beliefs[index],
                groups=test.tokens,
            )
            for index in range(n_factors)
        },
    }
    observation_baselines = {
        "joint_belief": _fit_target(
            train.observations,
            test.observations,
            train.joint_belief,
            test.joint_belief,
            groups=test.tokens,
        ),
        **{
            f"factor_{index}": _fit_target(
                train.observations,
                test.observations,
                train.factor_beliefs[index],
                test.factor_beliefs[index],
                groups=test.tokens,
            )
            for index in range(n_factors)
        },
    }
    pcjr_probe = fit_product_constrained_joint_probe(
        train.activations,
        train.joint_belief,
        train.factor_beliefs,
        test.activations,
        ridge=PROBE_RIDGE,
    )
    pcjr = {
        **product_constrained_joint_metrics(
            pcjr_probe,
            test.joint_belief,
        ),
        **_pcjr_bootstrap(
            direct_prediction=pcjr_probe.direct_prediction,
            product_prediction=pcjr_probe.product_prediction,
            target=test.joint_belief,
            clusters=_episode_clusters(test),
            n_resamples=100 if context.smoke else 1_000,
            seed=int(streams["pcjr_bootstrap"].generate_state(1)[0]),
        ),
    }
    crd_probe = fit_correlation_residual_probe(
        train.activations,
        train.joint_belief,
        train.factor_beliefs,
        test.activations,
        test.joint_belief,
        test.factor_beliefs,
        ridge=PROBE_RIDGE,
    )
    crd = correlation_residual_metrics(crd_probe)
    jres = joint_readout_excess_subspace(
        pcjr_probe.direct_weight,
        pcjr_probe.factor_weights,
        joint_rank=FACTOR_SIZE**n_factors - 1,
        factor_ranks=(FACTOR_SIZE - 1,) * n_factors,
    )
    predictions = representation_dimension_predictions(
        [FACTOR_SIZE] * n_factors
    )
    metrics = {
        "checkpoint_step": int(agent_steps),
        "is_untrained": agent_steps == 0,
        "representation": "pre_final_layer_norm_decision_token",
        "sampling_distribution": "process_weighted_random_action_rollout",
        "probe": "held_out_affine_ridge_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "n_fit": train_steps,
        "n_test": test_steps,
        "n_envs": N_ENVS,
        "warmup": warmup,
        "n_factors": n_factors,
        "targets": targets,
        "observation_only_baselines": observation_baselines,
        "product_constrained_joint_reconstruction": pcjr,
        "correlation_residual_decodability": crd,
        "joint_readout_excess_subspace": jres,
        "probe_degrees_of_freedom": {
            "activation_width": int(train.activations.shape[1]),
            "direct_joint_simplex_adjusted": (
                (train.activations.shape[1] + 1)
                * (FACTOR_SIZE**n_factors - 1)
            ),
            "all_factor_simplex_adjusted": (
                (train.activations.shape[1] + 1)
                * n_factors
                * (FACTOR_SIZE - 1)
            ),
        },
        "geometry": geometry,
        "dimension_predictions": predictions,
        "behavior_reward_mean": float(test.rewards.mean()),
        "interpretation": (
            "Factored native geometry requires decodable factor beliefs, "
            "activation dimension near the direct-sum prediction, and low "
            "principal-angle overlap. No single metric is decisive, and "
            "decodability does not establish causal policy use."
        ),
    }
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    _plot_cev(
        metrics,
        path=context.results_dir / "cev.png",
        n_factors=n_factors,
    )
    return metrics
