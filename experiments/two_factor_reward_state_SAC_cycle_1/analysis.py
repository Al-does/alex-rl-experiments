"""Action-aware belief probes for joint and factor representations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checkpoints import load_algorithm
from analysis.probes import global_mse_metrics, r2_score, variance_geometry
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.hmm import HMMEnv, factor_marginals, product_distribution
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from experiments.factored_representations_reproduction_SAC_2026_08.analysis import (
    cross_validated_svd_affine,
)
from experiments.two_factor_reward_state_SAC_cycle_1.process import (
    FACTOR_CARDINALITY,
    FACTOR_COUNT,
    decode_joint_indices,
    environment_config,
)


FULL_PROBE_TRAIN_STEPS = 20_000
FULL_PROBE_TEST_STEPS = 20_000
SMOKE_PROBE_STEPS = 128
N_ENVS = 8
WARMUP = 8
_STREAM_KEYS = {
    "probe_train": (700,),
    "probe_test": (701,),
    "regression_joint": (702,),
    "regression_factor_1": (703,),
    "regression_factor_2": (704,),
}


@dataclass(frozen=True, slots=True)
class ProbeData:
    """Aligned actor activations and exact action-conditioned filter targets."""

    activations: np.ndarray
    joint_beliefs: np.ndarray
    factor_beliefs: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    product_consistency_max_abs: float


def _device(context: RunContext) -> torch.device:
    profile = context.hardware
    return torch.device(
        "cuda"
        if profile is not None
        and profile.learner_device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )


@torch.inference_mode()
def collect_probe_data(
    module: Any,
    *,
    condition: str,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
) -> ProbeData:
    """Collect greedy actor residuals and exact transducer beliefs."""

    config = environment_config(condition)
    config["diagnostics"] = {"belief": True, "state": True}
    module = module.to(device).eval()

    def make_environment() -> HMMEnv:
        return HMMEnv(config)

    def step_adapter(
        observations: np.ndarray,
        state: None,
        randomness: PolicyRandomness,
        action_spaces: Any,
    ):
        del state, randomness, action_spaces
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        residual = module.actor_hidden(tensor)
        normalized = module.encoder.final_norm(residual)
        logits = module.action_distribution_inputs(normalized)
        return (
            logits.argmax(dim=-1).cpu().numpy(),
            None,
            residual.cpu().numpy(),
        )

    def target_adapter(
        observations: np.ndarray,
        infos: list[Mapping[str, Any]],
        episode_steps: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        del observations, episode_steps
        joint = np.stack([info["belief_current"] for info in infos])
        factors = np.stack(
            factor_marginals(
                joint,
                (FACTOR_CARDINALITY,) * FACTOR_COUNT,
            ),
            axis=1,
        )
        return {
            "joint_belief": joint,
            "factor_belief": factors,
            "state": np.asarray(
                [info["state_current"] for info in infos],
                dtype=np.int64,
            ),
        }

    collected = collect_batched_rollout_data(
        make_environment,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=N_ENVS,
        warmup=WARMUP,
    )
    joint = np.asarray(collected.targets["joint_belief"], dtype=np.float64)
    factors = np.asarray(collected.targets["factor_belief"], dtype=np.float64)
    reconstructed = product_distribution(
        [factors[:, index] for index in range(FACTOR_COUNT)]
    )
    return ProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        joint_beliefs=joint,
        factor_beliefs=factors,
        states=np.asarray(collected.targets["state"], dtype=np.int64),
        actions=np.asarray(collected.actions, dtype=np.int64).reshape(-1),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
        product_consistency_max_abs=float(np.max(np.abs(joint - reconstructed))),
    )


def _fit_report(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
    *,
    target_name: str,
    seed: int,
) -> dict[str, Any]:
    weight, bias, fit = cross_validated_svd_affine(
        train_features,
        train_target,
        seed=seed,
    )
    predicted = test_features @ weight + bias
    residual = predicted - test_target
    return {
        "target": target_name,
        "fit": fit,
        **global_mse_metrics(predicted, test_target),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r_squared": r2_score(predicted, test_target),
    }


def analyze_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
) -> dict[str, Any]:
    """Probe the actor for joint belief, each factor belief, and 95% CEV."""

    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = (
        SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TRAIN_STEPS
    )
    test_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TEST_STEPS
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        train = collect_probe_data(
            module,
            condition=condition,
            n_steps=train_steps,
            seed=streams["probe_train"],
            device=_device(context),
        )
        test = collect_probe_data(
            module,
            condition=condition,
            n_steps=test_steps,
            seed=streams["probe_test"],
            device=_device(context),
        )

    consistency = max(
        train.product_consistency_max_abs,
        test.product_consistency_max_abs,
    )
    if consistency > 1e-10:
        raise AssertionError(
            "controlled independent-factor belief lost product structure: "
            f"{consistency:.3e}"
        )
    fits = {
        "joint_mixed_state": _fit_report(
            train.activations,
            train.joint_beliefs,
            test.activations,
            test.joint_beliefs,
            target_name="exact_action_conditioned_joint_predictive_belief",
            seed=seed_sequence_to_int(streams["regression_joint"], bits=32),
        ),
        "factor_1": _fit_report(
            train.activations,
            train.factor_beliefs[:, 0],
            test.activations,
            test.factor_beliefs[:, 0],
            target_name="exact_action_conditioned_factor_1_predictive_belief",
            seed=seed_sequence_to_int(streams["regression_factor_1"], bits=32),
        ),
        "factor_2": _fit_report(
            train.activations,
            train.factor_beliefs[:, 1],
            test.activations,
            test.factor_beliefs[:, 1],
            target_name="exact_action_conditioned_factor_2_predictive_belief",
            seed=seed_sequence_to_int(streams["regression_factor_2"], bits=32),
        ),
    }
    decoded_states = decode_joint_indices(test.states)
    result = {
        "condition": condition,
        "checkpoint": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "is_initialization": training_iteration == 0,
        "representation": "actor_final_block_residual_before_final_layer_norm",
        "probe_variant": (
            "transducer: exact Bayesian targets update through the executed "
            "factor-action pair before conditioning on each new joint token"
        ),
        "n_fit": len(train.activations),
        "n_test": len(test.activations),
        "product_consistency_max_abs": consistency,
        "probe_fits": fits,
        "cev": {
            "actor_activation": variance_geometry(test.activations),
            "joint_mixed_state_target": variance_geometry(test.joint_beliefs),
            "factor_1_target": variance_geometry(test.factor_beliefs[:, 0]),
            "factor_2_target": variance_geometry(test.factor_beliefs[:, 1]),
        },
        "policy": {
            "mean_reward": float(test.rewards.mean()),
            "factor_1_state_2_fraction": float(
                np.mean(decoded_states[:, 0] == 2)
            ),
            "factor_2_state_2_fraction": float(
                np.mean(decoded_states[:, 1] == 2)
            ),
            "greedy_action_fractions": np.bincount(
                test.actions,
                minlength=9,
            ).astype(np.float64).tolist(),
        },
        "scope_warning": (
            "Linear accessibility and low CEV dimension do not by themselves "
            "establish causal use or orthogonal factor subspaces."
        ),
    }
    result["policy"]["greedy_action_fractions"] = (
        np.asarray(result["policy"]["greedy_action_fractions"]) / len(test.actions)
    ).tolist()
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "probe_battery.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def plot_probe_trajectory(
    reports: list[dict[str, Any]],
    *,
    condition: str,
    path: Path,
) -> None:
    """Plot longitudinal factor fits and actor effective dimension."""

    steps = np.asarray([report["agent_steps"] for report in reports])
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for target in ("joint_mixed_state", "factor_1", "factor_2"):
        axes[0].plot(
            steps,
            [report["probe_fits"][target]["r_squared"] for report in reports],
            marker="o",
            label=target,
        )
    axes[0].set_ylabel("Held-out linear-probe R²")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        steps,
        [
            report["cev"]["actor_activation"]["cev95_dimension"]
            for report in reports
        ],
        marker="o",
    )
    axes[1].set_ylabel("Actor dimensions for 95% CEV")
    for axis in axes:
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.2)
    figure.suptitle(condition.replace("_", " "))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
