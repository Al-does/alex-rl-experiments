from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from analysis.checkpoints import load_algorithm
from analysis.probes import global_mse_metrics, r2_score, variance_geometry
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.hmm import HMMEnv, factor_marginals, product_distribution
from experiments.factored_representations_reproduction_PPO_2026_08.analysis import (
    cross_validated_svd_affine,
)
from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from experiments.wing_token_guess_cycle_1.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    FACTOR_COUNT,
    JOINT_TOKEN_COUNT,
    environment_config,
)
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int


FULL_PROBE_TRAIN_STEPS = 20_000
FULL_PROBE_TEST_STEPS = 20_000
SMOKE_PROBE_STEPS = 256
N_ENVS = 8
WARMUP = CONTEXT_LENGTH
_STREAM_KEYS = {
    "probe_train": (800,),
    "probe_test": (801,),
    "regression": (802,),
    "shuffle": (803,),
}


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    joint_beliefs: np.ndarray
    factor_beliefs: np.ndarray
    observations: np.ndarray
    states: np.ndarray
    hidden_tokens: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_steps: np.ndarray
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


def _target_adapter(
    observations: np.ndarray,
    infos: list[Mapping[str, Any]],
    episode_steps: np.ndarray,
) -> Mapping[str, np.ndarray]:
    del observations
    joint = np.stack([info["belief_current"] for info in infos])
    return {
        "joint_belief": joint,
        "factor_belief": np.stack(
            factor_marginals(joint, (3, 3)),
            axis=1,
        ),
        "state": np.asarray(
            [info["state_current"] for info in infos],
            dtype=np.int64,
        ),
        "hidden_token": np.asarray(
            [info["raw_token_current"] for info in infos],
            dtype=np.int64,
        ),
        "episode_step": np.asarray(episode_steps, dtype=np.int64),
    }


def _sample_actions(
    logits: torch.Tensor,
    randomness: PolicyRandomness,
) -> np.ndarray:
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.asarray(
        [
            randomness.numpy.choice(len(probabilities_row), p=probabilities_row)
            for probabilities_row in probabilities
        ],
        dtype=np.int64,
    )


@torch.inference_mode()
def collect_probe_data(
    module: Any,
    *,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
) -> ProbeData:
    config = environment_config()
    config["diagnostics"] = {
        "belief": True,
        "state": True,
        "tokens": True,
    }
    module = module.to(device).eval()
    blocks = module.encoder.blocks
    captured: dict[int, torch.Tensor] = {}

    def capture(index: int):
        def hook(block, inputs, output):
            del block, inputs
            captured[index] = output[:, -1, :].detach()

        return hook

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices):
        fresh = initial_state(len(indices))
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    def step_adapter(observations, state, randomness, action_spaces):
        del action_spaces
        captured.clear()
        tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=device,
        )
        residual, state_out = module.encode_step_pre_final_norm(tensor, state)
        if len(captured) != len(blocks):
            raise RuntimeError("every encoder block must execute once per probe step")
        layers = torch.stack(
            [captured[index] for index in range(len(blocks))],
            dim=1,
        )
        logits = module.action_distribution_inputs(
            module.encoder.final_norm(residual)
        )
        return (
            _sample_actions(logits, randomness),
            state_out,
            layers.cpu().numpy(),
        )

    handles = []
    try:
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(capture(index)))
        collected = collect_batched_rollout_data(
            lambda: HMMEnv(config),
            step_adapter,
            _target_adapter,
            n_steps=n_steps,
            seed=seed,
            n_envs=N_ENVS,
            initial_state=initial_state,
            reset_state=reset_state,
            warmup=WARMUP,
            store_observations=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if collected.observations is None:
        raise AssertionError("probe collection requires policy observations")
    joint = np.asarray(collected.targets["joint_belief"], dtype=np.float64)
    factors = np.asarray(collected.targets["factor_belief"], dtype=np.float64)
    reconstructed = product_distribution(
        [factors[:, index] for index in range(FACTOR_COUNT)]
    )
    actions = np.asarray(collected.actions, dtype=np.int64).reshape(-1)
    hidden_tokens = np.asarray(
        collected.targets["hidden_token"],
        dtype=np.int64,
    )
    rewards = np.asarray(collected.rewards, dtype=np.float64)
    if not np.array_equal(rewards, (actions == hidden_tokens).astype(np.float64)):
        raise AssertionError("token-guess rewards are misaligned with action-time targets")
    return ProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        joint_beliefs=joint,
        factor_beliefs=factors,
        observations=np.asarray(collected.observations, dtype=np.float64),
        states=np.asarray(collected.targets["state"], dtype=np.int64),
        hidden_tokens=hidden_tokens,
        actions=actions,
        rewards=rewards,
        episode_steps=np.asarray(
            collected.targets["episode_step"],
            dtype=np.int64,
        ),
        product_consistency_max_abs=float(
            np.max(np.abs(joint - reconstructed))
        ),
    )


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    metrics = global_mse_metrics(predicted, target)
    return {
        **metrics,
        "normalized_mse": metrics["global_mse_ratio"],
        "rmse": float(np.sqrt(np.mean(np.square(predicted - target)))),
        "r_squared": r2_score(predicted, target),
    }


def _factor_report(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    direction = np.asarray([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    return {
        **_metrics(predicted, target),
        "delta": _metrics(
            (predicted @ direction)[:, None],
            (target @ direction)[:, None],
        ),
    }


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
    return {
        "target": target_name,
        "fit": fit,
        **_factor_report(test_features @ weight + bias, test_target),
    }


def _observation_groups(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != JOINT_TOKEN_COUNT:
        raise ValueError("expected one delayed joint-token one-hot")
    return np.where(
        observations.sum(axis=1) > 0,
        observations.argmax(axis=1),
        JOINT_TOKEN_COUNT,
    )


def _branch_prediction(
    train: ProbeData,
    test: ProbeData,
    factor: int,
) -> np.ndarray:
    train_groups = _observation_groups(train.observations)
    test_groups = _observation_groups(test.observations)
    target = train.factor_beliefs[:, factor]
    predicted = np.broadcast_to(
        target.mean(axis=0),
        (len(test_groups), 3),
    ).copy()
    for group in np.unique(train_groups):
        predicted[test_groups == group] = target[train_groups == group].mean(axis=0)
    return predicted


def _policy_report(data: ProbeData) -> dict[str, Any]:
    count = len(data.actions)
    return {
        "mode": "learned_stochastic",
        "mean_reward": float(data.rewards.mean()),
        "token_accuracy": float(np.mean(data.actions == data.hidden_tokens)),
        "guess_fractions": (
            np.bincount(data.actions, minlength=JOINT_TOKEN_COUNT) / count
        ).tolist(),
        "hidden_token_fractions": (
            np.bincount(data.hidden_tokens, minlength=JOINT_TOKEN_COUNT) / count
        ).tolist(),
    }


def _analyze_samples(
    context: RunContext,
    *,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
    train: ProbeData,
    test: ProbeData,
    streams: Mapping[str, np.random.SeedSequence],
) -> dict[str, Any]:
    consistency = max(
        train.product_consistency_max_abs,
        test.product_consistency_max_abs,
    )
    if consistency > 1e-10:
        raise AssertionError(
            f"factor belief lost product structure: {consistency:.3e}"
        )
    regression_seed = seed_sequence_to_int(streams["regression"], bits=32)
    permutation = np.random.default_rng(streams["shuffle"]).permutation(
        len(train.activations)
    )
    layers = {}
    for layer in range(train.activations.shape[1]):
        factors = {}
        shuffled = {}
        for factor in range(FACTOR_COUNT):
            name = f"factor_{factor + 1}"
            arguments = {
                "test_features": test.activations[:, layer],
                "test_target": test.factor_beliefs[:, factor],
                "target_name": f"exact_delayed_{name}_arrival_belief",
                "seed": regression_seed + factor,
            }
            factors[name] = _fit_report(
                train.activations[:, layer],
                train.factor_beliefs[:, factor],
                **arguments,
            )
            shuffled[name] = _fit_report(
                train.activations[:, layer],
                train.factor_beliefs[permutation, factor],
                **arguments,
            )
        layers[f"layer_{layer + 1}"] = {
            "representation": (
                f"encoder.blocks.{layer}."
                "output_current_position_pre_final_layer_norm"
            ),
            "feature_width": int(train.activations.shape[2]),
            "probe_fits": factors,
            "shuffled_training_labels": shuffled,
            "cev": variance_geometry(test.activations[:, layer]),
        }
    controls = {
        "train_mean": {},
        "current_visible_token": {},
    }
    for factor in range(FACTOR_COUNT):
        name = f"factor_{factor + 1}"
        train_target = train.factor_beliefs[:, factor]
        test_target = test.factor_beliefs[:, factor]
        controls["train_mean"][name] = _factor_report(
            np.broadcast_to(train_target.mean(axis=0), test_target.shape),
            test_target,
        )
        controls["current_visible_token"][name] = _factor_report(
            _branch_prediction(train, test, factor),
            test_target,
        )
    final_layer = layers[f"layer_{train.activations.shape[1]}"]
    result = {
        "checkpoint": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "is_initialization": training_iteration == 0,
        "representation": (
            "each_block_current_position_residual_before_final_layer_norm"
        ),
        "n_fit": len(train.activations),
        "n_test": len(test.activations),
        "metadata": {
            "seed": context.seed,
            "smoke": context.smoke,
            "sampling_distribution": "process_weighted_rollout",
            "policy_mode": "learned_stochastic",
            "n_envs": N_ENVS,
            "warmup_per_episode": WARMUP,
            "context_length": CONTEXT_LENGTH,
            "episode_length": EPISODE_LENGTH,
            "full_train_budget": FULL_PROBE_TRAIN_STEPS,
            "full_test_budget": FULL_PROBE_TEST_STEPS,
            "smoke_budget_per_split": SMOKE_PROBE_STEPS,
            "sample_budgets_exclude_warmup": True,
            "independent_train_test_rollouts": True,
            "belief_conditioning": (
                "delayed_observed_tokens_only_never_actions_or_rewards"
            ),
            "belief_timing": "info.belief_current_before_current_guess",
            "guess_target": (
                "info.raw_token_current scored as event.raw_token_before "
                "on the completed step"
            ),
            "delta_definition": "(p0-p2)/sqrt(2)",
            "shuffle_control": (
                "shared training-row permutation scored on true test labels"
            ),
            "cv_scope": "training rows only; final test rollouts independent",
        },
        "product_consistency_max_abs": consistency,
        "layers": layers,
        "probe_fits": final_layer["probe_fits"],
        "controls": controls,
        "cev": {
            "actor_activation": final_layer["cev"],
            "joint_state_target": variance_geometry(test.joint_beliefs),
            **{
                f"factor_{index + 1}_target": variance_geometry(
                    test.factor_beliefs[:, index]
                )
                for index in range(FACTOR_COUNT)
            },
        },
        "policy": _policy_report(test),
        "train_policy": _policy_report(train),
        "scope_warning": (
            "Linear accessibility and CEV do not establish causal use."
        ),
    }
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "probe_battery.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def analyze_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = (
        SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TRAIN_STEPS
    )
    test_steps = (
        SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TEST_STEPS
    )
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        train = collect_probe_data(
            module,
            n_steps=train_steps,
            seed=streams["probe_train"],
            device=_device(context),
        )
        test = collect_probe_data(
            module,
            n_steps=test_steps,
            seed=streams["probe_test"],
            device=_device(context),
        )
    return _analyze_samples(
        context,
        checkpoint_label=checkpoint_label,
        agent_steps=agent_steps,
        training_iteration=training_iteration,
        train=train,
        test=test,
        streams=streams,
    )


__all__ = ["ProbeData", "analyze_checkpoint", "collect_probe_data"]


if __name__ == "__main__":
    import sys
    from experiments.wing_two_factor_explore_cycle_1.control_analysis import main

    main(["--study", "token_guess", *sys.argv[1:]])
