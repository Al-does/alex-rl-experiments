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
    _episode_ids,
    _initial_state,
)
from experiments.wing_two_factor_explore_cycle_1.process import (
    ACTION_PAIRS,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    FACTOR_COUNT,
    JOINT_TOKEN_COUNT,
    WING_ALPHA,
    WING_X,
    decode_joint_indices,
    environment_config,
)
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int


FULL_PROBE_TRAIN_STEPS = 20_000
FULL_PROBE_TEST_STEPS = 20_000
SMOKE_PROBE_STEPS = 256
N_ENVS = 8
WARMUP = CONTEXT_LENGTH
POLICY_TEMPERATURE = 1.5
_STREAM_KEYS = {
    "probe_train": (700,),
    "probe_test": (701,),
    "regression": (702,),
    "shuffle": (703,),
}


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    joint_beliefs: np.ndarray
    factor_beliefs: np.ndarray
    observations: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_ids: np.ndarray
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
        "factor_belief": np.stack(factor_marginals(joint, (3, 3)), axis=1),
        "state": np.asarray([info["state_current"] for info in infos], dtype=np.int64),
        "env_index": np.arange(len(infos), dtype=np.int64),
        "episode_step": np.asarray(episode_steps, dtype=np.int64),
    }


def _sample_actions(logits: torch.Tensor, randomness: PolicyRandomness) -> np.ndarray:
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.asarray(
        [randomness.numpy.choice(len(row), p=row) for row in probabilities],
        dtype=np.int64,
    )


@torch.inference_mode()
def collect_probe_data(
    module: Any,
    *,
    condition: str,
    reward_state: int,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
    env_config: Mapping[str, Any] | None = None,
) -> ProbeData:
    config = (
        environment_config(condition, reward_state)
        if env_config is None
        else dict(env_config)
    )
    config["diagnostics"] = {"belief": True, "state": True}
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
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        residual, state_out = module.encode_step_pre_final_norm(tensor, state)
        if len(captured) != len(blocks):
            raise RuntimeError("every encoder block must execute once per probe step")
        layers = torch.stack([captured[index] for index in range(len(blocks))], dim=1)
        if layers.shape[0] != len(observations):
            raise RuntimeError("probe hooks must return one current-position row per environment")
        logits = module.action_distribution_inputs(module.encoder.final_norm(residual))
        return _sample_actions(logits, randomness), state_out, layers.cpu().numpy()

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
        raise AssertionError("probe collection requires current policy observations")
    joint = np.asarray(collected.targets["joint_belief"], dtype=np.float64)
    factors = np.asarray(collected.targets["factor_belief"], dtype=np.float64)
    reconstructed = product_distribution([factors[:, index] for index in range(FACTOR_COUNT)])
    steps = np.asarray(collected.targets["episode_step"], dtype=np.int64)
    return ProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        joint_beliefs=joint,
        factor_beliefs=factors,
        observations=np.asarray(collected.observations, dtype=np.float64),
        states=np.asarray(collected.targets["state"], dtype=np.int64),
        actions=np.asarray(collected.actions, dtype=np.int64).reshape(-1),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
        episode_ids=_episode_ids(collected.targets["env_index"], steps),
        episode_steps=steps,
        product_consistency_max_abs=float(np.max(np.abs(joint - reconstructed))),
    )


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    metrics = global_mse_metrics(predicted, target)
    return {
        **metrics,
        "normalized_mse": metrics["global_mse_ratio"],
        "rmse": float(np.sqrt(np.mean(np.square(predicted - target)))),
        "r_squared": r2_score(predicted, target),
    }


def _factor_report(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    direction = np.asarray([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    return {
        **_metrics(predicted, target),
        "delta": _metrics((predicted @ direction)[:, None], (target @ direction)[:, None]),
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
    weight, bias, fit = cross_validated_svd_affine(train_features, train_target, seed=seed)
    return {
        "target": target_name,
        "fit": fit,
        **_factor_report(test_features @ weight + bias, test_target),
    }


def _observation_groups(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != JOINT_TOKEN_COUNT + 3 * FACTOR_COUNT:
        raise ValueError("expected current joint-token and preceding per-factor action one-hots")
    tokens = observations[:, :JOINT_TOKEN_COUNT]
    actions = observations[:, JOINT_TOKEN_COUNT:].reshape(-1, FACTOR_COUNT, 3)
    token_ids = np.where(tokens.sum(axis=1) > 0, tokens.argmax(axis=1), JOINT_TOKEN_COUNT)
    factor_action_ids = actions.argmax(axis=2)
    action_ids = np.where(
        actions.sum(axis=(1, 2)) > 0,
        factor_action_ids[:, 0] * 3 + factor_action_ids[:, 1],
        len(ACTION_PAIRS),
    )
    return token_ids * (len(ACTION_PAIRS) + 1) + action_ids


def _branch_prediction(train: ProbeData, test: ProbeData, factor: int) -> np.ndarray:
    train_groups = _observation_groups(train.observations)
    test_groups = _observation_groups(test.observations)
    target = train.factor_beliefs[:, factor]
    predicted = np.broadcast_to(target.mean(axis=0), (len(test_groups), 3)).copy()
    for group in np.unique(train_groups):
        predicted[test_groups == group] = target[train_groups == group].mean(axis=0)
    return predicted


def _policy_report(data: ProbeData) -> dict[str, Any]:
    decoded_states = decode_joint_indices(data.states)
    decoded_actions = decode_joint_indices(data.actions)
    count = len(data.actions)
    return {
        "mode": "learned_stochastic",
        "temperature": POLICY_TEMPERATURE,
        "mean_reward": float(data.rewards.mean()),
        "joint_state_fractions": (np.bincount(data.states, minlength=9) / count).tolist(),
        "action_fractions": (np.bincount(data.actions, minlength=9) / count).tolist(),
        "factor_state_fractions": {
            f"factor_{index + 1}": (np.bincount(decoded_states[:, index], minlength=3) / count).tolist()
            for index in range(FACTOR_COUNT)
        },
        "factor_action_fractions": {
            f"factor_{index + 1}": (np.bincount(decoded_actions[:, index], minlength=3) / count).tolist()
            for index in range(FACTOR_COUNT)
        },
    }


def _analyze_samples(
    context: RunContext,
    *,
    condition: str,
    reward_state: int,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
    train: ProbeData,
    test: ProbeData,
    streams: Mapping[str, np.random.SeedSequence],
    emission_matrix: np.ndarray,
) -> dict[str, Any]:
    consistency = max(train.product_consistency_max_abs, test.product_consistency_max_abs)
    if consistency > 1e-10:
        raise AssertionError(f"factor belief lost product structure: {consistency:.3e}")
    emission_matrix = np.asarray(emission_matrix, dtype=np.float64)
    if emission_matrix.shape != (3, 2):
        raise ValueError("Wing next-token projection requires a (3, 2) emission matrix")
    regression_seed = seed_sequence_to_int(streams["regression"], bits=32)
    permutation = np.random.default_rng(streams["shuffle"]).permutation(len(train.activations))
    layers = {}
    for layer in range(train.activations.shape[1]):
        factors = {}
        shuffled = {}
        for factor in range(FACTOR_COUNT):
            name = f"factor_{factor + 1}"
            arguments = {
                "test_features": test.activations[:, layer],
                "test_target": test.factor_beliefs[:, factor],
                "target_name": f"exact_action_conditioned_{name}_predictive_belief",
                "seed": regression_seed + factor,
            }
            factors[name] = _fit_report(
                train.activations[:, layer], train.factor_beliefs[:, factor], **arguments
            )
            shuffled[name] = _fit_report(
                train.activations[:, layer], train.factor_beliefs[permutation, factor], **arguments
            )
        layers[f"layer_{layer + 1}"] = {
            "representation": f"encoder.blocks.{layer}.output_current_position_pre_final_layer_norm",
            "feature_width": int(train.activations.shape[2]),
            "probe_fits": factors,
            "shuffled_training_labels": shuffled,
            "cev": variance_geometry(test.activations[:, layer]),
        }
    controls = {name: {} for name in ("train_mean", "current_token_preceding_action", "ntp", "log_ntp")}
    for factor in range(FACTOR_COUNT):
        name = f"factor_{factor + 1}"
        train_target = train.factor_beliefs[:, factor]
        test_target = test.factor_beliefs[:, factor]
        controls["train_mean"][name] = _factor_report(
            np.broadcast_to(train_target.mean(axis=0), test_target.shape), test_target
        )
        controls["current_token_preceding_action"][name] = _factor_report(
            _branch_prediction(train, test, factor), test_target
        )
        train_ntp = train_target @ emission_matrix
        test_ntp = test_target @ emission_matrix
        for control in ("ntp", "log_ntp"):
            controls[control][name] = _fit_report(
                train_ntp if control == "ntp" else np.log(np.clip(train_ntp, 1e-12, None)),
                train_target,
                test_ntp if control == "ntp" else np.log(np.clip(test_ntp, 1e-12, None)),
                test_target,
                target_name=f"exact_action_conditioned_{name}_predictive_belief",
                seed=regression_seed + factor,
            )
    final_layer = layers[f"layer_{train.activations.shape[1]}"]
    result = {
        "condition": condition,
        "reward_state": reward_state,
        "checkpoint": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "is_initialization": training_iteration == 0,
        "representation": "each_block_current_position_residual_before_final_layer_norm",
        "n_fit": len(train.activations),
        "n_test": len(test.activations),
        "metadata": {
            "seed": context.seed,
            "smoke": context.smoke,
            "sampling_distribution": "process_weighted_rollout",
            "policy_mode": "learned_stochastic",
            "temperature": POLICY_TEMPERATURE,
            "logits_already_temperature_scaled": True,
            "n_envs": N_ENVS,
            "warmup_per_episode": WARMUP,
            "context_length": CONTEXT_LENGTH,
            "episode_length": EPISODE_LENGTH,
            "full_train_budget": FULL_PROBE_TRAIN_STEPS,
            "full_test_budget": FULL_PROBE_TEST_STEPS,
            "smoke_budget_per_split": SMOKE_PROBE_STEPS,
            "sample_budgets_exclude_warmup": True,
            "independent_train_test_rollouts": True,
            "rollout_seed_spawn_keys": {name: list(streams[name].spawn_key) for name in ("probe_train", "probe_test")},
            "train_episodes_represented": int(len(np.unique(train.episode_ids))),
            "test_episodes_represented": int(len(np.unique(test.episode_ids))),
            "belief_conditioning": "observed_tokens_and_executed_actions_only_never_rewards",
            "belief_timing": "info.belief_current_before_current_action",
            "delta_definition": "(p0-p2)/sqrt(2)",
            "delta_decoder": "contrast_of_affine_factor_belief_prediction",
            "normalized_mse_definition": "mse/test_target_variance",
            "ntp_definition": "factor_belief @ wing_model.emission_matrix",
            "ntp_emission_matrix": emission_matrix.tolist(),
            "log_ntp_floor": 1e-12,
            "branch_baseline": "train_only_token_action_cell_means_with_train_global_mean_for_unseen_cells",
            "shuffle_control": "one_shared_training_row_permutation_scored_on_true_test_labels",
            "cv_scope": "training_rows_only; final_test_rollouts_independent",
        },
        "product_consistency_max_abs": consistency,
        "layers": layers,
        "probe_fits": final_layer["probe_fits"],
        "controls": controls,
        "cev": {
            "actor_activation": final_layer["cev"],
            "joint_mixed_state_target": variance_geometry(test.joint_beliefs),
            **{f"factor_{index + 1}_target": variance_geometry(test.factor_beliefs[:, index]) for index in range(FACTOR_COUNT)},
        },
        "policy": _policy_report(test),
        "train_policy": _policy_report(train),
        "scope_warning": "Linear accessibility and CEV do not establish causal use or orthogonal factor subspaces; learned-policy occupancy varies across checkpoints.",
    }
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "probe_battery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def analyze_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    reward_state: int,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
) -> dict[str, Any]:
    from envs.wing.model import wing_model

    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TRAIN_STEPS
    test_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TEST_STEPS
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        train = collect_probe_data(
            module, condition=condition, reward_state=reward_state,
            n_steps=train_steps, seed=streams["probe_train"], device=_device(context),
        )
        test = collect_probe_data(
            module, condition=condition, reward_state=reward_state,
            n_steps=test_steps, seed=streams["probe_test"], device=_device(context),
        )
    return _analyze_samples(
        context, condition=condition, reward_state=reward_state,
        checkpoint_label=checkpoint_label, agent_steps=agent_steps,
        training_iteration=training_iteration, train=train, test=test,
        streams=streams,
        emission_matrix=wing_model(alpha=WING_ALPHA, x=WING_X).emission_matrix,
    )


__all__ = ["analyze_checkpoint", "collect_probe_data", "ProbeData"]
