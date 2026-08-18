"""Cassandra representation and belief-target adapters for affine probes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.cassandra_machine import (
    N_COMPONENTS,
    N_CONDITIONS,
    N_OBSERVATIONS,
    N_STATES,
    Action,
    decode_state,
    observation_matrix,
    reward_vector,
    transition_matrix,
)
from harness.seeding import SeedSource


# A checkpoint-independent policy keeps init/trained probe histories matched.
BEHAVIOR_ACTION_PROBABILITIES = np.array(
    [0.68, 0.20, 0.08, 0.04],
    dtype=np.float64,
)
CONDITION_VALUES = np.arange(N_CONDITIONS, dtype=np.float64)
CONTRAST = np.linalg.qr(
    np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
)[0]
_STATE_COMPONENTS = np.stack(
    [decode_state(state) for state in range(N_STATES)]
)
_MARGINAL_INDICATORS = np.zeros(
    (N_STATES, N_COMPONENTS * N_CONDITIONS),
    dtype=np.float64,
)
for _state, _components in enumerate(_STATE_COMPONENTS):
    for _component, _condition in enumerate(_components):
        _MARGINAL_INDICATORS[
            _state,
            _component * N_CONDITIONS + int(_condition),
        ] = 1.0
_BROKEN_COUNT_INDICATORS = np.eye(N_COMPONENTS + 1)[
    (_STATE_COMPONENTS == 0).sum(axis=1)
]
_ACTION_REWARDS = np.stack(
    [reward_vector(action) for action in Action],
    axis=1,
)
_NEXT_OPERATE_PASS = (
    transition_matrix(Action.OPERATE)
    @ observation_matrix(Action.OPERATE)[:, N_OBSERVATIONS - 1]
)


@dataclass(frozen=True, slots=True)
class CassandraProbeData:
    """Aligned paper-comparable activations and public environment targets."""

    activations: np.ndarray
    observations: np.ndarray
    targets: dict[str, np.ndarray]
    env_indices: np.ndarray
    episode_steps: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    marginal_consistency_max_abs: float


def belief_targets(
    joint_belief: np.ndarray,
    factored_belief: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive coarse, factored, and control-relevant targets from beliefs."""

    joint = np.asarray(joint_belief, dtype=np.float64)
    marginals = np.asarray(factored_belief, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != N_STATES:
        raise ValueError(f"joint_belief must have shape (N, {N_STATES})")
    if marginals.shape != (len(joint), N_COMPONENTS, N_CONDITIONS):
        raise ValueError(
            "factored_belief must have shape "
            f"(N, {N_COMPONENTS}, {N_CONDITIONS})"
        )

    component_contrast = np.einsum(
        "nic,ck->nik",
        marginals,
        CONTRAST,
    )
    aggregate = marginals.mean(axis=1)
    aggregate_contrast = aggregate @ CONTRAST
    identity_deviation = np.einsum(
        "nic,ck->nik",
        marginals - aggregate[:, None, :],
        CONTRAST,
    )
    expected_condition = marginals @ CONDITION_VALUES

    independent = np.ones_like(joint)
    for component in range(N_COMPONENTS):
        independent *= marginals[
            :,
            component,
            _STATE_COMPONENTS[:, component],
        ]
    total_correlation = np.sum(
        joint
        * (
            np.log(np.maximum(joint, 1e-300))
            - np.log(np.maximum(independent, 1e-300))
        ),
        axis=1,
        keepdims=True,
    )

    return {
        "joint_belief": joint,
        "component_contrast": component_contrast.reshape(len(joint), -1),
        "identity_deviation": identity_deviation.reshape(len(joint), -1),
        "aggregate_contrast": aggregate_contrast,
        "labeled_expected_condition": expected_condition,
        "sorted_expected_condition": np.sort(expected_condition, axis=1),
        "next_operate_pass_probability": (
            joint @ _NEXT_OPERATE_PASS
        )[:, None],
        "expected_action_reward": joint @ _ACTION_REWARDS,
        "broken_count_distribution": joint @ _BROKEN_COUNT_INDICATORS,
        "total_correlation": total_correlation,
    }


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
    env_factory: Callable[[], Any],
    *,
    n_steps: int,
    seed: SeedSource,
    n_envs: int = 16,
    device: str | torch.device = "cpu",
    warmup: int = 64,
) -> CassandraProbeData:
    """Collect matched-history pre-final-LayerNorm Cassandra activations."""

    device = torch.device(device)
    module = module.to(device).eval()
    stateful = module.is_stateful()

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices: np.ndarray):
        fresh = _initial_state(module, len(indices), device)
        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=device,
        )
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
        if stateful:
            embedding, state = module.encode_step_pre_final_norm(
                observation_tensor,
                state,
            )
        else:
            embedding, _ = module.encode_step(observation_tensor)
        actions = randomness.numpy.choice(
            len(Action),
            size=len(observations),
            p=BEHAVIOR_ACTION_PROBABILITIES,
        )
        return actions, state, embedding.cpu().numpy()

    def target_adapter(
        observations: np.ndarray,
        infos: list[Mapping[str, Any]],
        episode_steps: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        del observations
        joint = np.stack([info["belief_current"] for info in infos])
        marginals = np.stack(
            [info["factored_belief_current"] for info in infos]
        )
        targets = belief_targets(joint, marginals)
        targets.update(
            {
                "diagnostic_marginals": marginals.reshape(len(infos), -1),
                "env_indices": np.arange(len(infos), dtype=np.int64),
                "episode_steps": np.asarray(episode_steps, dtype=np.int64),
            }
        )
        return targets

    collected = collect_batched_rollout_data(
        env_factory,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=n_envs,
        initial_state=initial_state if stateful else None,
        reset_state=reset_state if stateful else None,
        warmup=warmup,
        store_observations=True,
    )
    if collected.observations is None:
        raise AssertionError("probe collection requested policy observations")
    targets = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in collected.targets.items()
        if name not in {"env_indices", "episode_steps", "diagnostic_marginals"}
    }
    inferred_marginals = (
        targets["joint_belief"] @ _MARGINAL_INDICATORS
    )
    diagnostic_marginals = np.asarray(
        collected.targets["diagnostic_marginals"],
        dtype=np.float64,
    )
    consistency = float(
        np.max(np.abs(inferred_marginals - diagnostic_marginals))
    )

    return CassandraProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        observations=np.asarray(collected.observations, dtype=np.float64),
        targets=targets,
        env_indices=np.asarray(
            collected.targets["env_indices"],
            dtype=np.int64,
        ),
        episode_steps=np.asarray(
            collected.targets["episode_steps"],
            dtype=np.int64,
        ),
        actions=np.asarray(collected.actions, dtype=np.int64),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
        marginal_consistency_max_abs=consistency,
    )
