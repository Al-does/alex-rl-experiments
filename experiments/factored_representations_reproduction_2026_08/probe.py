"""Checkpoint-independent sampling adapters for the paper's probe battery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.hmm import factor_marginals, product_distribution
from harness.seeding import SeedSource

from experiments.factored_representations_reproduction_2026_08.process import (
    FACTOR_CARDINALITY,
    encode_joint_tokens,
    joint_token_count,
    paper_mess3_model,
)


@dataclass(frozen=True, slots=True)
class FactorProbeData:
    """Aligned residual activations and exact joint/factor predictive vectors."""

    activations: np.ndarray
    joint_beliefs: np.ndarray
    factor_beliefs: np.ndarray
    observations: np.ndarray
    env_indices: np.ndarray
    episode_steps: np.ndarray
    episode_ids: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    product_consistency_max_abs: float


@dataclass(frozen=True, slots=True)
class VaryOneData:
    """Per-factor controlled activations and position-aware centering groups."""

    activations: dict[str, np.ndarray]
    groups: dict[str, np.ndarray]


def _initial_state(
    module: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        .to(device)
        for key, value in module.get_initial_state().items()
    }


def _episode_ids(
    env_indices: np.ndarray,
    episode_steps: np.ndarray,
) -> np.ndarray:
    identifiers = np.empty(len(env_indices), dtype=np.int64)
    next_identifier = 0
    for env_index in np.unique(env_indices):
        members = np.flatnonzero(env_indices == env_index)
        previous = -1
        current = next_identifier
        for member in members:
            step = int(episode_steps[member])
            if step <= previous:
                current += 1
            identifiers[member] = current
            previous = step
        next_identifier = current + 1
    return identifiers


@torch.inference_mode()
def collect_probe_data(
    module: Any,
    env_factory: Callable[[], Any],
    *,
    factor_count: int,
    n_steps: int,
    seed: SeedSource,
    n_envs: int = 8,
    warmup: int = 8,
    device: str | torch.device = "cpu",
) -> FactorProbeData:
    """Collect matched process-weighted histories using public diagnostics."""

    device = torch.device(device)
    module = module.to(device).eval()
    factor_sizes = (FACTOR_CARDINALITY,) * factor_count

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(
        state: dict[str, torch.Tensor],
        indices: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        fresh = _initial_state(module, len(indices), device)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index_tensor, fresh[key])
        return state

    def step_adapter(
        observations: np.ndarray,
        state: dict[str, torch.Tensor],
        randomness: PolicyRandomness,
        action_spaces: Any,
    ):
        del randomness, action_spaces
        observation_tensor = torch.from_numpy(observations).float().to(device)
        residual, state = module.encode_step_pre_final_norm(
            observation_tensor,
            state,
        )
        normalized = module.encoder.final_norm(residual)
        logits = module.action_distribution_inputs(normalized)
        actions = logits.argmax(dim=-1)
        return (
            actions.cpu().numpy(),
            state,
            residual.cpu().numpy(),
        )

    def target_adapter(
        observations: np.ndarray,
        infos: list[Mapping[str, Any]],
        episode_steps: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        joint = np.stack([info["belief_current"] for info in infos])
        marginals = np.stack(factor_marginals(joint, factor_sizes), axis=1)
        return {
            "joint_belief": joint,
            "factor_belief": marginals,
            "env_index": np.arange(len(infos), dtype=np.int64),
            "episode_step": np.asarray(episode_steps, dtype=np.int64),
        }

    collected = collect_batched_rollout_data(
        env_factory,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=n_envs,
        initial_state=initial_state,
        reset_state=reset_state,
        warmup=warmup,
        store_observations=True,
    )
    if collected.observations is None:
        raise AssertionError("probe collection requested observations")
    joint = np.asarray(collected.targets["joint_belief"], dtype=np.float64)
    factors = np.asarray(collected.targets["factor_belief"], dtype=np.float64)
    reconstructed = product_distribution(
        [factors[:, index, :] for index in range(factor_count)]
    )
    consistency = float(np.max(np.abs(joint - reconstructed)))
    env_indices = np.asarray(collected.targets["env_index"], dtype=np.int64)
    episode_steps = np.asarray(
        collected.targets["episode_step"],
        dtype=np.int64,
    )
    return FactorProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        joint_beliefs=joint,
        factor_beliefs=factors,
        observations=np.asarray(collected.observations, dtype=np.float32),
        env_indices=env_indices,
        episode_steps=episode_steps,
        episode_ids=_episode_ids(env_indices, episode_steps),
        actions=np.asarray(collected.actions, dtype=np.int64),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
        product_consistency_max_abs=consistency,
    )


def _sample_factor_sequences(
    count: int,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    model = paper_mess3_model()
    states = rng.choice(
        model.n_states,
        size=count,
        p=model.initial_distribution,
    )
    output = np.empty((count, length), dtype=np.int64)
    for position in range(length):
        draws = rng.random(count)
        cumulative_emissions = np.cumsum(model.emission_matrix[states], axis=1)
        output[:, position] = (
            draws[:, None] > cumulative_emissions
        ).sum(axis=1)
        transition_draws = rng.random(count)
        cumulative_transitions = np.cumsum(
            model.transition_matrix[states],
            axis=1,
        )
        states = (
            transition_draws[:, None] > cumulative_transitions
        ).sum(axis=1)
    return output


@torch.inference_mode()
def _sequence_activations(
    module: Any,
    joint_tokens: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    token_count = int(module.observation_space.shape[0])
    expected = joint_token_count(
        int(round(np.log(token_count) / np.log(FACTOR_CARDINALITY)))
    )
    if expected != token_count:
        raise ValueError("module observation width is not a supported joint alphabet")
    tokens = torch.as_tensor(joint_tokens, dtype=torch.long, device=device)
    observations = torch_functional.one_hot(
        tokens,
        num_classes=token_count,
    ).to(dtype=torch.float32)
    bos = torch.zeros(
        (len(tokens), 1, token_count),
        dtype=observations.dtype,
        device=device,
    )
    observations = torch.cat([bos, observations], dim=1)
    state = _initial_state(module, len(tokens), device)
    residuals = module.encode_chunks_pre_final_norm(
        state["ctx"],
        state["len"].reshape(-1),
        observations,
    )
    return residuals[:, 1:, :].cpu().numpy().astype(np.float64, copy=False)


@torch.inference_mode()
def collect_vary_one_data(
    module: Any,
    *,
    factor_count: int,
    frozen_contexts: int,
    realizations_per_context: int,
    sequence_length: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> VaryOneData:
    """Generate Appendix H vary-one datasets with per-position groups."""

    if frozen_contexts <= 0 or realizations_per_context <= 1:
        raise ValueError("vary-one analysis needs contexts and repeated realizations")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    device = torch.device(device)
    module = module.to(device).eval()
    root = np.random.SeedSequence(seed)
    factor_seeds = root.spawn(factor_count)
    activations: dict[str, np.ndarray] = {}
    groups: dict[str, np.ndarray] = {}

    for varied_factor, factor_seed in enumerate(factor_seeds):
        rng = np.random.default_rng(factor_seed)
        batches = []
        for _ in range(frozen_contexts):
            subtokens = np.empty(
                (
                    realizations_per_context,
                    sequence_length,
                    factor_count,
                ),
                dtype=np.int64,
            )
            for factor in range(factor_count):
                count = (
                    realizations_per_context
                    if factor == varied_factor
                    else 1
                )
                sequence = _sample_factor_sequences(count, sequence_length, rng)
                subtokens[:, :, factor] = (
                    sequence
                    if count == realizations_per_context
                    else sequence[0]
                )
            batches.append(encode_joint_tokens(subtokens))
        joint_tokens = np.concatenate(batches, axis=0)
        factor_activations = _sequence_activations(
            module,
            joint_tokens,
            device=device,
        ).reshape(-1, module.reproduction_config.d_model)
        fixed_context = np.repeat(
            np.arange(frozen_contexts),
            realizations_per_context * sequence_length,
        )
        position = np.tile(
            np.arange(sequence_length),
            frozen_contexts * realizations_per_context,
        )
        name = f"factor_{varied_factor}"
        activations[name] = factor_activations
        # The authors center each sequence position within each frozen context.
        groups[name] = fixed_context * sequence_length + position

    return VaryOneData(activations=activations, groups=groups)
