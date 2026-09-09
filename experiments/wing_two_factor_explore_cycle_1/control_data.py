from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from envs.hmm import HMMEnv, factor_marginals
from envs.wing.model import controlled_kernels, wing_model
from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from harness.seeding import (
    child_seed_sequence,
    named_seed_sequences,
    seed_sequence_to_int,
)


@dataclass(frozen=True, slots=True)
class ControlData:
    activations: np.ndarray
    beliefs: np.ndarray
    policy_probabilities: np.ndarray
    observations: np.ndarray
    episode_ids: np.ndarray
    episode_steps: np.ndarray
    mask: np.ndarray
    tokens: np.ndarray
    preceding_actions: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@contextmanager
def _representation_step(module: Any, device: torch.device):
    training = module.training
    handles = []
    captured = {}
    blocks = module.encoder.blocks

    def capture(index):
        def hook(block, inputs, output):
            captured[index] = output[:, -1, :].detach()
        return hook

    def forward(observations, state):
        captured.clear()
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        residual, state_out = module.encode_step_pre_final_norm(tensor, state)
        if len(captured) != len(blocks):
            raise RuntimeError("every encoder block must execute during representation replay")
        layers = torch.stack([captured[index] for index in range(len(blocks))], dim=1)
        logits = module.action_distribution_inputs(module.encoder.final_norm(residual))
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return layers.cpu().numpy(), probabilities, state_out

    try:
        module.to(device).eval()
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(capture(index)))
        yield forward
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        module.train(training)


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@torch.inference_mode()
def collect_control_data(
    module: Any,
    *,
    env_config: Mapping,
    n_steps: int,
    seed: int,
    device: torch.device,
    n_envs: int = 8,
    warmup: int = 32,
) -> ControlData:
    n_steps = _integer(n_steps, "n_steps", 1)
    n_envs = min(_integer(n_envs, "n_envs", 1), n_steps)
    warmup = _integer(warmup, "warmup", 0)
    config = deepcopy(dict(env_config))
    if warmup >= config.get("episode_length", 1024):
        raise ValueError("warmup must be less than episode_length")
    config["diagnostics"] = {"belief": True, "tokens": True, "transitions": True}
    streams = named_seed_sequences(seed, {"environment": (0,), "policy": (1,)})
    policy_rngs = [
        np.random.default_rng(child_seed_sequence(streams["policy"], index))
        for index in range(n_envs)
    ]
    envs = []
    observations, infos = [], []
    episode_indices = np.zeros(n_envs, dtype=np.int64)
    episode_ids = np.arange(n_envs, dtype=np.int64)
    next_episode_id = n_envs
    rows = {name: [] for name in (
        "activations", "beliefs", "policy_probabilities", "observations",
        "episode_ids", "episode_steps", "mask", "tokens", "preceding_actions",
    )}

    def reset(index):
        episode_seed = seed_sequence_to_int(child_seed_sequence(
            streams["environment"], (index, int(episode_indices[index])),
        ))
        return envs[index].reset(seed=episode_seed)

    scored = 0
    try:
        for index in range(n_envs):
            env = HMMEnv(config)
            envs.append(env)
            passive = env.config.delay == 1
            expected_shape = (4,) if passive else (10,)
            if (
                env.model.n_states != 9 or env.model.n_tokens != 4
                or env.observation_space.shape != expected_shape
                or env.action_space.n != (4 if passive else 9)
                or env.config.observation.token_scrambling != "none"
                or env.config.observation.action_scrambling != "none"
            ):
                raise ValueError("control collection requires an unscrambled two-factor Wing recipe")
            observation, info = reset(index)
            observations.append(observation)
            infos.append(info)
        with _representation_step(module, device) as forward:
            state = _initial_state(module, n_envs, device)
            while scored < n_steps:
                batch = np.stack(observations)
                layers, probabilities, state = forward(batch, state)
                steps = np.asarray([info["decision_step"] for info in infos], dtype=np.int64)
                eligible = steps >= warmup
                remaining = n_steps - scored
                eligible_indices = np.flatnonzero(eligible)
                take = int(eligible_indices[remaining - 1]) + 1 if len(eligible_indices) >= remaining else n_envs
                joint_beliefs = np.stack([info["belief_current"] for info in infos])
                beliefs = np.stack(factor_marginals(joint_beliefs, (3, 3)), axis=1)
                tokens = np.full((n_envs, 2), -1, dtype=np.int64)
                preceding = np.full((n_envs, 2), -1, dtype=np.int64)
                for index, info in enumerate(infos):
                    token = info["visible_token_current"]
                    if token is not None:
                        tokens[index] = divmod(int(token), 2)
                    if steps[index] > 0:
                        preceding[index] = divmod(int(info["executed_action"]), 2 if passive else 3)
                values = (
                    layers, beliefs, probabilities, batch, episode_ids.copy(),
                    steps, eligible, tokens, preceding,
                )
                for name, value in zip(rows, values):
                    rows[name].append(value[:take].copy())
                scored += int(eligible[:take].sum())
                if scored == n_steps:
                    break
                reset_indices = []
                for index, env in enumerate(envs):
                    action = int(policy_rngs[index].choice(env.action_space.n, p=probabilities[index]))
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        episode_indices[index] += 1
                        episode_ids[index] = next_episode_id
                        next_episode_id += 1
                        observation, info = reset(index)
                        reset_indices.append(index)
                    observations[index], infos[index] = observation, info
                if reset_indices:
                    fresh = _initial_state(module, len(reset_indices), device)
                    indices = torch.as_tensor(reset_indices, dtype=torch.long, device=device)
                    for key, value in state.items():
                        value.index_copy_(0, indices, fresh[key])
    finally:
        for env in envs:
            env.close()
    return ControlData(
        **{name: np.concatenate(chunks, axis=0) for name, chunks in rows.items()},
        metadata={
            "env_config": deepcopy(dict(env_config)),
            "delay": 1 if passive else 0,
            "n_envs": n_envs,
            "n_steps": n_steps,
            "warmup": warmup,
            "suffix_zero_prior": "uniform source then transition" if passive else "uniform current",
            "ntp_timing": "pending hidden token from source" if passive else "next emitted token from current",
        },
    )


def _episode_members(data: ControlData) -> list[np.ndarray]:
    count = len(data.episode_ids)
    if np.shape(data.episode_steps) != (count,):
        raise ValueError("episode IDs and steps must be aligned")
    groups = {}
    for index, episode in enumerate(data.episode_ids):
        groups.setdefault(int(episode), []).append(index)
    members = [np.asarray(group, dtype=np.int64) for group in groups.values()]
    for indices in members:
        if not np.array_equal(data.episode_steps[indices], np.arange(len(indices))):
            raise ValueError("each episode must contain its complete ordered history starting at reset")
    return members


def replay_beliefs(
    data: ControlData,
    *,
    alpha: float,
    x: float,
    strength: float | None,
    suffix_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if suffix_length is not None:
        suffix_length = _integer(suffix_length, "suffix_length", 0)
    model = wing_model(alpha, x)
    edges = model.edge_transition_matrices
    passive = strength is None
    kernels = None if passive else controlled_kernels(alpha, x, strength)
    if kernels is not None and not np.allclose(
        kernels.sum(axis=-1), model.emission_matrix.T[None, :, :], atol=1e-12, rtol=0.0,
    ):
        raise AssertionError("Wing controls must preserve all action-conditional emission probabilities")
    count = len(data.episode_ids)
    if np.shape(data.tokens) != (count, 2) or np.shape(data.preceding_actions) != (count, 2):
        raise ValueError("tokens and preceding actions must have shape (n, 2)")
    if not np.issubdtype(data.tokens.dtype, np.integer) or not np.issubdtype(data.preceding_actions.dtype, np.integer):
        raise ValueError("tokens and preceding actions must be integers")
    groups = _episode_members(data)
    reset_rows = data.episode_steps == 0
    blank_rows = reset_rows & passive
    if (data.tokens[blank_rows] != -1).any() or ((data.tokens[~blank_rows] < 0) | (data.tokens[~blank_rows] > 1)).any():
        raise ValueError("only passive reset rows may have blank tokens")
    if (data.preceding_actions[reset_rows] != -1).any():
        raise ValueError("reset rows must have blank preceding actions")
    if not passive and ((data.preceding_actions[~reset_rows] < 0) | (data.preceding_actions[~reset_rows] > 2)).any():
        raise ValueError("controlled preceding actions must lie in [0, 3)")
    operators = np.broadcast_to(np.eye(3), (count, 2, 3, 3)).copy()
    if passive:
        operators[~blank_rows] = edges[data.tokens[~blank_rows]]
    else:
        operators[reset_rows] = edges[data.tokens[reset_rows]]
        operators[~reset_rows] = kernels[data.preceding_actions[~reset_rows], data.tokens[~reset_rows]]
    sources = np.empty((count, 2, 3), dtype=np.float64)

    def condition(belief, operator):
        updated = np.einsum("...i,...ij->...j", belief, operator)
        total = updated.sum(axis=-1, keepdims=True)
        if (total <= 0).any():
            raise ValueError("history contains an impossible token under the candidate parameters")
        return updated / total

    for indices in groups:
        if suffix_length is None:
            belief = np.full((2, 3), 1.0 / 3.0)
            for index in indices:
                belief = condition(belief, operators[index])
                sources[index] = belief
        else:
            beliefs = np.full((len(indices), 2, 3), 1.0 / 3.0)
            for offset in range(min(suffix_length, len(indices)) - 1, -1, -1):
                beliefs[offset:] = condition(beliefs[offset:], operators[indices[:len(indices) - offset]])
            sources[indices] = beliefs
    beliefs = sources @ model.transition_matrix if passive else sources
    ntp = sources @ model.emission_matrix
    return beliefs, ntp


def candidate_parameters(alpha: float, x: float, strength: float | None) -> list[dict]:
    wing_model(alpha, x)
    if strength is not None:
        controlled_kernels(alpha, x, strength)
    offsets = [
        (-0.12, 0.0), (-0.06, 0.0), (-0.025, 0.0), (0.02, 0.0),
        (0.0, -0.18), (0.0, -0.08), (0.0, 0.08), (0.0, 0.18),
        (-0.06, -0.12), (-0.06, 0.12), (0.02, -0.12), (0.02, 0.12),
    ]
    if strength is not None:
        offsets = offsets[:8]
    candidates = []

    def append(a, value_x, value_strength):
        candidate = {
            "alpha": float(np.clip(a, 0.01, 0.995)),
            "x": float(np.clip(value_x, 0.01, 0.99)),
            "strength": value_strength,
        }
        if candidate != {"alpha": alpha, "x": x, "strength": strength} and candidate not in candidates:
            candidates.append(candidate)

    for delta_alpha, delta_x in offsets:
        append(alpha + delta_alpha, x + delta_x, strength)
    if strength is not None:
        for alternative in (strength - 0.4, strength - 0.2, strength + 0.2, strength + 0.4, 0.0, 0.5, 1.0):
            append(alpha, x, float(np.clip(alternative, 0.0, 1.0)))
            if len(candidates) >= 12:
                break
    return candidates


def shuffled_history(data: ControlData, *, seed: int) -> ControlData:
    rng = np.random.default_rng(seed)
    order = np.arange(len(data.episode_ids))
    for indices in _episode_members(data):
        order[indices[1:]] = rng.permutation(indices[1:])
    return replace(
        data,
        observations=data.observations[order].copy(),
        tokens=data.tokens[order].copy(),
        preceding_actions=data.preceding_actions[order].copy(),
        metadata={
            **data.metadata,
            "history_control": "joint within-episode permutation preserving reset",
            "history_shuffle_seed": int(seed),
            "cached_targets_and_activations": "original history; recompute with replay APIs",
        },
    )


@torch.inference_mode()
def replay_activations(
    module: Any,
    data: ControlData,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    groups = _episode_members(data)
    if not groups:
        raise ValueError("activation replay requires at least one episode")
    batch_size = _integer(data.metadata.get("n_envs", 8), "n_envs", 1)
    activations, probabilities = None, None
    with _representation_step(module, device) as forward:
        for start in range(0, len(groups), batch_size):
            batch_groups = groups[start:start + batch_size]
            state = _initial_state(module, len(batch_groups), device)
            for step in range(max(map(len, batch_groups))):
                indices = np.asarray([group[min(step, len(group) - 1)] for group in batch_groups])
                layers, policy, state = forward(data.observations[indices], state)
                if activations is None:
                    activations = np.empty((len(data.episode_ids), *layers.shape[1:]), dtype=layers.dtype)
                    probabilities = np.empty((len(data.episode_ids), *policy.shape[1:]), dtype=policy.dtype)
                active = np.asarray([step < len(group) for group in batch_groups])
                activations[indices[active]] = layers[active]
                probabilities[indices[active]] = policy[active]
    return activations, probabilities
