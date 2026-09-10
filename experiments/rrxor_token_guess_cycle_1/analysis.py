"""Held-out RRXOR belief probes for layerwise and distributed geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.belief_geometry import (
    evaluate_belief_geometry,
    prediction_null_basis,
)
from analysis.checkpoints import load_algorithm
from analysis.probes import variance_geometry
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.hmm import HMMEnv
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from experiments.rrxor_token_guess_cycle_1.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    RRXOR_STATIONARY,
    RRXOR_TENSOR,
    environment_config,
    filter_source_belief,
    next_token_probabilities,
    reachable_beliefs,
)


FULL_PROBE_TRAIN_STEPS = 20_000
FULL_PROBE_TEST_STEPS = 20_000
SMOKE_PROBE_STEPS = 512
N_ENVS = 8
SMOKE_BOOTSTRAP_RESAMPLES = 100
FULL_BOOTSTRAP_RESAMPLES = 200
SMOKE_NULL_REPEATS = 1
FULL_NULL_REPEATS = 5
_STREAM_KEYS = {
    "probe_train": (900,),
    "probe_test": (901,),
    "geometry": (902,),
}


@dataclass(frozen=True, slots=True)
class ProbeData:
    """Aligned residual activations and exact decision-time source beliefs."""

    layers: np.ndarray
    post_final_norm: np.ndarray
    policy_probabilities: np.ndarray
    beliefs: np.ndarray
    next_token_probabilities: np.ndarray
    observations: np.ndarray
    visible_tokens: np.ndarray
    hidden_tokens: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_steps: np.ndarray
    episode_ids: np.ndarray
    history_suffix_2: np.ndarray
    belief_ids: np.ndarray
    diagnostic_alignment_max_abs: float


def _device(context: RunContext) -> torch.device:
    profile = context.hardware
    return torch.device(
        "cuda"
        if profile is not None
        and profile.learner_device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )


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


class _ExactTargets:
    """Reconstruct filtered source beliefs from public delayed-token diagnostics."""

    def __init__(self, n_envs: int) -> None:
        self._n_envs = n_envs
        self._beliefs = np.broadcast_to(
            RRXOR_STATIONARY,
            (n_envs, len(RRXOR_STATIONARY)),
        ).copy()
        self._episode_counters = np.full(n_envs, -1, dtype=np.int64)
        self._suffixes = np.zeros(n_envs, dtype=np.int64)
        self._suffix_lengths = np.zeros(n_envs, dtype=np.int64)
        self._reachable = reachable_beliefs()

    def _belief_ids(self) -> np.ndarray:
        distances = np.max(
            np.abs(self._beliefs[:, None, :] - self._reachable[None, :, :]),
            axis=2,
        )
        identifiers = distances.argmin(axis=1)
        if float(distances[np.arange(self._n_envs), identifiers].max()) > 1e-10:
            raise AssertionError("reconstructed belief is outside the RRXOR MSP")
        return identifiers.astype(np.int64)

    def __call__(
        self,
        observations: np.ndarray,
        infos: list[Mapping[str, Any]],
        episode_steps: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        visible_tokens = np.full(self._n_envs, -1, dtype=np.int64)
        diagnostic_errors = np.zeros(self._n_envs, dtype=np.float64)
        for index, (info, episode_step) in enumerate(
            zip(infos, episode_steps)
        ):
            if int(episode_step) == 0:
                self._beliefs[index] = RRXOR_STATIONARY
                self._episode_counters[index] += 1
                self._suffixes[index] = 0
                self._suffix_lengths[index] = 0
                if info["visible_source_token"] is not None:
                    raise AssertionError("delay-one reset exposed the pending token")
            else:
                token = int(info["visible_source_token"])
                if token != int(info["visible_token_current"]):
                    raise AssertionError("RRXOR token presentation was unexpectedly scrambled")
                visible_tokens[index] = token
                self._beliefs[index] = filter_source_belief(
                    self._beliefs[index],
                    token,
                )
                self._suffixes[index] = (
                    (self._suffixes[index] * 2 + token) % 4
                )
                self._suffix_lengths[index] = min(
                    2,
                    self._suffix_lengths[index] + 1,
                )

            observation = np.asarray(observations[index], dtype=np.float64)
            if visible_tokens[index] < 0:
                expected_observation = np.zeros(2, dtype=np.float64)
            else:
                expected_observation = np.eye(2)[visible_tokens[index]]
            if not np.array_equal(observation, expected_observation):
                raise AssertionError(
                    "policy observation is not the single delayed-token feature"
                )

            arrival = self._beliefs[index] @ RRXOR_TENSOR.sum(axis=0)
            diagnostic = np.asarray(info["belief_current"], dtype=np.float64)
            diagnostic_errors[index] = float(
                np.max(np.abs(arrival - diagnostic))
            )

        return {
            "belief": self._beliefs.copy(),
            "next_token_probabilities": next_token_probabilities(self._beliefs),
            "visible_token": visible_tokens,
            "hidden_token": np.asarray(
                [info["raw_token_current"] for info in infos],
                dtype=np.int64,
            ),
            "episode_step": np.asarray(episode_steps, dtype=np.int64),
            "episode_id": (
                np.arange(self._n_envs, dtype=np.int64)
                + self._n_envs * self._episode_counters
            ),
            "history_suffix_2": (
                4 * self._suffix_lengths + self._suffixes
            ).astype(np.int64),
            "belief_id": self._belief_ids(),
            "diagnostic_error": diagnostic_errors,
        }


def _sample_actions(
    probabilities: np.ndarray,
    randomness: PolicyRandomness,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.asarray(
        [
            randomness.numpy.choice(len(row), p=row)
            for row in probabilities
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
    n_envs: int = N_ENVS,
) -> ProbeData:
    """Collect complete paper-length prefixes without discarding reset transients."""

    module = module.to(device).eval()
    blocks = module.encoder.blocks
    layer_count = len(blocks)
    layer_width = module.paper_config.d_model
    captured: dict[int, torch.Tensor] = {}

    def capture(index: int):
        def hook(block, inputs, output):
            del block, inputs
            captured[index] = output[:, -1, :].detach()

        return hook

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(
        state: dict[str, torch.Tensor],
        indices: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        fresh = initial_state(len(indices))
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    def step_adapter(
        observations: np.ndarray,
        state: dict[str, torch.Tensor],
        randomness: PolicyRandomness,
        action_spaces: Any,
    ):
        del action_spaces
        captured.clear()
        observation = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=device,
        )
        embedding, state_out = module.encode_step(observation, state)
        if set(captured) != set(range(layer_count)):
            raise RuntimeError("every paper transformer block must be captured")
        layers = torch.stack(
            [captured[index] for index in range(layer_count)],
            dim=1,
        )
        logits = module.action_distribution_inputs(embedding)
        probabilities = torch.softmax(logits, dim=-1)
        actions = _sample_actions(
            probabilities.cpu().numpy().astype(np.float64),
            randomness,
        )
        representation = torch.cat(
            [
                layers.reshape(len(observations), layer_count * layer_width),
                embedding,
                probabilities,
            ],
            dim=1,
        )
        return actions, state_out, representation.cpu().numpy()

    handles = []
    try:
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(capture(index)))
        collected = collect_batched_rollout_data(
            lambda: HMMEnv(environment_config()),
            step_adapter,
            _ExactTargets(n_envs),
            n_steps=n_steps,
            seed=seed,
            n_envs=n_envs,
            initial_state=initial_state,
            reset_state=reset_state,
            warmup=0,
            store_observations=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    if collected.observations is None:
        raise AssertionError("probe collection requires policy observations")
    representation = np.asarray(
        collected.representations,
        dtype=np.float64,
    )
    layer_end = layer_count * layer_width
    post_final_end = layer_end + layer_width
    layers = representation[:, :layer_end].reshape(
        len(representation),
        layer_count,
        layer_width,
    )
    post_final_norm = representation[:, layer_end:post_final_end]
    policy_probabilities = representation[:, post_final_end:]
    actions = np.asarray(collected.actions, dtype=np.int64).reshape(-1)
    hidden_tokens = np.asarray(
        collected.targets["hidden_token"],
        dtype=np.int64,
    )
    rewards = np.asarray(collected.rewards, dtype=np.float64)
    if not np.array_equal(
        rewards,
        (actions == hidden_tokens).astype(np.float64),
    ):
        raise AssertionError("token-guess rewards are not action-time aligned")
    observations = np.asarray(collected.observations, dtype=np.float64)
    if observations.shape[1:] != (2,):
        raise AssertionError("policy received features beyond one delayed token")
    return ProbeData(
        layers=layers,
        post_final_norm=post_final_norm,
        policy_probabilities=policy_probabilities,
        beliefs=np.asarray(collected.targets["belief"], dtype=np.float64),
        next_token_probabilities=np.asarray(
            collected.targets["next_token_probabilities"],
            dtype=np.float64,
        ),
        observations=observations,
        visible_tokens=np.asarray(
            collected.targets["visible_token"],
            dtype=np.int64,
        ),
        hidden_tokens=hidden_tokens,
        actions=actions,
        rewards=rewards,
        episode_steps=np.asarray(
            collected.targets["episode_step"],
            dtype=np.int64,
        ),
        episode_ids=np.asarray(
            collected.targets["episode_id"],
            dtype=np.int64,
        ),
        history_suffix_2=np.asarray(
            collected.targets["history_suffix_2"],
            dtype=np.int64,
        ),
        belief_ids=np.asarray(
            collected.targets["belief_id"],
            dtype=np.int64,
        ),
        diagnostic_alignment_max_abs=float(
            np.max(collected.targets["diagnostic_error"])
        ),
    )


def _policy_report(data: ProbeData) -> dict[str, Any]:
    count = len(data.actions)
    return {
        "mode": "learned_stochastic",
        "token_accuracy": float(data.rewards.mean()),
        "bayes_expected_accuracy": float(
            data.next_token_probabilities.max(axis=1).mean()
        ),
        "bayes_argmax_empirical_accuracy": float(
            np.mean(
                data.next_token_probabilities.argmax(axis=1)
                == data.hidden_tokens
            )
        ),
        "guess_fractions": (
            np.bincount(data.actions, minlength=2) / count
        ).tolist(),
        "hidden_token_fractions": (
            np.bincount(data.hidden_tokens, minlength=2) / count
        ).tolist(),
    }


def _one_hot_groups(groups: np.ndarray, width: int) -> np.ndarray:
    groups = np.asarray(groups, dtype=np.int64)
    if groups.ndim != 1 or (groups < 0).any() or (groups >= width).any():
        raise ValueError("group identifiers exceed the declared one-hot width")
    return np.eye(width, dtype=np.float64)[groups]


def _feature_maps(
    data: ProbeData,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    representations = {
        f"layer_{layer + 1}": data.layers[:, layer]
        for layer in range(data.layers.shape[1])
    }
    representations["concatenated_layers"] = data.layers.reshape(
        len(data.layers),
        -1,
    )
    representations["post_final_layer_norm"] = data.post_final_norm
    visible_groups = data.visible_tokens + 1
    next_groups = np.rint(
        data.next_token_probabilities[:, 1] * 12
    ).astype(np.int64)
    nuisance = {
        "current_visible_token": _one_hot_groups(visible_groups, 3),
        "two_visible_token_suffix": _one_hot_groups(
            data.history_suffix_2,
            12,
        ),
        "exact_next_token_distribution_group": _one_hot_groups(
            next_groups,
            13,
        ),
        "exact_next_token_probabilities": data.next_token_probabilities,
        "log_exact_next_token_probabilities": np.log(
            np.maximum(data.next_token_probabilities, 1e-12)
        ),
        "policy_probabilities": data.policy_probabilities,
    }
    return representations, nuisance


def analyze_samples(
    context: RunContext,
    *,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
    train: ProbeData,
    test: ProbeData,
    streams: Mapping[str, np.random.SeedSequence],
) -> dict[str, Any]:
    """Fit the preregistered probe family and compact controls."""

    if max(
        train.diagnostic_alignment_max_abs,
        test.diagnostic_alignment_max_abs,
    ) > 1e-10:
        raise AssertionError("exact source beliefs do not match public diagnostics")
    train_features, train_nuisance = _feature_maps(train)
    test_features, test_nuisance = _feature_maps(test)
    null_basis = prediction_null_basis(RRXOR_TENSOR.sum(axis=2).T)
    battery = evaluate_belief_geometry(
        train_features,
        test_features,
        train.beliefs,
        test.beliefs,
        train_groups=train.episode_ids,
        test_groups=test.episode_ids,
        nuisance_features={
            name: (features, test_nuisance[name])
            for name, features in train_nuisance.items()
        },
        contrasts={
            f"next_token_invisible_{index + 1}": null_basis[:, index]
            for index in range(null_basis.shape[1])
        },
        matched_keys=(train.history_suffix_2, test.history_suffix_2),
        seed=seed_sequence_to_int(streams["geometry"], bits=32),
        n_null_repeats=(
            SMOKE_NULL_REPEATS if context.smoke else FULL_NULL_REPEATS
        ),
        n_resamples=(
            SMOKE_BOOTSTRAP_RESAMPLES
            if context.smoke
            else FULL_BOOTSTRAP_RESAMPLES
        ),
    )
    result = {
        "checkpoint": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "is_initialization": training_iteration == 0,
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        "metadata": {
            "paper": "Shai et al., arXiv:2405.15943",
            "seed": context.seed,
            "smoke": context.smoke,
            "sampling_distribution": "process_weighted_paper_length_prefixes",
            "policy_mode": "learned_stochastic",
            "n_envs": N_ENVS,
            "warmup_per_episode": 0,
            "prefix_lengths": [0, CONTEXT_LENGTH],
            "context_length": CONTEXT_LENGTH,
            "episode_length": EPISODE_LENGTH,
            "short_reset_prefixes_retained": True,
            "independent_train_test_rollouts": True,
            "belief_timing": (
                "filtered edge-source belief after visible history and before "
                "the pending token"
            ),
            "diagnostic_cross_check": (
                "target @ transition_matrix equals public info.belief_current"
            ),
            "target_dtype": "float64",
            "layer_probe": "each block output before encoder final LayerNorm",
            "concatenated_probe": "all four block outputs; separate 256-wide fit",
            "post_final_norm_control": "64-wide policy embedding",
            "probe_fitting": "training-episode grouped SVD-cutoff CV",
            "bootstrap_clusters": "complete environment episodes",
            "full_probe_budget_per_split": FULL_PROBE_TRAIN_STEPS,
            "smoke_probe_budget_per_split": SMOKE_PROBE_STEPS,
        },
        "coverage": {
            "reachable_beliefs_total": len(reachable_beliefs()),
            "fit_beliefs_observed": int(len(np.unique(train.belief_ids))),
            "test_beliefs_observed": int(len(np.unique(test.belief_ids))),
            "fit_prefix_lengths": sorted(
                int(value) for value in np.unique(train.episode_steps)
            ),
            "test_prefix_lengths": sorted(
                int(value) for value in np.unique(test.episode_steps)
            ),
        },
        "target_variance_geometry": variance_geometry(test.beliefs),
        "representation_variance_geometry": {
            name: variance_geometry(features)
            for name, features in test_features.items()
        },
        "geometry": battery.report,
        "policy": _policy_report(test),
        "train_policy": _policy_report(train),
        "diagnostic_alignment_max_abs": max(
            train.diagnostic_alignment_max_abs,
            test.diagnostic_alignment_max_abs,
        ),
        "scope_warning": (
            "Held-out affine decodability does not establish causal policy use "
            "or unique identification of the internal model."
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
    """Restore a public Algorithm checkpoint and run independent probe streams."""

    if context.seed is None:
        raise ValueError("RRXOR belief probing requires a resolved seed")
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
    return analyze_samples(
        context,
        checkpoint_label=checkpoint_label,
        agent_steps=agent_steps,
        training_iteration=training_iteration,
        train=train,
        test=test,
        streams=streams,
    )


__all__ = [
    "ProbeData",
    "analyze_checkpoint",
    "analyze_samples",
    "collect_probe_data",
]
