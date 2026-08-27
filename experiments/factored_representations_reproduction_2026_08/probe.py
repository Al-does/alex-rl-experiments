"""Checkpoint-independent sampling adapters for the paper's probe battery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
import time
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


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    bad_indices = (~finite).nonzero()[:16]
    bad_rows = (
        torch.unique(bad_indices[:, 0])[:16]
        if tensor.ndim and len(bad_indices)
        else torch.empty(0, dtype=torch.long, device=tensor.device)
    )
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "posinf_count": int(torch.isposinf(tensor).sum().item()),
        "neginf_count": int(torch.isneginf(tensor).sum().item()),
        "finite_abs_max": (
            float(finite_values.abs().max().item()) if finite_values.numel() else None
        ),
        "first_bad_indices": bad_indices.cpu().tolist(),
        "first_bad_rows": bad_rows.cpu().tolist(),
    }


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    bad_indices = np.argwhere(~finite)[:16]
    bad_rows = (
        np.unique(bad_indices[:, 0])[:16].tolist()
        if array.ndim and len(bad_indices)
        else []
    )
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nan_count": int(np.isnan(array).sum()),
        "posinf_count": int(np.isposinf(array).sum()),
        "neginf_count": int(np.isneginf(array).sum()),
        "finite_abs_max": (
            float(np.abs(finite_values).max()) if finite_values.size else None
        ),
        "first_bad_indices": bad_indices.tolist(),
        "first_bad_rows": bad_rows,
    }


def _contains_nonfinite(summary: Mapping[str, Any]) -> bool:
    return any(
        int(summary[key]) > 0
        for key in ("nan_count", "posinf_count", "neginf_count")
    )


def _module_state_summary(module: Any) -> dict[str, Any]:
    parameter_summaries = {
        name: _tensor_summary(parameter)
        for name, parameter in module.named_parameters()
    }
    buffer_summaries = {
        name: _tensor_summary(buffer) for name, buffer in module.named_buffers()
    }
    bad_parameters = {
        name: summary
        for name, summary in parameter_summaries.items()
        if _contains_nonfinite(summary)
    }
    bad_buffers = {
        name: summary
        for name, summary in buffer_summaries.items()
        if _contains_nonfinite(summary)
    }
    largest_parameter = max(
        parameter_summaries,
        key=lambda name: parameter_summaries[name]["finite_abs_max"] or 0.0,
        default=None,
    )
    return {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "training": bool(module.training),
        "optimized_module": hasattr(module, "_orig_mod"),
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "largest_parameter": (
            {
                "name": largest_parameter,
                "summary": parameter_summaries[largest_parameter],
            }
            if largest_parameter is not None
            else None
        ),
        "bad_parameters": bad_parameters,
        "bad_buffers": bad_buffers,
    }


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                }
            )
            + "\n"
        )


def _first_nonfinite_forward_stage(
    module: Any,
    observation: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    first_bad: dict[str, Any] | None = None
    last_finite: dict[str, Any] | None = None
    handles = []

    def inspect(name: str):
        def hook(_submodule: Any, inputs: tuple[Any, ...], output: Any) -> None:
            nonlocal first_bad, last_finite
            tensors = (
                [output]
                if isinstance(output, torch.Tensor)
                else [value for value in output if isinstance(value, torch.Tensor)]
                if isinstance(output, (tuple, list))
                else []
            )
            for index, tensor in enumerate(tensors):
                summary = _tensor_summary(tensor)
                record = {
                    "module": name,
                    "output_index": index,
                    "output": summary,
                }
                if _contains_nonfinite(summary):
                    if first_bad is None:
                        record["inputs"] = [
                            _tensor_summary(value)
                            for value in inputs
                            if isinstance(value, torch.Tensor)
                        ]
                        record["last_finite_stage"] = last_finite
                        first_bad = record
                else:
                    last_finite = record

        return hook

    for name, submodule in module.named_modules():
        if name:
            handles.append(submodule.register_forward_hook(inspect(name)))
    try:
        replay_residual, replay_state = module.encode_step_pre_final_norm(
            observation,
            state,
        )
        replay_normalized = module.encoder.final_norm(replay_residual)
        replay_logits = module.action_distribution_inputs(replay_normalized)
        replay = {
            "residual": _tensor_summary(replay_residual),
            "state": {
                key: _tensor_summary(value) for key, value in replay_state.items()
            },
            "normalized": _tensor_summary(replay_normalized),
            "logits": _tensor_summary(replay_logits),
        }
    finally:
        for handle in handles:
            handle.remove()
    return {
        "first_bad_stage": first_bad,
        "last_finite_stage": last_finite,
        "replay": replay,
    }


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
    module_summary = _module_state_summary(module)
    # region agent log
    _debug_log(
        "A,D,E",
        "probe.py:collect_probe_data:entry",
        "Probe collector runtime and module state",
        {
            "n_steps": n_steps,
            "n_envs": n_envs,
            "warmup": warmup,
            "device": str(device),
            "torch_version": torch.__version__,
            "inference_mode": torch.is_inference_mode_enabled(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "host_load_average": list(os.getloadavg()),
            "module": module_summary,
        },
    )
    # endregion
    if module_summary["bad_parameters"] or module_summary["bad_buffers"]:
        raise FloatingPointError(
            "probe module contains non-finite parameters or buffers; see debug.log"
        )
    factor_sizes = (FACTOR_CARDINALITY,) * factor_count
    inference_batches = 0

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
        nonlocal inference_batches
        del randomness, action_spaces
        observation_tensor = torch.from_numpy(observations).float().to(device)
        state_in = state
        residual, state_out = module.encode_step_pre_final_norm(
            observation_tensor,
            state_in,
        )
        normalized = module.encoder.final_norm(residual)
        logits = module.action_distribution_inputs(normalized)
        stage_summaries = {
            "observations": _tensor_summary(observation_tensor),
            "state_in": {
                key: _tensor_summary(value) for key, value in state_in.items()
            },
            "residual": _tensor_summary(residual),
            "state_out": {
                key: _tensor_summary(value) for key, value in state_out.items()
            },
            "normalized": _tensor_summary(normalized),
            "logits": _tensor_summary(logits),
        }
        if inference_batches == 0:
            # region agent log
            _debug_log(
                "B,D",
                "probe.py:collect_probe_data:step_adapter:first_batch",
                "First inference batch stage summaries",
                stage_summaries,
            )
            # endregion
        bad_stages = {
            name: summary
            for name, summary in stage_summaries.items()
            if (
                any(
                    _contains_nonfinite(item)
                    for item in summary.values()
                )
                if name in {"state_in", "state_out"}
                else _contains_nonfinite(summary)
            )
        }
        if bad_stages:
            replay = _first_nonfinite_forward_stage(
                module,
                observation_tensor,
                state_in,
            )
            # region agent log
            _debug_log(
                "A,B,D,E",
                "probe.py:collect_probe_data:step_adapter:nonfinite",
                "Non-finite inference batch detected",
                {
                    "inference_batch": inference_batches,
                    "bad_stages": bad_stages,
                    "observation_token_rows": observation_tensor.argmax(
                        dim=-1
                    ).cpu().tolist(),
                    "state_lengths": state_in["len"].reshape(-1).cpu().tolist(),
                    "replay_trace": replay,
                    "host_load_average": list(os.getloadavg()),
                },
            )
            # endregion
            raise FloatingPointError(
                "probe inference produced non-finite values; see debug.log"
            )
        inference_batches += 1
        actions = logits.argmax(dim=-1)
        return (
            actions.cpu().numpy(),
            state_out,
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
    collected_summary = _array_summary(collected.representations)
    # region agent log
    _debug_log(
        "C",
        "probe.py:collect_probe_data:assembled",
        "Collector assembled representation matrix",
        {
            "representations": collected_summary,
            "inference_batches": inference_batches,
        },
    )
    # endregion
    if _contains_nonfinite(collected_summary):
        raise FloatingPointError(
            "probe collector assembled non-finite representations; see debug.log"
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
    result = FactorProbeData(
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
    result_summary = _array_summary(result.activations)
    # region agent log
    _debug_log(
        "C",
        "probe.py:collect_probe_data:exit",
        "Final factor probe activation matrix",
        {"activations": result_summary},
    )
    # endregion
    if _contains_nonfinite(result_summary):
        raise FloatingPointError(
            "probe conversion produced non-finite activations; see debug.log"
        )
    return result


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
