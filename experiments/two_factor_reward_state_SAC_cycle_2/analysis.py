"""Belief probes for the cycle-2 actor representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from analysis.checkpoints import load_algorithm
from analysis.probes import variance_geometry
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.hmm import HMMEnv, factor_marginals, product_distribution
from experiments.two_factor_reward_state_SAC_cycle_1.analysis import (
    FULL_PROBE_TEST_STEPS,
    FULL_PROBE_TRAIN_STEPS,
    N_ENVS,
    ProbeData,
    SMOKE_PROBE_STEPS,
    WARMUP,
    _fit_report,
    plot_probe_trajectory,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    FACTOR_CARDINALITY,
    FACTOR_COUNT,
    decode_joint_indices,
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import N_ACTIONS
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int


_STREAM_KEYS = {
    "probe_train": (700,),
    "probe_test": (701,),
    "regression_joint": (702,),
    "regression_factor_1": (703,),
    "regression_factor_2": (704,),
}


def _device(context: RunContext) -> torch.device:
    profile = context.hardware
    return torch.device(
        "cuda"
        if profile is not None
        and profile.learner_device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )


def _probe_data(collected: Any) -> ProbeData:
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


def _target_adapter(
    observations: np.ndarray,
    infos: list[Mapping[str, Any]],
    episode_steps: np.ndarray,
) -> Mapping[str, np.ndarray]:
    del observations, episode_steps
    joint = np.stack([info["belief_current"] for info in infos])
    factors = np.stack(
        factor_marginals(joint, (FACTOR_CARDINALITY,) * FACTOR_COUNT),
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


@torch.inference_mode()
def collect_probe_data(
    module: Any,
    *,
    condition: str,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
) -> ProbeData:
    config = environment_config(condition)
    config["diagnostics"] = {"belief": True, "state": True}
    module = module.to(device).eval()

    def step_adapter(observations, state, randomness, action_spaces):
        del state, randomness, action_spaces
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        residual = module.actor_hidden(tensor)
        logits = module.action_distribution_inputs(module.encoder.final_norm(residual))
        return logits.argmax(dim=-1).cpu().numpy(), None, residual.cpu().numpy()

    collected = collect_batched_rollout_data(
        lambda: HMMEnv(config),
        step_adapter,
        _target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=N_ENVS,
        warmup=WARMUP,
    )
    return _probe_data(collected)


def _analyze_samples(
    context: RunContext,
    *,
    condition: str,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
    train: ProbeData,
    test: ProbeData,
    streams: Mapping[str, np.random.SeedSequence],
    representation: str,
) -> dict[str, Any]:
    consistency = max(
        train.product_consistency_max_abs,
        test.product_consistency_max_abs,
    )
    if consistency > 1e-10:
        raise AssertionError(f"factor belief lost product structure: {consistency:.3e}")
    fits = {}
    targets = {
        "joint_mixed_state": (train.joint_beliefs, test.joint_beliefs),
        "factor_1": (train.factor_beliefs[:, 0], test.factor_beliefs[:, 0]),
        "factor_2": (train.factor_beliefs[:, 1], test.factor_beliefs[:, 1]),
    }
    for name, (train_target, test_target) in targets.items():
        fits[name] = _fit_report(
            train.activations,
            train_target,
            test.activations,
            test_target,
            target_name=f"exact_action_conditioned_{name}_predictive_belief",
            seed=seed_sequence_to_int(
                streams[f"regression_{'joint' if name == 'joint_mixed_state' else name}"],
                bits=32,
            ),
        )
    decoded = decode_joint_indices(test.states)
    action_counts = np.bincount(test.actions, minlength=N_ACTIONS)
    result = {
        "condition": condition,
        "checkpoint": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "is_initialization": training_iteration == 0,
        "representation": representation,
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
            "factor_1_state_2_fraction": float(np.mean(decoded[:, 0] == 2)),
            "factor_2_state_2_fraction": float(np.mean(decoded[:, 1] == 2)),
            "greedy_action_fractions": (
                action_counts.astype(np.float64) / len(test.actions)
            ).tolist(),
        },
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
    condition: str,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TRAIN_STEPS
    test_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TEST_STEPS
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
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
    return _analyze_samples(
        context,
        condition=condition,
        checkpoint_label=checkpoint_label,
        agent_steps=agent_steps,
        training_iteration=training_iteration,
        train=train,
        test=test,
        streams=streams,
        representation="actor_final_block_residual_before_final_layer_norm",
    )


__all__ = ["analyze_checkpoint", "plot_probe_trajectory"]
