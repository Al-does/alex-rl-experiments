"""Stateful PPO belief probes matching the SAC study's target battery."""

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
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from experiments.two_factor_reward_state_PPO_cycle_1.process import (
    FACTOR_CARDINALITY,
    FACTOR_COUNT,
    decode_joint_indices,
    environment_config,
)
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


@torch.inference_mode()
def collect_probe_data(
    module: Any,
    *,
    condition: str,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
) -> ProbeData:
    """Collect recurrent actor residuals and exact controlled-filter targets."""

    config = environment_config(condition)
    config["diagnostics"] = {"belief": True, "state": True}
    module = module.to(device).eval()

    def make_environment() -> HMMEnv:
        return HMMEnv(config)

    def initial_state(batch_size: int) -> dict[str, torch.Tensor]:
        return _initial_state(module, batch_size, device)

    def reset_state(
        state: dict[str, torch.Tensor],
        indices: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        fresh = _initial_state(module, len(indices), device)
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
        del randomness, action_spaces
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        residual, state_out = module.encode_step_pre_final_norm(tensor, state)
        normalized = module.encoder.final_norm(residual)
        logits = module.action_distribution_inputs(normalized)
        return (
            logits.argmax(dim=-1).cpu().numpy(),
            state_out,
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
        initial_state=initial_state,
        reset_state=reset_state,
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


def analyze_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
) -> dict[str, Any]:
    """Probe PPO's shared actor-critic residual with the PR 65 battery."""

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
    action_counts = np.bincount(test.actions, minlength=9).astype(np.float64)
    result = {
        "condition": condition,
        "checkpoint": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "is_initialization": training_iteration == 0,
        "representation": (
            "shared_actor_critic_final_block_residual_before_final_layer_norm"
        ),
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
            "greedy_action_fractions": (action_counts / len(test.actions)).tolist(),
        },
        "scope_warning": (
            "Linear accessibility and low CEV dimension do not by themselves "
            "establish causal use or orthogonal factor subspaces."
        ),
    }
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "probe_battery.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


__all__ = [
    "analyze_checkpoint",
    "collect_probe_data",
    "plot_probe_trajectory",
]
