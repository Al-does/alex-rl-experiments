from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from analysis.checkpoints import load_algorithm
from analysis.probes import (
    conditional_mse_metrics,
    fit_affine_probe,
    global_mse_metrics,
    probe_predict,
    r2_score,
    variance_geometry,
)
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from envs.hmm import HMMEnv
from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_COUNT,
    CONTEXT_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    TOKEN_COUNT,
    environment_config,
    nonergodic_mess3_model,
)
from harness.context import RunContext
from harness.seeding import named_seed_sequences


FULL_PROBE_TRAIN_STEPS = 20_000
FULL_PROBE_TEST_STEPS = 20_000
SMOKE_PROBE_STEPS = 64
N_ENVS = 4
WARMUP = 0
_STREAM_KEYS = {
    "probe_train": (900,),
    "probe_test": (901,),
    "shuffle": (902,),
}


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    weighted_beliefs: np.ndarray
    component_posteriors: np.ndarray
    next_token_distributions: np.ndarray
    observations: np.ndarray
    states: np.ndarray
    hidden_tokens: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_steps: np.ndarray


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
    infos: list[Mapping[str, object]],
    episode_steps: np.ndarray,
) -> Mapping[str, np.ndarray]:
    del observations
    model = nonergodic_mess3_model()
    arrival_beliefs = np.stack([info["belief_current"] for info in infos])
    beliefs = np.linalg.solve(
        model.transition_matrix.T,
        arrival_beliefs.T,
    ).T
    if (beliefs < -1e-10).any():
        raise RuntimeError("source-belief inversion produced negative mass")
    beliefs = np.clip(beliefs, 0.0, None)
    beliefs /= beliefs.sum(axis=1, keepdims=True)
    component_posteriors = beliefs.reshape(
        -1,
        COMPONENT_COUNT,
        STATES_PER_COMPONENT,
    ).sum(axis=2)
    return {
        "weighted_belief": beliefs,
        "component_posterior": component_posteriors,
        "next_token_distribution": beliefs @ model.emission_matrix,
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
            randomness.numpy.choice(len(probability), p=probability)
            for probability in probabilities
        ],
        dtype=np.int64,
    )


@torch.inference_mode()
def collect_probe_data(
    module: object,
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
        state = {}
        for name, value in module.get_initial_state().items():
            array = np.asarray(value)
            state[name] = torch.as_tensor(array, device=device).unsqueeze(0).expand(
                batch_size,
                *array.shape,
            ).clone()
        return state

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
        raise AssertionError("probe collection requires policy observations")
    actions = np.asarray(collected.actions, dtype=np.int64).reshape(-1)
    hidden_tokens = np.asarray(collected.targets["hidden_token"], dtype=np.int64)
    rewards = np.asarray(collected.rewards, dtype=np.float64)
    if not np.array_equal(rewards, (actions == hidden_tokens).astype(np.float64)):
        raise AssertionError("token-guess rewards are misaligned with targets")
    return ProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        weighted_beliefs=np.asarray(
            collected.targets["weighted_belief"],
            dtype=np.float64,
        ),
        component_posteriors=np.asarray(
            collected.targets["component_posterior"],
            dtype=np.float64,
        ),
        next_token_distributions=np.asarray(
            collected.targets["next_token_distribution"],
            dtype=np.float64,
        ),
        observations=np.asarray(collected.observations, dtype=np.float64),
        states=np.asarray(collected.targets["state"], dtype=np.int64),
        hidden_tokens=hidden_tokens,
        actions=actions,
        rewards=rewards,
        episode_steps=np.asarray(
            collected.targets["episode_step"],
            dtype=np.int64,
        ),
    )


def _observation_groups(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != TOKEN_COUNT:
        raise ValueError("expected one delayed MESS3-token one-hot")
    return np.where(
        observations.sum(axis=1) > 0.5,
        observations.argmax(axis=1),
        TOKEN_COUNT,
    )


def _metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    return {
        **global_mse_metrics(predicted, target),
        **conditional_mse_metrics(predicted, target, groups),
        "r_squared": r2_score(predicted, target),
    }


def _fit_report(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
    test_groups: np.ndarray,
) -> dict[str, object]:
    weight, bias = fit_affine_probe(train_features, train_target, ridge=0.0)
    return _metrics(
        probe_predict(weight, bias, test_features),
        test_target,
        test_groups,
    )


def _branch_prediction(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    train_groups = _observation_groups(train_features)
    test_groups = _observation_groups(test_features)
    predicted = np.broadcast_to(
        train_target.mean(axis=0),
        (len(test_groups), train_target.shape[1]),
    ).copy()
    for group in np.unique(train_groups):
        predicted[test_groups == group] = train_target[train_groups == group].mean(
            axis=0
        )
    return predicted


def _analyze_samples(
    context: RunContext,
    *,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
    train: ProbeData,
    test: ProbeData,
    shuffle_seed: np.random.SeedSequence,
) -> dict[str, object]:
    targets = {
        "weighted_belief": (
            train.weighted_beliefs,
            test.weighted_beliefs,
        ),
        "component_posterior": (
            train.component_posteriors,
            test.component_posteriors,
        ),
        "next_token_distribution": (
            train.next_token_distributions,
            test.next_token_distributions,
        ),
    }
    test_groups = _observation_groups(test.observations)
    permutation = np.random.default_rng(shuffle_seed).permutation(
        len(train.activations)
    )
    layers = {}
    for layer in range(train.activations.shape[1]):
        probe_fits = {}
        shuffled = {}
        for name, (train_target, test_target) in targets.items():
            probe_fits[name] = _fit_report(
                train.activations[:, layer],
                train_target,
                test.activations[:, layer],
                test_target,
                test_groups,
            )
            shuffled[name] = _fit_report(
                train.activations[:, layer],
                train_target[permutation],
                test.activations[:, layer],
                test_target,
                test_groups,
            )
        layers[f"layer_{layer + 1}"] = {
            "representation": (
                f"encoder.blocks.{layer}.output_current_position_pre_final_norm"
            ),
            "feature_width": int(train.activations.shape[2]),
            "probe_fits": probe_fits,
            "shuffled_training_labels": shuffled,
            "cev": variance_geometry(test.activations[:, layer]),
        }

    mean_belief = np.broadcast_to(
        train.weighted_beliefs.mean(axis=0),
        test.weighted_beliefs.shape,
    )
    next_token_weight, next_token_bias = fit_affine_probe(
        train.next_token_distributions,
        train.weighted_beliefs,
        ridge=0.0,
    )
    controls = {
        "train_mean_weighted_belief": _metrics(
            mean_belief,
            test.weighted_beliefs,
            test_groups,
        ),
        "current_visible_token_to_weighted_belief": _metrics(
            _branch_prediction(
                train.observations,
                train.weighted_beliefs,
                test.observations,
            ),
            test.weighted_beliefs,
            test_groups,
        ),
        "next_token_distribution_to_weighted_belief": _metrics(
            probe_predict(
                next_token_weight,
                next_token_bias,
                test.next_token_distributions,
            ),
            test.weighted_beliefs,
            test_groups,
        ),
    }
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
            "n_envs": N_ENVS,
            "warmup_per_episode": WARMUP,
            "context_length": CONTEXT_LENGTH,
            "belief_timing": (
                "filtered edge-source belief that predicts the pending token; "
                "recovered from info.belief_current using the exact invertible "
                "transition matrix"
            ),
            "belief_target": (
                "six weighted state probabilities "
                "(w_A eta_A, w_B eta_B)"
            ),
            "next_token_target": (
                "exact pending-token distribution from the same belief"
            ),
            "component_target": "posterior mass in each three-state component",
            "cv_scope": "independent fit and test rollout seed streams",
        },
        "layers": layers,
        "probe_fits": final_layer["probe_fits"],
        "controls": controls,
        "cev": {
            "actor_activation": final_layer["cev"],
            "weighted_belief_target": variance_geometry(test.weighted_beliefs),
            "component_posterior_target": variance_geometry(
                test.component_posteriors
            ),
            "next_token_distribution_target": variance_geometry(
                test.next_token_distributions
            ),
        },
        "policy": {
            "mean_sampled_reward": float(test.rewards.mean()),
            "bayes_greedy_expected_accuracy": float(
                test.next_token_distributions.max(axis=1).mean()
            ),
            "guess_fractions": (
                np.bincount(test.actions, minlength=TOKEN_COUNT) / len(test.actions)
            ).tolist(),
            "hidden_token_fractions": (
                np.bincount(test.hidden_tokens, minlength=TOKEN_COUNT)
                / len(test.hidden_tokens)
            ).tolist(),
        },
        "coverage": {
            "state_counts": np.bincount(
                test.states,
                minlength=STATE_COUNT,
            ).tolist(),
            "generating_component_counts": np.bincount(
                test.states // STATES_PER_COMPONENT,
                minlength=COMPONENT_COUNT,
            ).tolist(),
            "episode_step_min": int(test.episode_steps.min()),
            "episode_step_max": int(test.episode_steps.max()),
        },
        "scope_warning": (
            "Linear accessibility beyond the next-token control does not "
            "establish causal use or unique representation."
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
) -> dict[str, object]:
    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TRAIN_STEPS
    test_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TEST_STEPS
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        train = collect_probe_data(
            module,
            n_steps=steps,
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
        shuffle_seed=streams["shuffle"],
    )


__all__ = ["ProbeData", "analyze_checkpoint", "collect_probe_data"]
