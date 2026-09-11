from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from envs.hmm import HMMEnv, HMMModel, condition_edge
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.process import (
    COMPONENT_COUNT,
    CONTEXT_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    TOKEN_COUNT,
    environment_config,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.task import (
    N_ACTIONS,
    REWARD_STATE,
)
from harness.context import RunContext
from harness.seeding import named_seed_sequences


FULL_PROBE_TRAIN_STEPS = 20_000
FULL_PROBE_TEST_STEPS = 20_000
SMOKE_PROBE_STEPS = 64
N_ENVS = 4
WARMUP = 0
PROBE_RIDGE = 1e-6
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
    reward_state_beliefs: np.ndarray
    antisymmetric_beliefs: np.ndarray
    pending_token_distributions: np.ndarray
    diagnostic_beliefs: np.ndarray
    observations: np.ndarray
    states: np.ndarray
    tokens: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_steps: np.ndarray
    env_indices: np.ndarray


class ActionConditionedTransducerTarget:
    def __init__(self, model: HMMModel, n_envs: int) -> None:
        if model.edge_transition_matrices is None:
            raise ValueError("edge-emitting transducer requires edge kernels")
        self.model = model
        self.n_envs = n_envs
        self.source_beliefs = np.repeat(
            model.initial_distribution[None, :],
            n_envs,
            axis=0,
        )
        self.pending_edges = np.repeat(
            model.edge_transition_matrices[None, :, :, :],
            n_envs,
            axis=0,
        )

    def __call__(
        self,
        observations: np.ndarray,
        infos: Sequence[Mapping[str, Any]],
        episode_steps: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        del observations
        beliefs = np.empty((self.n_envs, STATE_COUNT), dtype=np.float64)
        pending_tokens = np.empty(
            (self.n_envs, TOKEN_COUNT),
            dtype=np.float64,
        )
        tokens = np.full(self.n_envs, -1, dtype=np.int64)
        actions = np.full(self.n_envs, -1, dtype=np.int64)
        for index, (info, episode_step) in enumerate(
            zip(infos, episode_steps)
        ):
            if episode_step == 0:
                self.source_beliefs[index] = self.model.initial_distribution
                self.pending_edges[index] = (
                    self.model.edge_transition_matrices
                )
                beliefs[index] = (
                    self.source_beliefs[index]
                    @ self.model.transition_matrix
                )
            else:
                visible_token = info.get("visible_token_current")
                if visible_token is None:
                    raise ValueError(
                        "post-reset transducer update requires a visible token"
                    )
                tokens[index] = int(visible_token)
                self.source_beliefs[index] = condition_edge(
                    self.source_beliefs[index],
                    self.pending_edges[index],
                    tokens[index],
                )
                try:
                    transition = np.asarray(
                        info["executed_transition_matrix"],
                        dtype=np.float64,
                    )
                    edges = np.asarray(
                        info["executed_edge_transition_matrices"],
                        dtype=np.float64,
                    )
                    actions[index] = int(info["executed_action"])
                except KeyError as error:
                    raise KeyError(
                        "action-conditioned transducer requires transition "
                        "diagnostics"
                    ) from error
                beliefs[index] = self.source_beliefs[index] @ transition
                self.pending_edges[index] = edges
            pending_tokens[index] = np.einsum(
                "i,yij->y",
                self.source_beliefs[index],
                self.pending_edges[index],
            )

        component_posteriors = beliefs.reshape(
            -1,
            COMPONENT_COUNT,
            STATES_PER_COMPONENT,
        ).sum(axis=2)
        reward_state_beliefs = beliefs[
            :,
            REWARD_STATE::STATES_PER_COMPONENT,
        ].sum(axis=1, keepdims=True)
        antisymmetric_beliefs = (
            beliefs[:, 0::STATES_PER_COMPONENT]
            - beliefs[:, 1::STATES_PER_COMPONENT]
        ).sum(axis=1, keepdims=True)
        return {
            "weighted_belief": beliefs,
            "component_posterior": component_posteriors,
            "reward_state_belief": reward_state_beliefs,
            "antisymmetric_belief": antisymmetric_beliefs,
            "pending_token_distribution": pending_tokens,
            "diagnostic_belief": np.stack(
                [info["belief_current"] for info in infos]
            ),
            "state": np.asarray(
                [info["state_current"] for info in infos],
                dtype=np.int64,
            ),
            "token": tokens,
            "executed_action": actions,
            "episode_step": np.asarray(episode_steps, dtype=np.int64),
            "env_index": np.arange(self.n_envs, dtype=np.int64),
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


def _sample_actions(
    logits: torch.Tensor,
    randomness: PolicyRandomness,
    action_spaces: Sequence[Any],
    policy_mode: str,
) -> np.ndarray:
    if policy_mode == "random":
        return np.asarray(
            [action_space.sample() for action_space in action_spaces],
            dtype=np.int64,
        )
    if policy_mode == "greedy":
        return logits.argmax(dim=-1).cpu().numpy()
    if policy_mode != "policy":
        raise ValueError(f"unsupported policy mode {policy_mode!r}")
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
    module: Any,
    *,
    variant: int,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
    policy_mode: str = "greedy",
    n_envs: int = N_ENVS,
    warmup: int = WARMUP,
) -> ProbeData:
    config = environment_config(variant)
    config["diagnostics"] = {
        "belief": True,
        "state": True,
        "tokens": True,
        "transitions": True,
    }
    model_environment = HMMEnv(config)
    try:
        transducer = ActionConditionedTransducerTarget(
            model_environment.model,
            n_envs,
        )
    finally:
        model_environment.close()
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
            state[name] = (
                torch.as_tensor(array, device=device)
                .unsqueeze(0)
                .expand(batch_size, *array.shape)
                .clone()
            )
        return state

    def reset_state(state, indices):
        fresh = initial_state(len(indices))
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    def step_adapter(observations, state, randomness, action_spaces):
        captured.clear()
        tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=device,
        )
        residual, state_out = module.encode_step_pre_final_norm(tensor, state)
        if len(captured) != len(blocks):
            raise RuntimeError("every encoder block must execute once")
        layers = torch.stack(
            [captured[index] for index in range(len(blocks))],
            dim=1,
        )
        logits = module.action_distribution_inputs(
            module.encoder.final_norm(residual)
        )
        actions = _sample_actions(
            logits,
            randomness,
            action_spaces,
            policy_mode,
        )
        return actions, state_out, layers.cpu().numpy()

    handles = []
    try:
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(capture(index)))
        collected = collect_batched_rollout_data(
            lambda: HMMEnv(config),
            step_adapter,
            transducer,
            n_steps=n_steps,
            seed=seed,
            n_envs=n_envs,
            initial_state=initial_state,
            reset_state=reset_state,
            warmup=warmup,
            store_observations=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if collected.observations is None:
        raise AssertionError("probe collection requires observations")
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
        reward_state_beliefs=np.asarray(
            collected.targets["reward_state_belief"],
            dtype=np.float64,
        ),
        antisymmetric_beliefs=np.asarray(
            collected.targets["antisymmetric_belief"],
            dtype=np.float64,
        ),
        pending_token_distributions=np.asarray(
            collected.targets["pending_token_distribution"],
            dtype=np.float64,
        ),
        diagnostic_beliefs=np.asarray(
            collected.targets["diagnostic_belief"],
            dtype=np.float64,
        ),
        observations=np.asarray(collected.observations, dtype=np.float64),
        states=np.asarray(collected.targets["state"], dtype=np.int64),
        tokens=np.asarray(collected.targets["token"], dtype=np.int64),
        actions=np.asarray(collected.actions, dtype=np.int64).reshape(-1),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
        episode_steps=np.asarray(
            collected.targets["episode_step"],
            dtype=np.int64,
        ),
        env_indices=np.asarray(
            collected.targets["env_index"],
            dtype=np.int64,
        ),
    )


def _observation_groups(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations)
    expected = TOKEN_COUNT + N_ACTIONS
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError("expected one token and one action one-hot")
    token_features = values[:, :TOKEN_COUNT]
    action_features = values[:, TOKEN_COUNT:]
    tokens = np.where(
        token_features.sum(axis=1) > 0.5,
        token_features.argmax(axis=1),
        TOKEN_COUNT,
    )
    actions = np.where(
        action_features.sum(axis=1) > 0.5,
        action_features.argmax(axis=1),
        N_ACTIONS,
    )
    return tokens * (N_ACTIONS + 1) + actions


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
    weight, bias = fit_affine_probe(
        train_features,
        train_target,
        ridge=PROBE_RIDGE,
    )
    return _metrics(
        probe_predict(weight, bias, test_features),
        test_target,
        test_groups,
    )


def _branch_prediction(
    train_groups: np.ndarray,
    train_target: np.ndarray,
    test_groups: np.ndarray,
) -> np.ndarray:
    predicted = np.broadcast_to(
        train_target.mean(axis=0),
        (len(test_groups), train_target.shape[1]),
    ).copy()
    for group in np.unique(train_groups):
        predicted[test_groups == group] = train_target[
            train_groups == group
        ].mean(axis=0)
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
    target_error = max(
        float(
            np.max(
                np.abs(
                    data.weighted_beliefs - data.diagnostic_beliefs
                )
            )
        )
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "transducer target is misaligned with diagnostics: "
            f"{target_error:.3e}"
        )
    targets = {
        "weighted_belief": (
            train.weighted_beliefs,
            test.weighted_beliefs,
        ),
        "component_posterior": (
            train.component_posteriors,
            test.component_posteriors,
        ),
        "reward_state_belief": (
            train.reward_state_beliefs,
            test.reward_state_beliefs,
        ),
        "antisymmetric_belief": (
            train.antisymmetric_beliefs,
            test.antisymmetric_beliefs,
        ),
        "pending_token_distribution": (
            train.pending_token_distributions,
            test.pending_token_distributions,
        ),
    }
    train_groups = _observation_groups(train.observations)
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
    pending_weight, pending_bias = fit_affine_probe(
        train.pending_token_distributions,
        train.weighted_beliefs,
        ridge=PROBE_RIDGE,
    )
    controls = {
        "train_mean_weighted_belief": _metrics(
            mean_belief,
            test.weighted_beliefs,
            test_groups,
        ),
        "current_token_action_to_weighted_belief": _metrics(
            _branch_prediction(
                train_groups,
                train.weighted_beliefs,
                test_groups,
            ),
            test.weighted_beliefs,
            test_groups,
        ),
        "pending_token_distribution_to_weighted_belief": _metrics(
            probe_predict(
                pending_weight,
                pending_bias,
                test.pending_token_distributions,
            ),
            test.weighted_beliefs,
            test_groups,
        ),
    }
    final_layer = layers[f"layer_{train.activations.shape[1]}"]
    action_counts = np.bincount(test.actions, minlength=N_ACTIONS)
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
            "policy_mode": "greedy",
            "n_envs": N_ENVS,
            "warmup_per_episode": WARMUP,
            "context_length": CONTEXT_LENGTH,
            "belief_timing": (
                "decision-time arrival belief after conditioning the newly "
                "visible edge outcome under the preceding action and applying "
                "the preceding executed transition reported in the current "
                "environment diagnostics"
            ),
            "belief_target": (
                "six weighted state probabilities "
                "(w_A eta_A, w_B eta_B)"
            ),
            "cv_scope": "independent fit and test rollout seed streams",
        },
        "target_consistency_max_abs": target_error,
        "layers": layers,
        "probe_fits": final_layer["probe_fits"],
        "controls": controls,
        "cev": {
            "actor_activation": final_layer["cev"],
            "weighted_belief_target": variance_geometry(
                test.weighted_beliefs
            ),
        },
        "policy": {
            "mean_reward_state_occupancy": float(test.rewards.mean()),
            "greedy_action_fractions": (
                action_counts / max(action_counts.sum(), 1)
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
            "Linear accessibility does not establish causal policy use or "
            "unique representation."
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
    variant: int,
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
            variant=variant,
            n_steps=steps,
            seed=streams["probe_train"],
            device=_device(context),
        )
        test = collect_probe_data(
            module,
            variant=variant,
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


__all__ = [
    "ActionConditionedTransducerTarget",
    "ProbeData",
    "analyze_checkpoint",
    "collect_probe_data",
]
