"""Distinguish fine and coarse filtering with independent token-0/1 flips."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from analysis.probes import (
    fit_affine_probe,
    global_mse_metrics,
    mean_squared_error,
    predictive_belief_update,
    probe_predict,
    r2_score,
)
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.analysis import (
    _coarse_spec,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.token_swap_diagnostic.analysis import (
    _contiguous_segments,
    _device,
    _initial_state,
    _resolve_checkpoint,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    environment_config,
)
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int
from learners.models import TransformerModel

PROBE_RIDGE = 1e-6
N_ENVS = 16
N_RANDOMIZATIONS = 8
FLIP_PROBABILITY = 0.5
FULL_TRAIN_STEPS = 30_000
FULL_TEST_STEPS = 40_000
FULL_CLOSED_LOOP_STEPS = 40_000
SMOKE_STEPS = 4_096
SMOKE_RANDOMIZATIONS = 2
_STREAM_KEYS = {
    "probe_train": (710,),
    "probe_test": (711,),
    "closed_loop": (712,),
    "local_flip": (713,),
    "closed_loop_flip": (714,),
}


def independently_flip_state_0_1_tokens(
    observations: np.ndarray,
    *,
    rng: np.random.Generator,
    probability: float = FLIP_PROBABILITY,
) -> np.ndarray:
    """Independently exchange token channels 0/1 at each eligible position."""

    values = np.asarray(observations)
    if values.ndim < 1 or values.shape[-1] < 3:
        raise ValueError("observations must end with at least three token channels")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    randomized = np.array(values, copy=True)
    eligible = (values[..., 0] == 1.0) | (values[..., 1] == 1.0)
    flipped = eligible & (rng.random(eligible.shape) < probability)
    randomized[..., 0][flipped] = values[..., 1][flipped]
    randomized[..., 1][flipped] = values[..., 0][flipped]
    return randomized


def _validate_complete_histories(data: ProbeData) -> list[np.ndarray]:
    segments = _contiguous_segments(data)
    if not segments:
        raise RuntimeError("rollout collection produced no episode segments")
    incomplete = [
        int(data.episode_steps[segment[0]])
        for segment in segments
        if int(data.episode_steps[segment[0]]) != 0
    ]
    if incomplete:
        raise ValueError(
            "independent-flip targets require histories beginning at episode step 0"
        )
    return segments


def _token_ids(observations: np.ndarray) -> np.ndarray:
    token_channels = np.asarray(observations)[..., :3]
    if not np.allclose(token_channels.sum(axis=-1), 1.0):
        raise ValueError("each observation must contain one active token channel")
    return token_channels.argmax(axis=-1).astype(np.int64)


def _filter_targets(
    data: ProbeData,
    observations: np.ndarray,
    *,
    initial: np.ndarray,
    emission: np.ndarray,
    transitions: dict[int, np.ndarray],
) -> np.ndarray:
    """Filter one complete token history per environment episode."""

    tokens = _token_ids(observations)
    actions = np.asarray(data.actions, dtype=np.int64).reshape(-1)
    targets = np.empty((len(tokens), len(initial)), dtype=np.float64)
    for segment in _validate_complete_histories(data):
        belief = np.asarray(initial, dtype=np.float64)
        for offset, row in enumerate(segment):
            measurement = np.diag(emission[:, int(tokens[row])])
            operator = (
                measurement
                if offset == 0
                else transitions[int(actions[segment[offset - 1]])] @ measurement
            )
            belief = predictive_belief_update(belief, operator)
            targets[row] = belief
    return targets


def _coarse_observations(observations: np.ndarray) -> np.ndarray:
    tokens = _token_ids(observations)
    coarse = np.zeros((*tokens.shape, 2), dtype=np.float32)
    coarse[..., 0] = tokens != 2
    coarse[..., 1] = tokens == 2
    return coarse


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


@torch.no_grad()
def paired_local_flip_replay(
    module: TransformerModel,
    data: ProbeData,
    *,
    randomization_seeds: list[int],
    device: str,
    warmup: int,
) -> dict[str, np.ndarray | float]:
    """Replay factual and independently randomized fixed-action histories."""

    if not module.is_stateful():
        raise TypeError("independent-flip diagnostic requires a stateful transformer")
    if data.observations is None:
        raise ValueError("paired replay requires stored rollout observations")
    segments = _validate_complete_histories(data)
    factual_observations = np.asarray(data.observations, dtype=np.float32)
    randomized_observations = np.stack(
        [
            independently_flip_state_0_1_tokens(
                factual_observations,
                rng=np.random.default_rng(seed),
            )
            for seed in randomization_seeds
        ]
    )
    max_length = max(map(len, segments))
    n_segments = len(segments)
    n_versions = 1 + len(randomization_seeds)
    padded = np.zeros(
        (
            n_versions,
            n_segments,
            max_length,
            factual_observations.shape[1],
        ),
        dtype=np.float32,
    )
    lengths = np.asarray([len(segment) for segment in segments], dtype=np.int64)
    for segment_index, segment in enumerate(segments):
        padded[0, segment_index, : len(segment)] = factual_observations[segment]
        for randomization_index, observations in enumerate(
            randomized_observations,
            start=1,
        ):
            padded[
                randomization_index,
                segment_index,
                : len(segment),
            ] = observations[segment]
    paired = padded.reshape(
        n_versions * n_segments,
        max_length,
        factual_observations.shape[1],
    )

    torch_device = torch.device(device)
    module = module.to(torch_device).eval()
    state = _initial_state(module, len(paired), torch_device)
    embedding_dim = data.activations.shape[1]
    action_count = module.action_space.n
    factual_activations = np.empty((len(data.activations), embedding_dim))
    factual_logits = np.empty((len(data.activations), action_count))
    randomized_activations = np.empty(
        (len(randomization_seeds), len(data.activations), embedding_dim)
    )
    randomized_logits = np.empty(
        (len(randomization_seeds), len(data.activations), action_count)
    )
    eligible: list[int] = []
    lookback = int(module.sequence_lookback)

    for step in range(max_length):
        inputs = torch.from_numpy(paired[:, step]).to(torch_device)
        embedding, state = module.encode_step(inputs, state)
        logits = module.action_distribution_inputs(embedding)
        encoded = embedding.detach().cpu().numpy()
        action_logits = logits.detach().cpu().numpy()
        active_segments = np.flatnonzero(lengths > step)
        for segment_index in active_segments:
            row = int(segments[segment_index][step])
            factual_activations[row] = encoded[segment_index]
            factual_logits[row] = action_logits[segment_index]
            for randomization_index in range(len(randomization_seeds)):
                source = (randomization_index + 1) * n_segments + segment_index
                randomized_activations[randomization_index, row] = encoded[source]
                randomized_logits[randomization_index, row] = action_logits[source]
            if step >= max(lookback, warmup):
                eligible.append(row)

    indices = np.asarray(sorted(eligible), dtype=np.int64)
    if not len(indices):
        raise RuntimeError("no histories exceeded the replay warmup")
    reconstruction_error = float(
        np.max(np.abs(factual_activations[indices] - data.activations[indices]))
    )
    if reconstruction_error > 2e-5:
        raise AssertionError(
            "reconstructed factual histories do not match rollout activations: "
            f"{reconstruction_error:.3e}"
        )
    return {
        "factual_activations": factual_activations[indices],
        "factual_logits": factual_logits[indices],
        "randomized_activations": randomized_activations[:, indices],
        "randomized_logits": randomized_logits[:, indices],
        "randomized_observations": randomized_observations,
        "indices": indices,
        "reconstruction_error": reconstruction_error,
    }


def _probe_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    metrics = global_mse_metrics(predictions, targets)
    return {
        **metrics,
        "r_squared": r2_score(predictions, targets),
    }


def _policy_metrics(
    factual_logits: np.ndarray,
    randomized_logits: np.ndarray,
) -> dict[str, float]:
    factual_probabilities = _softmax(factual_logits)
    randomized_probabilities = _softmax(randomized_logits)
    return {
        "logit_shift_rms": float(
            np.sqrt(np.mean(np.square(randomized_logits - factual_logits)))
        ),
        "probability_total_variation_mean": float(
            0.5
            * np.abs(randomized_probabilities - factual_probabilities).sum(axis=-1).mean()
        ),
        "greedy_action_agreement": float(
            np.mean(
                randomized_logits.argmax(axis=-1)
                == factual_logits.argmax(axis=-1)
            )
        ),
    }


def _evaluate_replay(
    *,
    replay: dict[str, np.ndarray | float],
    factual_fine_targets: np.ndarray,
    randomized_fine_targets: np.ndarray,
    coarse_targets: np.ndarray,
    fine_weight: np.ndarray,
    fine_bias: np.ndarray,
    coarse_weight: np.ndarray,
    coarse_bias: np.ndarray,
) -> dict[str, object]:
    factual_activations = np.asarray(replay["factual_activations"])
    randomized_activations = np.asarray(replay["randomized_activations"])
    factual_logits = np.asarray(replay["factual_logits"])
    randomized_logits = np.asarray(replay["randomized_logits"])
    factual_s = factual_fine_targets[:, 2:3]
    randomized_s = randomized_fine_targets[..., 2:3]
    factual_fine_prediction = probe_predict(
        fine_weight,
        fine_bias,
        factual_activations,
    )
    factual_coarse_prediction = probe_predict(
        coarse_weight,
        coarse_bias,
        factual_activations,
    )
    randomized_fine_prediction = np.stack(
        [
            probe_predict(fine_weight, fine_bias, activations)
            for activations in randomized_activations
        ]
    )
    randomized_coarse_prediction = np.stack(
        [
            probe_predict(coarse_weight, coarse_bias, activations)
            for activations in randomized_activations
        ]
    )
    repeated_coarse = np.broadcast_to(
        coarse_targets,
        randomized_coarse_prediction.shape,
    )
    exact_s_shift = randomized_s - factual_s
    decoded_s_shift = randomized_fine_prediction - factual_fine_prediction
    decoded_coarse_shift = (
        randomized_coarse_prediction - factual_coarse_prediction
    )

    by_randomization = []
    for index in range(len(randomized_activations)):
        fine_metrics = _probe_metrics(
            randomized_fine_prediction[index],
            randomized_s[index],
        )
        coarse_metrics = _probe_metrics(
            randomized_coarse_prediction[index],
            coarse_targets,
        )
        by_randomization.append(
            {
                "randomization": index,
                "fine_probe": fine_metrics,
                "coarse_probe": coarse_metrics,
                "exact_s_shift_rms": float(
                    np.sqrt(np.mean(np.square(exact_s_shift[index])))
                ),
                "decoded_s_shift_rms": float(
                    np.sqrt(np.mean(np.square(decoded_s_shift[index])))
                ),
                "decoded_s_shift_mse": mean_squared_error(
                    decoded_s_shift[index],
                    exact_s_shift[index],
                ),
                "decoded_s_shift_r_squared": r2_score(
                    decoded_s_shift[index],
                    exact_s_shift[index],
                ),
                "decoded_coarse_invariance_rmse": float(
                    np.sqrt(np.mean(np.square(decoded_coarse_shift[index])))
                ),
                "policy": _policy_metrics(
                    factual_logits,
                    randomized_logits[index],
                ),
            }
        )

    return {
        "factual": {
            "fine_probe_s": _probe_metrics(
                factual_fine_prediction,
                factual_s,
            ),
            "coarse_probe_c": _probe_metrics(
                factual_coarse_prediction,
                coarse_targets,
            ),
        },
        "randomized": {
            "fine_probe_s": _probe_metrics(
                randomized_fine_prediction.reshape(-1, 1),
                randomized_s.reshape(-1, 1),
            ),
            "coarse_probe_c": _probe_metrics(
                randomized_coarse_prediction.reshape(-1, 1),
                repeated_coarse.reshape(-1, 1),
            ),
            "exact_s_shift_rms": float(
                np.sqrt(np.mean(np.square(exact_s_shift)))
            ),
            "decoded_s_shift_rms": float(
                np.sqrt(np.mean(np.square(decoded_s_shift)))
            ),
            "decoded_s_shift_mse": mean_squared_error(
                decoded_s_shift,
                exact_s_shift,
            ),
            "decoded_s_shift_r_squared": r2_score(
                decoded_s_shift.reshape(-1, 1),
                exact_s_shift.reshape(-1, 1),
            ),
            "decoded_coarse_invariance_rmse": float(
                np.sqrt(np.mean(np.square(decoded_coarse_shift)))
            ),
            "activation_shift_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            randomized_activations
                            - factual_activations[None, ...]
                        )
                    )
                )
            ),
            "policy": _policy_metrics(
                np.broadcast_to(factual_logits, randomized_logits.shape),
                randomized_logits,
            ),
        },
        "fine_probe_counterfactual_over_factual_mse": float(
            mean_squared_error(
                randomized_fine_prediction,
                randomized_s,
            )
            / mean_squared_error(factual_fine_prediction, factual_s)
        ),
        "coarse_probe_counterfactual_over_factual_mse": float(
            mean_squared_error(
                randomized_coarse_prediction,
                repeated_coarse,
            )
            / mean_squared_error(factual_coarse_prediction, coarse_targets)
        ),
        "by_randomization": by_randomization,
    }


def _closed_loop_evaluation(
    *,
    module: TransformerModel,
    make_environment: Callable[[], HMMEnv],
    device: str,
    steps: int,
    warmup: int,
    rollout_seed: int,
    randomization_seeds: list[int],
) -> dict[str, object]:
    factual = collect_probe_data(
        module,
        make_environment,
        n_steps=steps,
        seed=rollout_seed,
        policy_mode="greedy",
        n_envs=N_ENVS,
        device=device,
        warmup=warmup,
    )
    randomized_runs = []
    for seed in randomization_seeds:
        rng = np.random.default_rng(seed)
        randomized = collect_probe_data(
            module,
            make_environment,
            n_steps=steps,
            seed=rollout_seed,
            policy_mode="greedy",
            n_envs=N_ENVS,
            device=device,
            warmup=warmup,
            observation_transform=lambda observations, rng=rng: (
                independently_flip_state_0_1_tokens(observations, rng=rng)
            ),
        )
        randomized_runs.append(
            {
                "reward_mean": float(randomized.rewards.mean()),
                "reward_delta": float(
                    randomized.rewards.mean() - factual.rewards.mean()
                ),
                "action_fractions": np.bincount(
                    np.asarray(randomized.actions, dtype=np.int64).reshape(-1),
                    minlength=3,
                ).astype(float).tolist(),
            }
        )
        total = sum(randomized_runs[-1]["action_fractions"])
        randomized_runs[-1]["action_fractions"] = [
            value / total for value in randomized_runs[-1]["action_fractions"]
        ]
    deltas = np.asarray(
        [run["reward_delta"] for run in randomized_runs],
        dtype=np.float64,
    )
    factual_counts = np.bincount(
        np.asarray(factual.actions, dtype=np.int64).reshape(-1),
        minlength=3,
    ).astype(float)
    return {
        "factual_reward_mean": float(factual.rewards.mean()),
        "factual_action_fractions": (factual_counts / factual_counts.sum()).tolist(),
        "randomized_reward_mean": float(
            np.mean([run["reward_mean"] for run in randomized_runs])
        ),
        "randomized_reward_delta_mean": float(deltas.mean()),
        "randomized_reward_delta_range": [
            float(deltas.min()),
            float(deltas.max()),
        ],
        "by_randomization": randomized_runs,
    }


@contextmanager
def _load_checkpoint_module(checkpoint: Path) -> Iterator[TransformerModel]:
    policy = (
        checkpoint
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
    )
    with (policy / "class_and_ctor_args.pkl").open("rb") as handle:
        specification = pickle.load(handle)
    module_class = specification["class"]
    args, kwargs = specification["ctor_args_and_kwargs"]
    module = module_class(*args, **kwargs)
    if not isinstance(module, TransformerModel):
        raise TypeError("checkpoint module is not a TransformerModel")
    with (policy / "module_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    module.set_state(state)
    try:
        yield module
    finally:
        del module


def probe_checkpoint(
    context: RunContext,
    checkpoint: Path,
) -> dict[str, object]:
    """Fit factual probes and evaluate independent local token flips."""

    if context.seed is None:
        raise ValueError("independent-flip diagnostic requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = SMOKE_STEPS if context.smoke else FULL_TRAIN_STEPS
    test_steps = SMOKE_STEPS if context.smoke else FULL_TEST_STEPS
    closed_loop_steps = (
        SMOKE_STEPS if context.smoke else FULL_CLOSED_LOOP_STEPS
    )
    n_randomizations = (
        SMOKE_RANDOMIZATIONS if context.smoke else N_RANDOMIZATIONS
    )
    warmup = 4 if context.smoke else 64
    flip_root = streams["local_flip"].spawn(n_randomizations)
    closed_loop_flip_root = streams["closed_loop_flip"].spawn(n_randomizations)
    randomization_seeds = [
        seed_sequence_to_int(seed, bits=32) for seed in flip_root
    ]
    closed_loop_randomization_seeds = [
        seed_sequence_to_int(seed, bits=32) for seed in closed_loop_flip_root
    ]

    with _load_checkpoint_module(checkpoint) as module:
        env_class = HMMEnv
        env_config = environment_config(2)
        env_config["randomize_first_episode_length"] = False
        env_config["diagnostics"] = {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        }

        def make_environment():
            return env_class(env_config)

        environment = make_environment()
        try:
            if environment.task.variant != 2:
                raise ValueError(
                    "independent-flip diagnostic requires action-symmetry variant 2"
                )
            fine_initial, action_outcome, initial_outcome = (
                make_transducer_target(environment)
            )
            fine_emission = np.asarray(
                environment.model.emission_matrix,
                dtype=np.float64,
            )
            fine_transitions = {
                action: np.asarray(
                    environment.task.transition_matrix_for_action(action),
                    dtype=np.float64,
                )
                for action in range(environment.action_space.n)
            }
            coarse_initial, coarse_emission, coarse_transitions = (
                _coarse_spec(environment)
            )
        finally:
            environment.close()
        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": _device(context),
            "n_envs": N_ENVS,
            "warmup": 0,
            "store_observations": True,
            "initial_belief": fine_initial,
            "action_outcome_operator": action_outcome,
            "initial_outcome_operator": initial_outcome,
        }
        train = collect_probe_data(
            n_steps=train_steps,
            seed=streams["probe_train"],
            **common,
        )
        test = collect_probe_data(
            n_steps=test_steps,
            seed=streams["probe_test"],
            **common,
        )
        if train.observations is None or test.observations is None:
            raise AssertionError("stored observations are required")
        train_coarse = _filter_targets(
            train,
            _coarse_observations(train.observations),
            initial=coarse_initial,
            emission=coarse_emission,
            transitions=coarse_transitions,
        )
        train_indices = np.flatnonzero(train.episode_steps >= warmup)
        fine_weight, fine_bias = fit_affine_probe(
            train.activations[train_indices],
            train.beliefs[train_indices, 2:3],
            ridge=PROBE_RIDGE,
        )
        coarse_weight, coarse_bias = fit_affine_probe(
            train.activations[train_indices],
            train_coarse[train_indices, 1:2],
            ridge=PROBE_RIDGE,
        )
        replay = paired_local_flip_replay(
            module,
            test,
            randomization_seeds=randomization_seeds,
            device=_device(context),
            warmup=warmup,
        )
        indices = np.asarray(replay["indices"], dtype=np.int64)
        randomized_observations = np.asarray(
            replay["randomized_observations"]
        )
        randomized_fine_targets = np.stack(
            [
                _filter_targets(
                    test,
                    observations,
                    initial=fine_initial,
                    emission=fine_emission,
                    transitions=fine_transitions,
                )[indices]
                for observations in randomized_observations
            ]
        )
        test_coarse = _filter_targets(
            test,
            _coarse_observations(test.observations),
            initial=coarse_initial,
            emission=coarse_emission,
            transitions=coarse_transitions,
        )
        evaluation = _evaluate_replay(
            replay=replay,
            factual_fine_targets=test.beliefs[indices],
            randomized_fine_targets=randomized_fine_targets,
            coarse_targets=test_coarse[indices, 1:2],
            fine_weight=fine_weight,
            fine_bias=fine_bias,
            coarse_weight=coarse_weight,
            coarse_bias=coarse_bias,
        )
        closed_loop = _closed_loop_evaluation(
            module=module,
            make_environment=make_environment,
            device=_device(context),
            steps=closed_loop_steps,
            warmup=warmup,
            rollout_seed=seed_sequence_to_int(
                streams["closed_loop"],
                bits=32,
            ),
            randomization_seeds=closed_loop_randomization_seeds,
        )

    target_error = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "fine Bayesian target disagrees with environment diagnostics: "
            f"{target_error:.3e}"
        )
    randomized_coarse = np.stack(
        [
            _filter_targets(
                test,
                _coarse_observations(observations),
                initial=coarse_initial,
                emission=coarse_emission,
                transitions=coarse_transitions,
            )[indices, 1:2]
            for observations in randomized_observations
        ]
    )
    coarse_invariance_error = float(
        np.max(np.abs(randomized_coarse - test_coarse[indices, 1:2]))
    )
    if coarse_invariance_error > 1e-12:
        raise AssertionError(
            "independent flips changed the coarse-filter target: "
            f"{coarse_invariance_error:.3e}"
        )

    return {
        "schema_version": 1,
        "study": "independent_state_0_1_token_flip_diagnostic",
        "cycle": 6,
        "variant": 2,
        "seed": context.seed,
        "checkpoint": str(checkpoint.resolve()),
        "hypotheses": {
            "coarse_filter": (
                "Representations and policy outputs remain invariant when each "
                "token 0/1 identity is independently flipped while the coarse "
                "not-2/2 history is fixed."
            ),
            "fine_filter": (
                "The decoded state-2 belief tracks the exact change induced in "
                "the full three-state filter by the independently flipped history."
            ),
        },
        "intervention": (
            "Independently exchange each token-0/token-1 channel with probability "
            "0.5; preserve token-2 positions and factual previous-action features."
        ),
        "decoder": "frozen factual affine least-squares probes for s_t and c_t",
        "representation": "post_final_layer_norm",
        "sampling_distribution": "greedy_process_weighted_factual_rollout",
        "train_steps": train_steps,
        "test_steps": test_steps,
        "closed_loop_steps": closed_loop_steps,
        "n_randomizations": n_randomizations,
        "randomization_seeds": randomization_seeds,
        "closed_loop_randomization_seeds": closed_loop_randomization_seeds,
        "n_evaluated": int(len(indices)),
        "n_envs": N_ENVS,
        "warmup": warmup,
        "flip_probability": FLIP_PROBABILITY,
        "policy_feedback_during_paired_replay": False,
        "probe_ridge": PROBE_RIDGE,
        "target_consistency_max_abs": target_error,
        "coarse_target_invariance_max_abs": coarse_invariance_error,
        "activation_reconstruction_max_abs": float(
            replay["reconstruction_error"]
        ),
        "paired_fixed_action_replay": evaluation,
        "closed_loop_policy": closed_loop,
    }


def run_independent_flip_diagnostic(
    context: RunContext,
) -> dict[str, object]:
    if context.resume_from is None:
        raise ValueError(
            "resume_from must name an Algorithm checkpoint or a bundle with "
            "final_checkpoint/"
        )
    checkpoint = _resolve_checkpoint(Path(context.resume_from))
    summary = probe_checkpoint(context, checkpoint)
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "independent_flip_diagnostic.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
