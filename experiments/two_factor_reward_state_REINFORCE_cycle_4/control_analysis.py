"""Frozen-input controls for the Reward Relevance probe battery."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from analysis.probes.controls import (
    fit_grouped_affine,
    gaussian_feature_null,
    score_prediction,
)
from envs.hmm import HMMEnv, factor_marginals, product_distribution
from harness.seeding import (
    child_seed_sequence,
    named_seed_sequences,
    seed_sequence_to_int,
)

from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from experiments.two_factor_reward_state_PPO_cycle_2.process import (
    environment_config,
)


TARGETS = ("joint_mixed_state", "factor_1", "factor_2")
FULL_STEPS = 20_000
SMOKE_STEPS = 512
N_ENVS = 8
WARMUP = 8
N_NULL_REPEATS = 5
_PROBE_STREAMS = {
    "probe_train": (700,),
    "probe_test": (701,),
}
_ROLLOUT_STREAMS = {
    "episode_seeds": (0,),
    "action_spaces": (1,),
    "policy_sampling": (2,),
}
_CONTROL_STREAMS = {
    "shuffle": (0,),
    "empirical_random": (1,),
    "uniform_random": (2,),
    "nulls": (3,),
}


@dataclass(frozen=True, slots=True)
class HistoryData:
    observations: np.ndarray
    beliefs: np.ndarray
    factor_beliefs: np.ndarray
    episode_ids: np.ndarray
    episode_steps: np.ndarray
    mask: np.ndarray
    metadata: Mapping[str, object]


def _integer(value: int, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _episode_seed(
    stream: np.random.SeedSequence,
    environment_index: int,
    episode_index: int,
) -> int:
    return seed_sequence_to_int(
        child_seed_sequence(stream, (environment_index, episode_index))
    )


def _episode_members(data: HistoryData) -> list[np.ndarray]:
    count = len(data.episode_ids)
    if data.episode_steps.shape != (count,):
        raise ValueError("episode IDs and steps must be aligned")
    groups: dict[int, list[int]] = {}
    for index, episode in enumerate(data.episode_ids):
        groups.setdefault(int(episode), []).append(index)
    members = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    for indices in members:
        expected = np.arange(len(indices), dtype=np.int64)
        if not np.array_equal(data.episode_steps[indices], expected):
            raise ValueError(
                "each episode must contain a complete ordered prefix from reset"
            )
    return members


def _representation_step(module: object, device: torch.device):
    def forward(
        observations: np.ndarray,
        state: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
        tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=device,
        )
        residual, state_out = module.encode_step_pre_final_norm(tensor, state)
        normalized = module.encoder.final_norm(residual)
        logits = module.action_distribution_inputs(normalized)
        actions = logits.argmax(dim=-1)
        return (
            residual.cpu().numpy(),
            actions.cpu().numpy(),
            state_out,
        )

    module.to(device).eval()
    return forward


@torch.inference_mode()
def collect_history_data(
    module: object,
    *,
    condition: str,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
    n_envs: int = N_ENVS,
    warmup: int = WARMUP,
) -> HistoryData:
    n_steps = _integer(n_steps, "n_steps", 1)
    n_envs = min(_integer(n_envs, "n_envs", 1), n_steps)
    warmup = _integer(warmup, "warmup", 0)
    config = environment_config(condition)
    if warmup >= int(config["episode_length"]):
        raise ValueError("warmup must be shorter than the episode")
    config["diagnostics"] = {"belief": True}
    streams = named_seed_sequences(seed, _ROLLOUT_STREAMS)
    envs: list[HMMEnv] = []
    observations: list[np.ndarray] = []
    infos: list[Mapping[str, object]] = []
    episode_indices = np.zeros(n_envs, dtype=np.int64)
    episode_ids = np.arange(n_envs, dtype=np.int64)
    next_episode_id = n_envs
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "observations",
            "beliefs",
            "factor_beliefs",
            "episode_ids",
            "episode_steps",
            "mask",
        )
    }

    def reset(index: int) -> tuple[np.ndarray, Mapping[str, object]]:
        envs[index].action_space.seed(
            _episode_seed(
                streams["action_spaces"],
                index,
                int(episode_indices[index]),
            )
        )
        return envs[index].reset(
            seed=_episode_seed(
                streams["episode_seeds"],
                index,
                int(episode_indices[index]),
            )
        )

    scored = 0
    try:
        for index in range(n_envs):
            env = HMMEnv(config)
            envs.append(env)
            if (
                env.model.n_states != 9
                or env.model.n_tokens != 9
                or env.action_space.n != 9
                or env.observation_space.shape != (18,)
            ):
                raise ValueError(
                    "control collection requires the Reward Relevance recipe"
                )
            observation, info = reset(index)
            observations.append(observation)
            infos.append(info)

        forward = _representation_step(module, device)
        state = _initial_state(module, n_envs, device)
        while scored < n_steps:
            batch = np.stack(observations)
            _, actions, state = forward(batch, state)
            steps = np.asarray(
                [int(info["decision_step"]) for info in infos],
                dtype=np.int64,
            )
            eligible = steps >= warmup
            remaining = n_steps - scored
            eligible_indices = np.flatnonzero(eligible)
            take = (
                int(eligible_indices[remaining - 1]) + 1
                if len(eligible_indices) >= remaining
                else n_envs
            )
            joint = np.stack([info["belief_current"] for info in infos])
            factors = np.stack(factor_marginals(joint, (3, 3)), axis=1)
            values = (
                batch,
                joint,
                factors,
                episode_ids.copy(),
                steps,
                eligible,
            )
            for name, value in zip(rows, values):
                rows[name].append(np.asarray(value[:take]).copy())
            scored += int(eligible[:take].sum())
            if scored == n_steps:
                break

            reset_indices = []
            for index, env in enumerate(envs):
                observation, _, terminated, truncated, info = env.step(
                    int(actions[index])
                )
                if terminated or truncated:
                    episode_indices[index] += 1
                    episode_ids[index] = next_episode_id
                    next_episode_id += 1
                    observation, info = reset(index)
                    reset_indices.append(index)
                observations[index] = observation
                infos[index] = info
            if reset_indices:
                fresh = _initial_state(module, len(reset_indices), device)
                indices = torch.as_tensor(
                    reset_indices,
                    dtype=torch.long,
                    device=device,
                )
                for key, value in state.items():
                    value.index_copy_(0, indices, fresh[key])
    finally:
        for env in envs:
            env.close()

    data = HistoryData(
        **{
            name: np.concatenate(chunks, axis=0)
            for name, chunks in rows.items()
        },
        metadata={
            "condition": condition,
            "n_steps": n_steps,
            "n_envs": n_envs,
            "warmup": warmup,
            "policy_mode": "greedy_argmax",
            "sampling_distribution": "process_weighted_rollout",
            "representation": "final_block_residual_before_final_layer_norm",
        },
    )
    if int(data.mask.sum()) != n_steps:
        raise AssertionError("history collection did not retain the requested rows")
    _episode_members(data)
    return data


def _decode_observations(
    data: HistoryData,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(data.observations, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] != 18:
        raise ValueError("observations must have shape (n, 18)")
    tokens = observations[:, :9]
    actions = observations[:, 9:]
    if not np.allclose(tokens.sum(axis=1), 1.0):
        raise ValueError("every observation must contain one joint token")
    if not np.all((tokens == 0.0) | (tokens == 1.0)):
        raise ValueError("joint tokens must be one-hot")
    reset = data.episode_steps == 0
    if not np.allclose(actions[reset], 0.0):
        raise ValueError("reset observations must have a blank action")
    if not np.allclose(actions[~reset].sum(axis=1), 1.0):
        raise ValueError("non-reset observations must contain one previous action")
    if not np.all((actions == 0.0) | (actions == 1.0)):
        raise ValueError("previous actions must be one-hot or blank")
    token_indices = tokens.argmax(axis=1).astype(np.int64)
    action_indices = np.full(len(observations), -1, dtype=np.int64)
    action_indices[~reset] = actions[~reset].argmax(axis=1)
    return token_indices, action_indices


def shuffled_histories(data: HistoryData, *, seed: int) -> HistoryData:
    rng = np.random.default_rng(seed)
    order = np.arange(len(data.observations))
    for indices in _episode_members(data):
        order[indices[1:]] = rng.permutation(indices[1:])
    shuffled = data.observations[order].copy()
    output = replace(
        data,
        observations=shuffled,
        metadata={
            **data.metadata,
            "history_control": (
                "complete token/previous-action pairs permuted within episodes; "
                "reset row fixed"
            ),
            "history_seed": int(seed),
        },
    )
    _decode_observations(output)
    return output


def empirical_random_histories(
    data: HistoryData,
    *,
    seed: int,
) -> HistoryData:
    rng = np.random.default_rng(seed)
    reset = data.episode_steps == 0
    nonreset = ~reset
    if not reset.any() or not nonreset.any():
        raise ValueError("empirical randomization requires reset and non-reset rows")
    observations = np.empty_like(data.observations)
    reset_pool = data.observations[reset]
    nonreset_pool = data.observations[nonreset]
    observations[reset] = reset_pool[
        rng.integers(len(reset_pool), size=int(reset.sum()))
    ]
    observations[nonreset] = nonreset_pool[
        rng.integers(len(nonreset_pool), size=int(nonreset.sum()))
    ]
    output = replace(
        data,
        observations=observations,
        metadata={
            **data.metadata,
            "history_control": (
                "valid token/previous-action pairs sampled independently with "
                "replacement from the split's empirical reset/non-reset pools"
            ),
            "history_seed": int(seed),
        },
    )
    _decode_observations(output)
    return output


def uniform_random_histories(
    data: HistoryData,
    *,
    seed: int,
) -> HistoryData:
    rng = np.random.default_rng(seed)
    reset = data.episode_steps == 0
    tokens = rng.integers(9, size=len(data.observations))
    actions = rng.integers(9, size=len(data.observations))
    observations = np.zeros_like(data.observations)
    observations[np.arange(len(observations)), tokens] = 1.0
    nonreset_rows = np.flatnonzero(~reset)
    observations[nonreset_rows, 9 + actions[nonreset_rows]] = 1.0
    output = replace(
        data,
        observations=observations,
        metadata={
            **data.metadata,
            "history_control": (
                "joint token and previous action sampled independently and "
                "uniformly from valid discrete inputs; reset action blank"
            ),
            "history_seed": int(seed),
        },
    )
    _decode_observations(output)
    return output


def replay_beliefs(
    data: HistoryData,
    *,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    tokens, actions = _decode_observations(data)
    env = HMMEnv(environment_config(condition))
    try:
        initial = np.asarray(env.model.initial_distribution, dtype=np.float64)
        emission = np.asarray(env.model.emission_matrix, dtype=np.float64)
        transition = tuple(
            np.asarray(
                env.task.transition_matrix_for_action(action),
                dtype=np.float64,
            )
            for action in range(env.action_space.n)
        )
    finally:
        env.close()
    beliefs = np.empty((len(tokens), 9), dtype=np.float64)
    for indices in _episode_members(data):
        belief = initial.copy()
        for offset, index in enumerate(indices):
            if offset:
                belief = belief @ transition[int(actions[index])]
            belief = belief * emission[:, int(tokens[index])]
            total = float(belief.sum())
            if total <= 0.0:
                raise ValueError("randomized history is impossible under the model")
            belief = belief / total
            beliefs[index] = belief
    factors = np.stack(factor_marginals(beliefs, (3, 3)), axis=1)
    reconstructed = product_distribution(
        [factors[:, index] for index in range(2)]
    )
    if not np.allclose(reconstructed, beliefs, atol=1e-12, rtol=0.0):
        raise ValueError("replayed joint beliefs do not factorize")
    return beliefs, factors


@torch.inference_mode()
def replay_activations(
    module: object,
    data: HistoryData,
    *,
    device: torch.device,
) -> np.ndarray:
    groups = _episode_members(data)
    if not groups:
        raise ValueError("activation replay requires at least one episode")
    batch_size = int(data.metadata["n_envs"])
    activations: np.ndarray | None = None
    forward = _representation_step(module, device)
    for start in range(0, len(groups), batch_size):
        batch_groups = groups[start : start + batch_size]
        state = _initial_state(module, len(batch_groups), device)
        for step in range(max(map(len, batch_groups))):
            indices = np.asarray(
                [group[min(step, len(group) - 1)] for group in batch_groups]
            )
            residual, _, state = forward(data.observations[indices], state)
            if activations is None:
                activations = np.empty(
                    (len(data.observations), residual.shape[1]),
                    dtype=np.float64,
                )
            active = np.asarray([step < len(group) for group in batch_groups])
            activations[indices[active]] = residual[active]
    if activations is None:
        raise AssertionError("activation replay produced no values")
    return activations


def _targets(
    joint: np.ndarray,
    factors: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "joint_mixed_state": np.asarray(joint, dtype=np.float64),
        "factor_1": np.asarray(factors[:, 0], dtype=np.float64),
        "factor_2": np.asarray(factors[:, 1], dtype=np.float64),
    }


def _score(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    train_groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    weight, bias, fit = fit_grouped_affine(
        train_features,
        train_target,
        train_groups,
        seed=seed,
    )
    metrics = score_prediction(test_features @ weight + bias, test_target)
    return {
        "mse": metrics["mse"],
        "target_variance": metrics["target_variance"],
        "global_mse_ratio": metrics["normalized_mse"],
        "r_squared": metrics["r_squared"],
        "n_evaluated": metrics["n_evaluated"],
        "fit": {
            "method": fit["method"],
            "selected_rcond": fit["selected_rcond"],
            "rank": fit["rank"],
            "n_samples": fit["n_samples"],
            "n_groups": fit["n_groups"],
        },
    }


def _mean_metrics(repeats: list[dict[str, object]]) -> dict[str, object]:
    names = ("mse", "target_variance", "global_mse_ratio", "r_squared")
    return {
        "mean": {
            name: float(np.mean([repeat[name] for repeat in repeats]))
            for name in names
        },
        "repeats": repeats,
    }


def analyze_module_controls(
    module: object,
    *,
    condition: str,
    seed: int,
    n_steps: int = FULL_STEPS,
    device: str | torch.device = "cpu",
    n_null_repeats: int = N_NULL_REPEATS,
) -> dict[str, object]:
    seed = _integer(seed, "seed", 0)
    n_steps = _integer(n_steps, "n_steps", 1)
    n_null_repeats = _integer(n_null_repeats, "n_null_repeats", 1)
    device = torch.device(device)
    probe_streams = named_seed_sequences(seed, _PROBE_STREAMS)
    train = collect_history_data(
        module,
        condition=condition,
        n_steps=n_steps,
        seed=probe_streams["probe_train"],
        device=device,
    )
    test = collect_history_data(
        module,
        condition=condition,
        n_steps=n_steps,
        seed=probe_streams["probe_test"],
        device=device,
    )
    replayed = (
        replay_beliefs(train, condition=condition),
        replay_beliefs(test, condition=condition),
    )
    replay_error = max(
        float(np.max(np.abs(replayed[0][0] - train.beliefs))),
        float(np.max(np.abs(replayed[1][0] - test.beliefs))),
    )
    if replay_error > 1e-10:
        raise ValueError(
            f"belief replay disagrees with environment diagnostics: {replay_error}"
        )

    control_streams = named_seed_sequences(seed, _CONTROL_STREAMS)
    variants = {
        "intact": (train, test),
        "shuffled_history": (
            shuffled_histories(
                train,
                seed=seed_sequence_to_int(control_streams["shuffle"]),
            ),
            shuffled_histories(
                test,
                seed=seed_sequence_to_int(
                    child_seed_sequence(control_streams["shuffle"], 1)
                ),
            ),
        ),
        "empirical_random_history": (
            empirical_random_histories(
                train,
                seed=seed_sequence_to_int(control_streams["empirical_random"]),
            ),
            empirical_random_histories(
                test,
                seed=seed_sequence_to_int(
                    child_seed_sequence(
                        control_streams["empirical_random"],
                        1,
                    )
                ),
            ),
        ),
        "uniform_random_history": (
            uniform_random_histories(
                train,
                seed=seed_sequence_to_int(control_streams["uniform_random"]),
            ),
            uniform_random_histories(
                test,
                seed=seed_sequence_to_int(
                    child_seed_sequence(control_streams["uniform_random"], 1)
                ),
            ),
        ),
    }
    variant_data = {}
    for name, (fit_data, test_data) in variants.items():
        variant_data[name] = {
            "train_activations": replay_activations(
                module,
                fit_data,
                device=device,
            )[fit_data.mask],
            "test_activations": replay_activations(
                module,
                test_data,
                device=device,
            )[test_data.mask],
            "train_recomputed": _targets(
                *replay_beliefs(fit_data, condition=condition)
            ),
            "test_recomputed": _targets(
                *replay_beliefs(test_data, condition=condition)
            ),
            "train_metadata": dict(fit_data.metadata),
            "test_metadata": dict(test_data.metadata),
        }

    original_train = _targets(train.beliefs, train.factor_beliefs)
    original_test = _targets(test.beliefs, test.factor_beliefs)
    train_groups = train.episode_ids[train.mask]
    null_stream = control_streams["nulls"]
    report: dict[str, object] = {
        "schema_version": 1,
        "metadata": {
            "condition": condition,
            "seed": seed,
            "n_fit": n_steps,
            "n_test": n_steps,
            "n_envs": N_ENVS,
            "warmup": WARMUP,
            "n_null_repeats": n_null_repeats,
            "device": str(device),
            "policy_mode": "greedy_argmax",
            "fit_protocol": (
                "independent fit/test rollouts; training-only whole-episode "
                "SVD-cutoff cross-validation"
            ),
            "history_controls": {
                name: data["train_metadata"].get("history_control", "none")
                for name, data in variant_data.items()
            },
            "history_control_seeds": {
                name: {
                    "fit": data["train_metadata"].get("history_seed"),
                    "test": data["test_metadata"].get("history_seed"),
                }
                for name, data in variant_data.items()
            },
            "original_label_interpretation": (
                "deliberately broken history/target alignment"
            ),
            "recomputed_target_interpretation": (
                "exact Bayesian targets for the counterfactual replayed history"
            ),
            "scope_warning": (
                "Frozen replay and probe nulls test linear accessibility, not "
                "causal policy use. Random histories are distribution shifts."
            ),
        },
        "belief_replay_max_abs_error": replay_error,
        "targets": {},
    }
    for target_index, target in enumerate(TARGETS):
        target_report: dict[str, object] = {}
        for variant, data in variant_data.items():
            target_report[variant] = {
                "original_target": _score(
                    data["train_activations"],
                    data["test_activations"],
                    original_train[target][train.mask],
                    original_test[target][test.mask],
                    train_groups,
                    seed=seed + target_index,
                ),
                "recomputed_target": _score(
                    data["train_activations"],
                    data["test_activations"],
                    data["train_recomputed"][target][train.mask],
                    data["test_recomputed"][target][test.mask],
                    train_groups,
                    seed=seed + target_index,
                ),
            }

        label_repeats = []
        gaussian_repeats = []
        intact = variant_data["intact"]
        for repeat in range(n_null_repeats):
            repeat_seed = seed_sequence_to_int(
                child_seed_sequence(null_stream, repeat)
            )
            rng = np.random.default_rng(repeat_seed)
            label_repeats.append(
                _score(
                    intact["train_activations"],
                    intact["test_activations"],
                    original_train[target][train.mask][
                        rng.permutation(n_steps)
                    ],
                    original_test[target][test.mask],
                    train_groups,
                    seed=repeat_seed + target_index,
                )
            )
            null_train, null_test, sampling = gaussian_feature_null(
                intact["train_activations"],
                n_steps,
                seed=repeat_seed,
            )
            gaussian_score = _score(
                null_train,
                null_test,
                original_train[target][train.mask],
                original_test[target][test.mask],
                train_groups,
                seed=repeat_seed + target_index,
            )
            gaussian_score["sampling"] = {
                key: value
                for key, value in sampling.items()
                if key not in {"training_mean"}
            }
            gaussian_repeats.append(gaussian_score)
        target_report["nulls"] = {
            "shuffled_training_labels": _mean_metrics(label_repeats),
            "gaussian_covariance_matched_features": _mean_metrics(
                gaussian_repeats
            ),
        }
        report["targets"][target] = target_report
    return report


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=FULL_STEPS)
    parser.add_argument("--null-repeats", type=int, default=N_NULL_REPEATS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    checkpoint = args.checkpoint.resolve()
    module = load_module_only(checkpoint)
    report = analyze_module_controls(
        module,
        condition=args.condition,
        seed=args.seed,
        n_steps=SMOKE_STEPS if args.smoke else args.steps,
        device=args.device,
        n_null_repeats=1 if args.smoke else args.null_repeats,
    )
    repository = Path(__file__).resolve().parents[2]
    harness = repository.parent / "rl-harness"
    report["provenance"] = {
        "checkpoint": str(checkpoint),
        "checkpoint_files": {
            path.name: _sha256(path)
            for path in sorted(checkpoint.iterdir())
            if path.is_file()
        },
        "experiment_git_commit": _git(repository, "rev-parse", "HEAD"),
        "harness_git_commit": _git(harness, "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
