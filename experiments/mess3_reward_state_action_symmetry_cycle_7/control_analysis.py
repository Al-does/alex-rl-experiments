"""Frozen-history controls for the cycle-7 Belief-Quotient Ladder."""

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

from analysis.belief_geometry import evaluate_belief_geometry
from analysis.checkpoints import load_module_only
from envs.hmm import HMMEnv
from harness.seeding import (
    child_seed_sequence,
    named_seed_sequences,
    seed_sequence_to_int,
)

from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    environment_config,
)


FULL_STEPS = 20_000
SMOKE_STEPS = 512
N_ENVS = 16
WARMUP = 128
_PROBE_STREAMS = {
    "probe_train": (700,),
    "probe_test": (701,),
}
_ROLLOUT_STREAMS = {
    "episode_seeds": (0,),
    "action_spaces": (1,),
}
_CONTROL_STREAMS = {
    "shuffle": (0,),
    "empirical_random": (1,),
}


@dataclass(frozen=True, slots=True)
class HistoryData:
    observations: np.ndarray
    activations: np.ndarray
    beliefs: np.ndarray
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
        embedding, state_out = module.encode_step(tensor, state)
        logits = module.action_distribution_inputs(embedding)
        actions = logits.argmax(dim=-1)
        return (
            embedding.cpu().numpy(),
            actions.cpu().numpy(),
            state_out,
        )

    module.to(device).eval()
    return forward


@torch.inference_mode()
def collect_history_data(
    module: object,
    *,
    variant: int,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
    n_envs: int = N_ENVS,
    warmup: int = WARMUP,
) -> HistoryData:
    n_steps = _integer(n_steps, "n_steps", 1)
    n_envs = min(_integer(n_envs, "n_envs", 1), n_steps)
    warmup = _integer(warmup, "warmup", 0)
    config = environment_config(variant)
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
            "activations",
            "beliefs",
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
                env.model.n_states != 3
                or env.model.n_tokens != 3
                or env.action_space.n != 3
                or env.observation_space.shape != (6,)
            ):
                raise ValueError(
                    "control collection requires the cycle-7 MESS3 recipe"
                )
            observation, info = reset(index)
            observations.append(observation)
            infos.append(info)

        forward = _representation_step(module, device)
        state = _initial_state(module, n_envs, device)
        while scored < n_steps:
            batch = np.stack(observations)
            activations, actions, state = forward(batch, state)
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
            beliefs = np.stack([info["belief_current"] for info in infos])
            values = (
                batch,
                activations,
                beliefs,
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
            "variant": variant,
            "n_steps": n_steps,
            "n_envs": n_envs,
            "warmup": warmup,
            "policy_mode": "greedy_argmax",
            "sampling_distribution": "process_weighted_rollout",
            "representation": "post_final_layer_norm",
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
    if observations.ndim != 2 or observations.shape[1] != 6:
        raise ValueError("observations must have shape (n, 6)")
    tokens = observations[:, :3]
    actions = observations[:, 3:]
    if not np.allclose(tokens.sum(axis=1), 1.0):
        raise ValueError("every observation must contain one token")
    if not np.all((tokens == 0.0) | (tokens == 1.0)):
        raise ValueError("tokens must be one-hot")
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
    output = replace(
        data,
        observations=data.observations[order].copy(),
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


def replay_beliefs(data: HistoryData, *, variant: int) -> np.ndarray:
    tokens, actions = _decode_observations(data)
    env = HMMEnv(environment_config(variant))
    try:
        initial = np.asarray(env.model.initial_distribution, dtype=np.float64)
        emission = np.asarray(env.model.emission_matrix, dtype=np.float64)
        transitions = tuple(
            np.asarray(
                env.task.transition_matrix_for_action(action),
                dtype=np.float64,
            )
            for action in range(env.action_space.n)
        )
    finally:
        env.close()
    beliefs = np.empty((len(tokens), 3), dtype=np.float64)
    for indices in _episode_members(data):
        belief = initial.copy()
        for offset, index in enumerate(indices):
            if offset:
                belief = belief @ transitions[int(actions[index])]
            belief = belief * emission[:, int(tokens[index])]
            total = float(belief.sum())
            if total <= 0.0:
                raise ValueError("randomized history is impossible under the model")
            belief = belief / total
            beliefs[index] = belief
    return beliefs


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
            embedding, _, state = forward(data.observations[indices], state)
            if activations is None:
                activations = np.empty(
                    (len(data.observations), embedding.shape[1]),
                    dtype=np.float64,
                )
            active = np.asarray([step < len(group) for group in batch_groups])
            activations[indices[active]] = embedding[active]
    if activations is None:
        raise AssertionError("activation replay produced no values")
    return activations


def _score(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    battery = evaluate_belief_geometry(
        {"post_final_layer_norm": train_features},
        {"post_final_layer_norm": test_features},
        train_target,
        test_target,
        train_groups=train_groups,
        test_groups=test_groups,
        seed=seed,
        n_null_repeats=1,
        n_resamples=1,
    )
    record = battery.report["representations"]["post_final_layer_norm"]
    metrics = record["metrics"]
    return {
        "mse": metrics["mse"],
        "target_variance": metrics["target_variance"],
        "global_mse_ratio": metrics["normalized_mse"],
        "r_squared": metrics["r_squared"],
        "n_evaluated": metrics["n_evaluated"],
        "fit": record["fit"],
    }


def analyze_module_controls(
    module: object,
    *,
    variant: int,
    seed: int,
    n_steps: int = FULL_STEPS,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    variant = _integer(variant, "variant", 1)
    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    seed = _integer(seed, "seed", 0)
    n_steps = _integer(n_steps, "n_steps", 1)
    device = torch.device(device)
    probe_streams = named_seed_sequences(seed, _PROBE_STREAMS)
    train = collect_history_data(
        module,
        variant=variant,
        n_steps=n_steps,
        seed=probe_streams["probe_train"],
        device=device,
    )
    test = collect_history_data(
        module,
        variant=variant,
        n_steps=n_steps,
        seed=probe_streams["probe_test"],
        device=device,
    )
    replay_error = max(
        float(np.max(np.abs(replay_beliefs(train, variant=variant) - train.beliefs))),
        float(np.max(np.abs(replay_beliefs(test, variant=variant) - test.beliefs))),
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
                    child_seed_sequence(control_streams["empirical_random"], 1)
                ),
            ),
        ),
    }
    train_groups = train.episode_ids[train.mask]
    test_groups = test.episode_ids[test.mask]
    report: dict[str, object] = {
        "schema_version": 1,
        "metadata": {
            "variant": variant,
            "seed": seed,
            "n_fit": n_steps,
            "n_test": n_steps,
            "n_envs": N_ENVS,
            "warmup": WARMUP,
            "device": str(device),
            "policy_mode": "greedy_argmax",
            "representation": "post_final_layer_norm",
            "fit_protocol": (
                "independent fit/test rollouts; training-only whole-episode "
                "SVD-cutoff cross-validation"
            ),
            "history_controls": {
                name: fit_data.metadata.get("history_control", "none")
                for name, (fit_data, _) in variants.items()
            },
            "history_control_seeds": {
                name: {
                    "fit": fit_data.metadata.get("history_seed"),
                    "test": test_data.metadata.get("history_seed"),
                }
                for name, (fit_data, test_data) in variants.items()
            },
            "original_label_interpretation": (
                "time-indexed exact beliefs from the unmodified on-policy rollout"
            ),
            "recomputed_target_interpretation": (
                "exact Bayesian beliefs for the counterfactual replayed history"
            ),
            "scope_warning": (
                "Frozen replay tests linear accessibility, not causal policy use. "
                "Random histories are distribution shifts."
            ),
        },
        "belief_replay_max_abs_error": replay_error,
        "controls": {},
    }
    for index, (name, (fit_data, test_data)) in enumerate(variants.items()):
        if name == "intact":
            train_activations = train.activations
            test_activations = test.activations
        else:
            train_activations = replay_activations(
                module,
                fit_data,
                device=device,
            )
            test_activations = replay_activations(
                module,
                test_data,
                device=device,
            )
        report["controls"][name] = {
            "original_target": _score(
                train_activations[train.mask],
                test_activations[test.mask],
                train.beliefs[train.mask],
                test.beliefs[test.mask],
                train_groups,
                test_groups,
                seed=seed + index,
            ),
            "recomputed_target": _score(
                train_activations[train.mask],
                test_activations[test.mask],
                replay_beliefs(fit_data, variant=variant)[train.mask],
                replay_beliefs(test_data, variant=variant)[test.mask],
                train_groups,
                test_groups,
                seed=seed + index,
            ),
        }
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
    parser.add_argument("--variant", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=FULL_STEPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    repository = Path(__file__).resolve().parents[2]
    harness = Path(__import__("analysis").__file__).resolve().parents[1]
    module = load_module_only(args.checkpoint)
    report = analyze_module_controls(
        module,
        variant=args.variant,
        seed=args.seed,
        n_steps=args.steps,
        device=args.device,
    )
    report["source"] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_files": {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(args.checkpoint.iterdir())
            if path.is_file()
        },
        "experiment_git_commit": _git(repository, "rev-parse", "HEAD"),
        "experiment_git_dirty": bool(
            _git(repository, "status", "--porcelain")
        ),
        "harness_git_commit": _git(harness, "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
