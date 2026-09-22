"""Exact coarse-filter probes for cycle-7 Reward-State Quotient agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from analysis.probes import predictive_belief_update
from envs.hmm import HMMEnv
from harness.seeding import named_seed_sequences

from experiments.mess3_reward_state_action_symmetry_cycle_7.component_probe_analysis import (
    FIT_STEPS,
    RIDGE,
    TEST_STEPS,
    WARMUP,
    score_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_analysis import (
    N_ENVS,
    HistoryData,
    collect_history_data,
    replay_beliefs,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    environment_config,
)


VARIANT = 2
CONDITION = "variant_2_ctx32_ent"
_PROBE_STREAMS = {
    "probe_train": (510,),
    "probe_test": (511,),
}
TARGETS = {
    "coarse_b2": {
        "symbol": "c_t",
        "definition": (
            "P(B={state 3} | tokens A/B coarsened to not-C, token C, "
            "and executed actions)"
        ),
        "description": "separately updated exact two-state coarse belief",
    },
    "full_rho": {
        "symbol": "rho_t",
        "definition": "b_3,t",
        "description": "state-3 component of the exact full three-state belief",
    },
}


def coarse_filter_spec(
    *,
    variant: int = VARIANT,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Construct and validate the exact A={1,2}, B={3} controlled HMM."""

    if variant != VARIANT:
        raise ValueError("the cycle-7 coarse probe is defined only for variant 2")
    environment = HMMEnv(environment_config(variant))
    try:
        emission = np.asarray(
            environment.model.emission_matrix,
            dtype=np.float64,
        )
        coarse_emission = np.column_stack(
            (emission[:, :2].sum(axis=1), emission[:, 2])
        )
        if not np.allclose(
            coarse_emission[0],
            coarse_emission[1],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "states 1 and 2 do not have equal coarsened emissions"
            )
        lumped_emission = np.stack(
            (coarse_emission[0], coarse_emission[2])
        )
        expected = np.asarray(
            [[0.925, 0.075], [0.15, 0.85]],
            dtype=np.float64,
        )
        if not np.allclose(
            lumped_emission,
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"unexpected coarse emission matrix: {lumped_emission}"
            )

        transitions: dict[int, np.ndarray] = {}
        for action in range(environment.action_space.n):
            full = np.asarray(
                environment.task.transition_matrix_for_action(action),
                dtype=np.float64,
            )
            destination_lumps = np.column_stack(
                (full[:, :2].sum(axis=1), full[:, 2])
            )
            if not np.allclose(
                destination_lumps[0],
                destination_lumps[1],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"action {action} is not strongly lumpable over states 1/2"
                )
            lumped = np.stack(
                (destination_lumps[0], destination_lumps[2])
            )
            if not np.allclose(
                lumped.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"action {action} produced a non-stochastic coarse transition"
                )
            transitions[action] = lumped

        initial = np.asarray(
            environment.model.initial_distribution,
            dtype=np.float64,
        )
    finally:
        environment.close()
    coarse_initial = np.asarray(
        [initial[:2].sum(), initial[2]],
        dtype=np.float64,
    )
    return coarse_initial, lumped_emission, transitions


def _episode_members(data: HistoryData) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for episode_id in dict.fromkeys(map(int, data.episode_ids)):
        indices = np.flatnonzero(data.episode_ids == episode_id)
        if not np.array_equal(
            data.episode_steps[indices],
            np.arange(len(indices), dtype=np.int64),
        ):
            raise ValueError(
                "each episode must contain a complete ordered prefix from reset"
            )
        groups.append(indices)
    return groups


def _tokens_and_previous_actions(
    data: HistoryData,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(data.observations, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] != 6:
        raise ValueError("observations must have shape (n, 6)")
    token_rows = observations[:, :3]
    action_rows = observations[:, 3:]
    if not np.allclose(token_rows.sum(axis=1), 1.0):
        raise ValueError("every observation must contain one token")
    reset = data.episode_steps == 0
    if not np.allclose(action_rows[reset], 0.0):
        raise ValueError("reset observations must have a blank action")
    if not np.allclose(action_rows[~reset].sum(axis=1), 1.0):
        raise ValueError(
            "non-reset observations must contain the previous executed action"
        )
    tokens = token_rows.argmax(axis=1).astype(np.int64)
    previous_actions = np.full(len(observations), -1, dtype=np.int64)
    previous_actions[~reset] = action_rows[~reset].argmax(axis=1)
    return tokens, previous_actions


def coarse_beliefs(
    data: HistoryData,
    *,
    variant: int = VARIANT,
) -> np.ndarray:
    """Replay the exact coarse filter over complete episode prefixes."""

    initial, emission, transitions = coarse_filter_spec(variant=variant)
    tokens, previous_actions = _tokens_and_previous_actions(data)
    beliefs = np.empty((len(tokens), 2), dtype=np.float64)
    for indices in _episode_members(data):
        current = initial.copy()
        for offset, index in enumerate(indices):
            coarse_token = 1 if int(tokens[index]) == 2 else 0
            measurement = np.diag(emission[:, coarse_token])
            if offset == 0:
                current = predictive_belief_update(initial, measurement)
            else:
                current = predictive_belief_update(
                    current,
                    transitions[int(previous_actions[index])] @ measurement,
                )
            beliefs[index] = current
    return beliefs


def _coarse_model_record(
    initial: np.ndarray,
    emission: np.ndarray,
    transitions: dict[int, np.ndarray],
) -> dict[str, object]:
    return {
        "state_partition": {
            "A": ["state 1", "state 2"],
            "B": ["state 3"],
        },
        "observation_partition": {
            "not-C": ["token A", "token B"],
            "C": ["token C"],
        },
        "initial": initial.tolist(),
        "emission_rows_A_B_columns_not_C_C": emission.tolist(),
        "transition_rows_A_B_columns_A_B": {
            str(action): matrix.tolist()
            for action, matrix in transitions.items()
        },
        "implementation_action_names": {
            "0": "noop",
            "1": "positive",
            "2": "negative",
        },
        "manuscript_action_names": {
            "0": "Noop",
            "1": "Tilt X",
            "2": "Tilt Y",
        },
    }


def analyze_module_coarse_probe(
    module: object,
    *,
    variant: int,
    seed: int,
    fit_steps: int = FIT_STEPS,
    test_steps: int = TEST_STEPS,
    warmup: int = WARMUP,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Fit coarse and projected-full scalar probes on paired rollouts."""

    if variant != VARIANT:
        raise ValueError("the coarse probe campaign is variant-2 only")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if fit_steps < 1 or test_steps < 1:
        raise ValueError("fit_steps and test_steps must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    device = torch.device(device)
    streams = named_seed_sequences(seed, _PROBE_STREAMS)
    train = collect_history_data(
        module,
        variant=variant,
        n_steps=fit_steps,
        seed=streams["probe_train"],
        device=device,
        warmup=warmup,
    )
    test = collect_history_data(
        module,
        variant=variant,
        n_steps=test_steps,
        seed=streams["probe_test"],
        device=device,
        warmup=warmup,
    )
    replay_error = max(
        float(
            np.max(
                np.abs(
                    replay_beliefs(train, variant=variant) - train.beliefs
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    replay_beliefs(test, variant=variant) - test.beliefs
                )
            )
        ),
    )
    if replay_error > 1e-10:
        raise ValueError(
            "full belief replay disagrees with environment diagnostics: "
            f"{replay_error}"
        )

    train_coarse = coarse_beliefs(train, variant=variant)[train.mask, 1:2]
    test_coarse = coarse_beliefs(test, variant=variant)[test.mask, 1:2]
    train_full = train.beliefs[train.mask, 2:3].astype(
        np.float64,
        copy=False,
    )
    test_full = test.beliefs[test.mask, 2:3].astype(
        np.float64,
        copy=False,
    )
    train_features = train.activations[train.mask]
    test_features = test.activations[test.mask]
    targets = {
        "coarse_b2": {
            **TARGETS["coarse_b2"],
            **score_target(
                train_features,
                test_features,
                train_coarse,
                test_coarse,
            ),
        },
        "full_rho": {
            **TARGETS["full_rho"],
            **score_target(
                train_features,
                test_features,
                train_full,
                test_full,
            ),
        },
    }
    differences = np.concatenate(
        (
            train_coarse[:, 0] - train_full[:, 0],
            test_coarse[:, 0] - test_full[:, 0],
        )
    )
    initial, emission, transitions = coarse_filter_spec(variant=variant)
    return {
        "schema_version": 1,
        "metadata": {
            "variant": variant,
            "condition": CONDITION,
            "seed": seed,
            "n_fit": fit_steps,
            "n_test": test_steps,
            "n_envs": N_ENVS,
            "warmup": warmup,
            "device": str(device),
            "policy_mode": "greedy_argmax",
            "sampling_distribution": "process_weighted_rollout",
            "representation": "post_final_layer_norm",
            "fit_protocol": (
                "fixed-ridge affine probe fit on one rollout stream and "
                "scored on an independent rollout stream"
            ),
            "ridge": RIDGE,
            "seed_streams": {
                name: list(key) for name, key in _PROBE_STREAMS.items()
            },
            "stream_relationship": (
                "The coarse and projected-full targets share each paired "
                "fit/test rollout, and the streams match the cycle-7 scalar "
                "component campaign."
            ),
            "history_protocol": (
                "Both filters consume complete episode prefixes; the "
                f"{warmup}-step warmup mask is applied only after recursive "
                "filtering."
            ),
            "target_timing": (
                "At reset, condition the prior on the first token. Later "
                "tokens are conditioned after the transition induced by the "
                "previous executed action encoded in the observation."
            ),
            "scope_warning": (
                "Affine decodability measures linear accessibility, not "
                "causal policy use or unique identification of a filter."
            ),
        },
        "coarse_model": _coarse_model_record(
            initial,
            emission,
            transitions,
        ),
        "full_belief_replay_max_abs_error": replay_error,
        "target_difference": {
            "mse": float(np.mean(np.square(differences))),
            "max_abs_difference": float(np.max(np.abs(differences))),
            "n_compared": int(len(differences)),
            "interpretation": (
                "The coarse filter discards token A/B identity; it is not the "
                "state-3 projection of the full filter."
            ),
        },
        "targets": targets,
    }


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
    parser.add_argument("--variant", type=int, default=VARIANT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fit-steps", type=int, default=FIT_STEPS)
    parser.add_argument("--test-steps", type=int, default=TEST_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    repository = Path(__file__).resolve().parents[2]
    harness = Path(__import__("analysis").__file__).resolve().parents[1]
    module = load_module_only(args.checkpoint)
    report = analyze_module_coarse_probe(
        module,
        variant=args.variant,
        seed=args.seed,
        fit_steps=args.fit_steps,
        test_steps=args.test_steps,
        warmup=args.warmup,
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
