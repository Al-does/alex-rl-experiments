"""Exact coarse-observation probes for the cycle-7 quotient ladder."""

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


CONDITIONS = {
    1: "variant_1_ctx32_ent",
    2: "variant_2_ctx32_ent",
    3: "variant_3_ctx32_ent",
}
CONDITION_LABELS = {
    1: "Trivial Quotient",
    2: "Reward-State Quotient",
    3: "Full-State Quotient",
}
_PROBE_STREAMS = {
    "probe_train": (510,),
    "probe_test": (511,),
}
TARGETS = {
    "coarse_b2": {
        "symbol": "c_t[B]",
        "definition": (
            "P(state 3 | tokens A/B coarsened to not-C, token C, and executed actions)"
        ),
        "description": "exact posterior from the coarse-observation history",
    },
    "full_rho": {
        "symbol": "rho_t",
        "definition": "b_3,t",
        "description": "state-3 component of the exact full-observation belief",
    },
}


def _build_coarse_filter_spec(
    variant: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    dict[str, object],
]:
    if variant not in CONDITIONS:
        raise ValueError("variant must be one of 1, 2, or 3")
    environment = HMMEnv(environment_config(variant))
    try:
        full_emission = np.asarray(
            environment.model.emission_matrix,
            dtype=np.float64,
        )
        coarse_emission = np.column_stack(
            (full_emission[:, :2].sum(axis=1), full_emission[:, 2])
        )
        if not np.allclose(
            coarse_emission[0],
            coarse_emission[1],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("states 1 and 2 do not have equal coarsened emissions")
        expected_emission = np.asarray(
            [[0.925, 0.075], [0.15, 0.85]],
            dtype=np.float64,
        )
        if not np.allclose(
            np.stack((coarse_emission[0], coarse_emission[2])),
            expected_emission,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"unexpected coarse emission matrix: {coarse_emission}")

        full_transitions: dict[int, np.ndarray] = {}
        lumped_transitions: dict[int, np.ndarray] = {}
        lumpability_errors: dict[int, float] = {}
        for action in range(environment.action_space.n):
            full = np.asarray(
                environment.task.transition_matrix_for_action(action),
                dtype=np.float64,
            )
            full_transitions[action] = full
            destination_lumps = np.column_stack((full[:, :2].sum(axis=1), full[:, 2]))
            lumpability_errors[action] = float(
                np.max(np.abs(destination_lumps[0] - destination_lumps[1]))
            )
            lumped_transitions[action] = np.stack(
                (destination_lumps[0], destination_lumps[2])
            )

        initial = np.asarray(
            environment.model.initial_distribution,
            dtype=np.float64,
        )
    finally:
        environment.close()

    non_lumpable_actions = [
        action for action, error in lumpability_errors.items() if error > 1e-12
    ]
    if variant in (1, 2):
        if non_lumpable_actions:
            raise ValueError(
                "states 1 and 2 are not strongly lumpable for actions "
                f"{non_lumpable_actions}"
            )
        filter_initial = np.asarray(
            [initial[:2].sum(), initial[2]],
            dtype=np.float64,
        )
        filter_emission = np.stack((coarse_emission[0], coarse_emission[2]))
        transitions = lumped_transitions
        filter_states = {
            "A": ["state 1", "state 2"],
            "B": ["state 3"],
        }
        filter_kind = "exact_two_state_quotient"
    else:
        if not non_lumpable_actions:
            raise ValueError("variant 3 unexpectedly admits a two-state quotient")
        filter_initial = initial
        filter_emission = coarse_emission
        transitions = full_transitions
        filter_states = {
            "state 1": ["state 1"],
            "state 2": ["state 2"],
            "state 3": ["state 3"],
        }
        filter_kind = "exact_full_three_state"

    for action, transition in transitions.items():
        if not np.allclose(
            transition.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"action {action} produced a non-stochastic transition")

    diagnostics = {
        "variant": variant,
        "condition": CONDITIONS[variant],
        "condition_label": CONDITION_LABELS[variant],
        "filter_kind": filter_kind,
        "filter_states": filter_states,
        "observation_partition": {
            "not-C": ["token A", "token B"],
            "C": ["token C"],
        },
        "strong_lumpability_over_states_1_2": not non_lumpable_actions,
        "non_lumpable_actions": non_lumpable_actions,
        "lumpability_max_abs_row_difference_by_action": {
            str(action): error for action, error in lumpability_errors.items()
        },
    }
    return filter_initial, filter_emission, transitions, diagnostics


def coarse_filter_spec(
    *,
    variant: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Construct the exact filter for A/B-coarsened observations."""

    initial, emission, transitions, _ = _build_coarse_filter_spec(variant)
    return initial, emission, transitions


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
    variant: int = 2,
) -> np.ndarray:
    """Replay the exact posterior under A/B-coarsened observations."""

    initial, emission, transitions = coarse_filter_spec(variant=variant)
    tokens, previous_actions = _tokens_and_previous_actions(data)
    beliefs = np.empty(
        (len(tokens), len(initial)),
        dtype=np.float64,
    )
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
    diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        **diagnostics,
        "initial": initial.tolist(),
        "emission_rows_filter_states_columns_not_C_C": emission.tolist(),
        "transition_rows_filter_states_columns_filter_states": {
            str(action): matrix.tolist() for action, matrix in transitions.items()
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
    """Fit paired coarse-observation and full-observation scalar probes."""

    if variant not in CONDITIONS:
        raise ValueError("variant must be one of 1, 2, or 3")
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
        float(np.max(np.abs(replay_beliefs(train, variant=variant) - train.beliefs))),
        float(np.max(np.abs(replay_beliefs(test, variant=variant) - test.beliefs))),
    )
    if replay_error > 1e-10:
        raise ValueError(
            f"full belief replay disagrees with environment diagnostics: {replay_error}"
        )

    train_coarse = coarse_beliefs(train, variant=variant)[train.mask, -1:]
    test_coarse = coarse_beliefs(test, variant=variant)[test.mask, -1:]
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
    initial, emission, transitions, diagnostics = _build_coarse_filter_spec(variant)
    return {
        "schema_version": 2,
        "metadata": {
            "variant": variant,
            "condition": CONDITIONS[variant],
            "condition_label": CONDITION_LABELS[variant],
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
            "seed_streams": {name: list(key) for name, key in _PROBE_STREAMS.items()},
            "stream_relationship": (
                "The coarse-observation and full-observation targets share "
                "each paired fit/test rollout, matching the cycle-7 scalar "
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
                "causal policy use or environmental prediction accuracy."
            ),
        },
        "coarse_model": _coarse_model_record(
            initial,
            emission,
            transitions,
            diagnostics,
        ),
        "full_belief_replay_max_abs_error": replay_error,
        "target_difference": {
            "mse": float(np.mean(np.square(differences))),
            "max_abs_difference": float(np.max(np.abs(differences))),
            "n_compared": int(len(differences)),
            "interpretation": (
                "The coarse-observation filter discards token A/B identity; "
                "its state-3 posterior is not the full-observation rho target."
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
    parser.add_argument("--variant", type=int, required=True)
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
        "experiment_git_dirty": bool(_git(repository, "status", "--porcelain")),
        "harness_git_commit": _git(harness, "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
