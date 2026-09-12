from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.process import (
    COMPONENT_PARAMETERS,
    EFFECT_SIZE,
    EPISODE_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    nonergodic_mess3_model,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.task import (
    N_ACTIONS,
    NOOP_ACTION,
    REWARD_STATE,
    ActionSymmetryTask,
)


DEFAULT_AUDIT_EPISODES = 16_384
DEFAULT_AUDIT_SEED = 20_260_912
NORMAL_95 = 1.959963984540054


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def reward_state_indicator() -> np.ndarray:
    reward = np.zeros(STATE_COUNT, dtype=np.float64)
    reward[REWARD_STATE::STATES_PER_COMPONENT] = 1.0
    return reward


def controlled_kernels(
    variant: int,
    effect_size: float = EFFECT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    model = nonergodic_mess3_model()
    task = ActionSymmetryTask(
        model=model,
        variant=variant,
        effect_size=effect_size,
    )
    transitions = np.stack(
        [
            task.transition_matrix_for_action(action)
            for action in range(N_ACTIONS)
        ]
    )
    edges = np.stack(
        [
            task.edge_transition_matrices_for_action(action)
            for action in range(N_ACTIONS)
        ]
    )
    return transitions, edges


def expected_next_reward_scores(
    beliefs: np.ndarray,
    transitions: np.ndarray,
) -> np.ndarray:
    values = np.asarray(beliefs, dtype=np.float64)
    matrices = np.asarray(transitions, dtype=np.float64)
    if values.shape[-1] != STATE_COUNT:
        raise ValueError(f"beliefs must end in dimension {STATE_COUNT}")
    if matrices.shape != (N_ACTIONS, STATE_COUNT, STATE_COUNT):
        raise ValueError(
            "transitions must have shape "
            f"({N_ACTIONS}, {STATE_COUNT}, {STATE_COUNT})"
        )
    return np.einsum(
        "...i,aij,j->...a",
        values,
        matrices,
        reward_state_indicator(),
    )


def constant_action_expected_return(
    variant: int,
    action: int,
    *,
    effect_size: float = EFFECT_SIZE,
    horizon: int = EPISODE_LENGTH,
) -> float:
    horizon = _positive_integer(horizon, "horizon")
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise ValueError("action must be an integer")
    if not 0 <= int(action) < N_ACTIONS:
        raise ValueError("action is outside the action space")
    model = nonergodic_mess3_model()
    transitions, _ = controlled_kernels(variant, effect_size)
    belief = model.initial_distribution @ transitions[NOOP_ACTION]
    reward = reward_state_indicator()
    total = 0.0
    for _ in range(horizon):
        total += float(belief @ reward)
        belief = belief @ transitions[int(action)]
    return total


def _posterior(
    source_beliefs: np.ndarray,
    edge_matrices: np.ndarray,
    tokens: np.ndarray,
) -> np.ndarray:
    rows = np.arange(len(source_beliefs))
    mass = np.einsum(
        "bi,bij->bj",
        source_beliefs,
        edge_matrices[rows, tokens],
    )
    return mass / mass.sum(axis=1, keepdims=True)


def bayes_observer_reward_audit(
    variant: int,
    *,
    effect_size: float = EFFECT_SIZE,
    horizon: int = EPISODE_LENGTH,
    n_episodes: int = DEFAULT_AUDIT_EPISODES,
    seed: int = DEFAULT_AUDIT_SEED,
) -> dict[str, object]:
    horizon = _positive_integer(horizon, "horizon", 2)
    n_episodes = _positive_integer(n_episodes, "n_episodes", 2)
    seed = _positive_integer(seed, "seed", 0)
    model = nonergodic_mess3_model()
    transitions, edges = controlled_kernels(variant, effect_size)
    reward = reward_state_indicator()
    constant_returns = np.asarray(
        [
            constant_action_expected_return(
                variant,
                action,
                effect_size=effect_size,
                horizon=horizon,
            )
            for action in range(N_ACTIONS)
        ]
    )
    best_constant_action = int(np.argmax(constant_returns))
    best_constant_return = float(constant_returns[best_constant_action])

    rng = np.random.default_rng(seed)
    source_beliefs = np.broadcast_to(
        model.initial_distribution,
        (n_episodes, STATE_COUNT),
    ).copy()
    pending_actions = np.full(
        n_episodes,
        NOOP_ACTION,
        dtype=np.int64,
    )
    expected_returns = np.zeros(n_episodes, dtype=np.float64)
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    switches = 0
    previous_actions: np.ndarray | None = None
    action_gaps: list[np.ndarray] = []

    for step in range(horizon):
        beliefs = np.einsum(
            "bi,bij->bj",
            source_beliefs,
            transitions[pending_actions],
        )
        expected_returns += beliefs @ reward
        if step == horizon - 1:
            break

        scores = expected_next_reward_scores(beliefs, transitions)
        actions = scores.argmax(axis=1)
        action_counts += np.bincount(actions, minlength=N_ACTIONS)
        if previous_actions is not None:
            switches += int(np.count_nonzero(actions != previous_actions))
        previous_actions = actions
        partitioned = np.partition(scores, -2, axis=1)
        action_gaps.append(partitioned[:, -1] - partitioned[:, -2])

        pending_edges = edges[pending_actions]
        token_probabilities = np.einsum(
            "bi,byij->by",
            source_beliefs,
            pending_edges,
        )
        cumulative = np.cumsum(token_probabilities[:, :-1], axis=1)
        tokens = (
            rng.random(n_episodes)[:, None] > cumulative
        ).sum(axis=1)
        source_beliefs = _posterior(
            source_beliefs,
            pending_edges,
            tokens,
        )
        pending_actions = actions

    differences = expected_returns - best_constant_return
    advantage = float(differences.mean())
    standard_error = float(
        differences.std(ddof=1) / np.sqrt(n_episodes)
    )
    confidence_interval = [
        advantage - NORMAL_95 * standard_error,
        advantage + NORMAL_95 * standard_error,
    ]
    action_fractions = action_counts / action_counts.sum()
    gaps = np.concatenate(action_gaps)
    switch_denominator = n_episodes * max(horizon - 2, 1)
    uses_multiple_actions = int(np.count_nonzero(action_fractions > 0.01)) > 1
    certificate = uses_multiple_actions and confidence_interval[0] > 0.0

    return {
        "variant": variant,
        "effect_size": float(effect_size),
        "horizon": horizon,
        "n_episodes": n_episodes,
        "seed": seed,
        "status": (
            "Monte Carlo value of a feasible exact-filter policy, not the "
            "Bayes-optimal POMDP value"
        ),
        "policy": (
            "condition the exact six-state belief on delayed tokens and "
            "executed actions, then maximize expected next-step occupancy"
        ),
        "reward_timing": (
            "reward is pre-transition occupancy, so the current action is "
            "ranked by its effect on the next decision's expected reward"
        ),
        "filter": (
            "source <- normalize(source @ pending_edge[token]); "
            "belief <- source @ transition[pending_action]"
        ),
        "constant_action_expected_returns": constant_returns.tolist(),
        "best_constant_action": best_constant_action,
        "best_constant_expected_return": best_constant_return,
        "observer_expected_return": float(expected_returns.mean()),
        "observer_standard_error": float(
            expected_returns.std(ddof=1) / np.sqrt(n_episodes)
        ),
        "advantage_over_best_constant": advantage,
        "advantage_normal_95_interval": confidence_interval,
        "action_fractions": action_fractions.tolist(),
        "action_switch_rate": switches / switch_denominator,
        "action_gap_quantiles": {
            "p10": float(np.quantile(gaps, 0.1)),
            "p50": float(np.quantile(gaps, 0.5)),
            "p90": float(np.quantile(gaps, 0.9)),
        },
        "uses_multiple_actions": uses_multiple_actions,
        "nonconstant_optimum_certificate": certificate,
        "certificate_interpretation": (
            "When true, this feasible belief policy's lower approximate "
            "confidence limit exceeds the exact best constant return, so no "
            "constant policy can be Bayes optimal for this finite horizon."
        ),
    }


def design_audit(
    *,
    effect_size: float = EFFECT_SIZE,
    horizon: int = EPISODE_LENGTH,
    n_episodes: int = DEFAULT_AUDIT_EPISODES,
    seed: int = DEFAULT_AUDIT_SEED,
) -> dict[str, object]:
    return {
        "study": "nonergodic_mess3_reward_state_action_symmetry_cycle_5",
        "objective": (
            "Test whether exact Bayesian observation history creates a reward "
            "incentive beyond the best constant-action policy."
        ),
        "provenance": {
            "numpy_version": np.__version__,
            "arithmetic": "float64",
        },
        "components": [dict(parameters) for parameters in COMPONENT_PARAMETERS],
        "variant_1_role": (
            "intentional constant-positive-action control inherited from the "
            "cycle-5 symmetry ladder"
        ),
        "rejected_cycle_5_effect_size_variant_2": (
            bayes_observer_reward_audit(
                2,
                effect_size=1.5,
                horizon=horizon,
                n_episodes=n_episodes,
                seed=seed + 102,
            )
        ),
        "variants": [
            bayes_observer_reward_audit(
                variant,
                effect_size=effect_size,
                horizon=horizon,
                n_episodes=n_episodes,
                seed=seed + variant,
            )
            for variant in (1, 2, 3)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect-size", type=float, default=EFFECT_SIZE)
    parser.add_argument("--horizon", type=int, default=EPISODE_LENGTH)
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_AUDIT_EPISODES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_AUDIT_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = design_audit(
        effect_size=args.effect_size,
        horizon=args.horizon,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)


if __name__ == "__main__":
    main()
