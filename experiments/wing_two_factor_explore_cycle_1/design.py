from __future__ import annotations

import argparse
import itertools
import json
from typing import Any

import numpy as np

from envs.wing.model import controlled_kernels
from experiments.wing_two_factor_explore_cycle_1.process import (
    CONTEXT_LENGTH,
    CONTROL_STRENGTH,
    WING_ALPHA,
    WING_X,
)


GAMMA = 0.99
AUDIT_SEED = 20260901
AUDIT_CHAINS = 2048
AUDIT_STEPS = 4096
AUDIT_BURN_IN = 512
POLICY_MODES = ("full_qmdp", "full_myopic", "suffix_1_qmdp", "suffix_32_qmdp")


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    transition = np.asarray(transition, dtype=np.float64)
    system = transition.T - np.eye(len(transition))
    system[-1] = 1.0
    target = np.zeros(len(transition))
    target[-1] = 1.0
    return np.linalg.solve(system, target)


def reactive_transition(kernels: np.ndarray, policy: tuple[int, ...]) -> np.ndarray:
    return np.stack([
        kernels[policy[token], :, state, :].T.reshape(6)
        for state in range(3)
        for token in range(2)
    ])


def exact_baselines(kernels: np.ndarray) -> dict[str, Any]:
    transitions = kernels.sum(axis=1)
    constants = np.stack([stationary_distribution(t) for t in transitions])
    reactive = []
    for policy in itertools.product(range(3), repeat=2):
        joint = stationary_distribution(reactive_transition(kernels, policy)).reshape(3, 2)
        reactive.append({"policy": list(policy), "occupancy": joint.sum(axis=1).tolist()})
    oracle = []
    for policy in itertools.product(range(3), repeat=3):
        transition = transitions[np.array(policy), np.arange(3)]
        oracle.append({
            "policy": list(policy),
            "occupancy": stationary_distribution(transition).tolist(),
        })
    rewards = {}
    for reward_state in range(3):
        best_reactive = max(reactive, key=lambda row: row["occupancy"][reward_state])
        best_oracle = max(oracle, key=lambda row: row["occupancy"][reward_state])
        rewards[str(reward_state)] = {
            "best_constant": float(constants[:, reward_state].max()),
            "best_reactive": best_reactive["occupancy"][reward_state],
            "best_reactive_policy": best_reactive["policy"],
            "oracle_upper_bound": best_oracle["occupancy"][reward_state],
            "oracle_policy": best_oracle["policy"],
        }
    return {
        "constant_occupancies": constants.tolist(),
        "reactive_policies": reactive,
        "oracle_policies": oracle,
        "reward_states": rewards,
    }


def fully_observed_q_values(
    kernels: np.ndarray, reward_state: int, gamma: float = GAMMA,
) -> np.ndarray:
    if reward_state not in (0, 1, 2) or isinstance(reward_state, bool):
        raise ValueError("reward_state must be 0, 1, or 2")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    transitions = kernels.sum(axis=1)
    reward = np.eye(3)[reward_state]
    value = np.zeros(3)
    for _ in range(100000):
        q_values = np.einsum("asj,j->sa", transitions, reward + gamma * value)
        updated = q_values.max(axis=1)
        if np.max(np.abs(updated - value)) < 1e-12:
            return q_values
        value = updated
    raise RuntimeError("fully observed value iteration did not converge")


def filter_update(beliefs: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    posterior = np.einsum("...i,...ij->...j", beliefs, matrices)
    mass = posterior.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("zero-probability action/token history")
    return posterior / mass


def suffix_belief(kernels: np.ndarray, actions: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    actions, tokens = np.asarray(actions), np.asarray(tokens)
    if actions.shape != tokens.shape:
        raise ValueError("actions and tokens must have the same shape")
    belief = np.full(actions.shape[1:] + (3,), 1.0 / 3.0)
    for action, token in zip(actions, tokens, strict=True):
        belief = filter_update(belief, kernels[action, token])
    return belief


def _normalize_product(matrix: np.ndarray) -> np.ndarray:
    return matrix / matrix.sum(axis=(-2, -1), keepdims=True)


class _WindowFilter:
    def __init__(self, length: int, chains: int):
        self.length = length
        self.position = 0
        self.previous = None
        self.matrices = np.empty((length, chains, 3, 3))
        self.prefix = np.broadcast_to(np.eye(3), (chains, 3, 3)).copy()

    def update(self, matrix: np.ndarray) -> np.ndarray:
        index = self.position
        self.matrices[index] = matrix
        self.prefix = _normalize_product(self.prefix @ matrix)
        product = self.prefix
        if self.previous is not None and index + 1 < self.length:
            product = _normalize_product(self.previous[index + 1] @ self.prefix)
        belief = product.sum(axis=-2)
        belief /= belief.sum(axis=-1, keepdims=True)
        self.position += 1
        if self.position == self.length:
            self.previous = self.matrices.copy()
            for offset in range(self.length - 2, -1, -1):
                self.previous[offset] = _normalize_product(
                    self.previous[offset] @ self.previous[offset + 1]
                )
            self.position = 0
            self.prefix[:] = np.eye(3)
        return belief


def identifiability_summary(kernels: np.ndarray) -> dict[str, Any]:
    delta = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    emissions = kernels.sum(axis=-1)
    next_token_null = np.einsum("s,axs->ax", delta, emissions)
    two_token = np.einsum("s,axsj,byj->axby", delta, kernels, emissions)
    return {
        "hidden_null_direction": delta.tolist(),
        "action_emission_maps": emissions.transpose(0, 2, 1).tolist(),
        "action_emission_ranks": [int(np.linalg.matrix_rank(e.T)) for e in emissions],
        "max_one_step_null_residual": float(np.abs(next_token_null).max()),
        "max_two_step_token_difference": float(np.abs(two_token).max()),
        "arrival_reward_contrast_by_action": np.einsum(
            "s,asj->aj", delta, kernels.sum(axis=1)
        ).tolist(),
        "interpretation": (
            "States 0 and 2 have identical next-token distributions for every action; "
            "one-step NTP does not identify their belief contrast. This null is not "
            "an invariant unobservable subspace of the edge dynamics: two-step "
            "token distributions can distinguish it. Arrival rewards for states "
            "0 and 2 are sensitive to this contrast. State 1 remains a valid comparison."
        ),
    }


def design_summary(
    *, alpha: float = WING_ALPHA, x: float = WING_X,
    strength: float = CONTROL_STRENGTH,
) -> dict[str, Any]:
    kernels = controlled_kernels(alpha=alpha, x=x, strength=strength)
    return {
        "alpha": alpha, "x": x, "b": (1.0 - alpha) / 2.0, "strength": strength,
        "kernel_convention": "K[action, token, source, arrival] = K[token] @ C[action]",
        "reward_convention": "indicator of arrival state, per rewarded factor",
        "baseline_scope": (
            "Infinite-horizon stationary average reward, one factor. Best reactive "
            "means deterministic current-token-only policies (9), not policies using "
            "previous actions or randomized memoryless policies. Oracle enumerates "
            "27 deterministic hidden-state policies; all default chains are unichain. "
            "The study averages rewarded-factor indicators, so joint expected "
            "reward averages the selected per-factor values rather than summing them."
        ),
        "kernels": kernels.tolist(),
        "baselines": exact_baselines(kernels),
        "identifiability": identifiability_summary(kernels),
        "audit_defaults": {
            "seed": AUDIT_SEED, "chains": AUDIT_CHAINS, "steps": AUDIT_STEPS,
            "burn_in": AUDIT_BURN_IN, "gamma": GAMMA, "suffix_length": CONTEXT_LENGTH,
            "modes": list(POLICY_MODES),
            "default_policy_equivalence": (
                "Default fully observed Q is a positive affine transform of myopic "
                "arrival scores, so full_qmdp and full_myopic are the same policy "
                "apart from numerical ties; their simulations use independent seeds."
            ),
            "standard_error": "sample standard deviation of independent chain means / sqrt(chains)",
            "filter": "action-aware Bayes on action/token history, without reward observations",
            "suffix": "restart uniform stationary before oldest of last W generating-action/token frames",
            "mse": "on-policy suffix posterior minus full posterior; mean squared error per coordinate and null projection",
            "initialization": "uniform hidden state and belief; no initial token, burn-in discarded; no episode resets",
            "limitation": "QMDP uses fully observed discounted arrival Q, not optimal POMDP Q; evaluation is average reward",
        },
    }


def _estimate(chain_means: np.ndarray) -> dict[str, Any]:
    mean = chain_means.mean(axis=0)
    stderr = chain_means.std(axis=0, ddof=1) / np.sqrt(len(chain_means))
    return {"mean": mean.tolist(), "standard_error": stderr.tolist()}


def simulate_policy(
    kernels: np.ndarray, reward_state: int, mode: str, *,
    seed: int, chains: int, steps: int, burn_in: int,
) -> dict[str, Any]:
    if mode not in POLICY_MODES:
        raise ValueError(f"mode must be one of {POLICY_MODES}")
    if chains < 2 or not 0 <= burn_in < steps:
        raise ValueError("need at least two chains and 0 <= burn_in < steps")
    rng = np.random.default_rng(seed)
    states = rng.integers(3, size=chains)
    full = np.full((chains, 3), 1.0 / 3.0)
    short, window = full.copy(), full.copy()
    window_filter = _WindowFilter(CONTEXT_LENGTH, chains)
    score = (
        kernels.sum(axis=1)[:, :, reward_state].T
        if mode == "full_myopic"
        else fully_observed_q_values(kernels, reward_state)
    )
    occupancy = np.zeros(chains)
    squared_error = {length: np.zeros((chains, 4)) for length in (1, CONTEXT_LENGTH)}
    disagreement = np.zeros(chains)
    delta = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    for step in range(steps):
        belief = short if mode == "suffix_1_qmdp" else window if mode == "suffix_32_qmdp" else full
        actions = (belief @ score).argmax(axis=-1)
        probabilities = kernels[actions, :, states, :].reshape(chains, 6)
        outcomes = (rng.random((chains, 1)) > probabilities.cumsum(axis=1)).sum(axis=1)
        tokens, states = outcomes // 3, outcomes % 3
        matrices = kernels[actions, tokens]
        full = filter_update(full, matrices)
        short = matrices.sum(axis=1)
        short /= short.sum(axis=-1, keepdims=True)
        window = window_filter.update(matrices)
        if step >= burn_in:
            occupancy += states == reward_state
            disagreement += (window @ score).argmax(axis=-1) != (full @ score).argmax(axis=-1)
            for length, approximation in ((1, short), (CONTEXT_LENGTH, window)):
                difference = approximation - full
                squared_error[length][:, :3] += difference ** 2
                squared_error[length][:, 3] += (difference @ delta) ** 2
    samples = steps - burn_in
    return {
        "seed": seed,
        "occupancy": _estimate(occupancy / samples),
        "suffix_mse": {
            str(length): _estimate(error / samples)
            for length, error in squared_error.items()
        },
        "mse_coordinate_order": ["state_0", "state_1", "state_2", "hidden_null_projection"],
        "window_full_action_disagreement": _estimate(disagreement / samples),
    }


def demand_audit(
    *, seed: int = AUDIT_SEED, chains: int = AUDIT_CHAINS,
    steps: int = AUDIT_STEPS, burn_in: int = AUDIT_BURN_IN,
    alpha: float = WING_ALPHA, x: float = WING_X, strength: float = CONTROL_STRENGTH,
) -> dict[str, Any]:
    summary = design_summary(alpha=alpha, x=x, strength=strength)
    kernels = np.asarray(summary["kernels"])
    rewards = {}
    for reward_state in range(3):
        baseline = summary["baselines"]["reward_states"][str(reward_state)]
        policies = {}
        for index, mode in enumerate(POLICY_MODES):
            result = simulate_policy(
                kernels, reward_state, mode, seed=seed + 100 * reward_state + index,
                chains=chains, steps=steps, burn_in=burn_in,
            )
            mean = result["occupancy"]["mean"]
            stderr = result["occupancy"]["standard_error"]
            result["gap_over_best_constant"] = mean - baseline["best_constant"]
            result["gap_over_best_reactive"] = mean - baseline["best_reactive"]
            result["gap_standard_error"] = stderr
            result["reactive_gap_95_percent_interval"] = [
                result["gap_over_best_reactive"] - 1.96 * stderr,
                result["gap_over_best_reactive"] + 1.96 * stderr,
            ]
            policies[mode] = result
        full = policies["full_qmdp"]["occupancy"]
        for mode in ("suffix_1_qmdp", "suffix_32_qmdp"):
            other = policies[mode]["occupancy"]
            policies[mode]["full_qmdp_minus_this_policy"] = {
                "mean": full["mean"] - other["mean"],
                "standard_error": float(np.hypot(full["standard_error"], other["standard_error"])),
            }
        rewards[str(reward_state)] = policies
    return {
        "design": summary,
        "computation": {"seed": seed, "chains": chains, "steps": steps, "burn_in": burn_in},
        "reward_states": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--chains", type=int, default=AUDIT_CHAINS)
    parser.add_argument("--steps", type=int, default=AUDIT_STEPS)
    parser.add_argument("--burn-in", type=int, default=AUDIT_BURN_IN)
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
    args = parser.parse_args()
    summary = design_summary() if args.analytic_only else demand_audit(
        seed=args.seed, chains=args.chains, steps=args.steps, burn_in=args.burn_in,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
