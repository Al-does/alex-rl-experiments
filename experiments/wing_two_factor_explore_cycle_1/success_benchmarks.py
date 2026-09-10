from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.stats import t as student_t

from envs.wing.model import controlled_kernels, wing_model


DEFAULT_SEED = 20260923
SIMPLEX_TOLERANCE = 64 * np.finfo(np.float64).eps
FACTOR_NORMALIZATION = (
    "The two independent factors have separately selectable actions. Reward is "
    "the mean over selected arrival-state indicators, not joint success. Thus "
    "reward_both and reward_factor_1 have the same normalized optimal value."
)


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _reward_state(value: int) -> int:
    value = _integer(value, "reward_state", 0)
    if value > 2:
        raise ValueError("reward_state must be 0, 1, or 2")
    return value


def _provenance(kernels: np.ndarray) -> dict[str, str]:
    return {
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "kernel_sha256": hashlib.sha256(np.asarray(kernels, dtype="<f8").tobytes()).hexdigest(),
        "arithmetic": "float64; not interval-arithmetic certified",
    }


def _episode_estimate(episode_means: np.ndarray) -> dict[str, Any]:
    count = len(episode_means)
    mean = float(episode_means.mean())
    standard_error = float(episode_means.std(ddof=1) / np.sqrt(count))
    half_width = float(student_t.ppf(0.975, count - 1) * standard_error)
    return {
        "mean": mean,
        "percentage": 100 * mean,
        "standard_error": standard_error,
        "ci95": [mean - half_width, mean + half_width],
        "ci_method": "approximate Student-t interval over independent full-episode means",
        "confidence_scope": "Monte Carlo uncertainty only; not a rigorous confidence bound on the optimum",
    }


def _posterior(mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = mass.sum(axis=-1)
    posterior = np.full_like(mass, 1 / 3)
    np.divide(mass, probability[..., None], out=posterior, where=probability[..., None] > 0)
    return probability, posterior


def token_guess_bayes_accuracy(
    alpha: float = 0.94,
    x: float = 0.4,
    horizon: int = 1024,
    n_episodes: int = 4096,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    horizon = _integer(horizon, "horizon", 1)
    n_episodes = _integer(n_episodes, "n_episodes", 2)
    seed = _integer(seed, "seed", 0)
    model = wing_model(alpha=alpha, x=x)
    edges = model.edge_transition_matrices
    rng = np.random.default_rng(seed)
    source = np.full((n_episodes, 2, 3), 1 / 3)
    totals = np.zeros(n_episodes)
    first_decision_accuracy = None
    for step in range(horizon):
        probabilities = source @ model.emission_matrix
        accuracy = probabilities.max(axis=-1).prod(axis=-1)
        totals += accuracy
        if step == 0:
            first_decision_accuracy = float(accuracy[0])
        tokens = (rng.random((n_episodes, 2)) >= probabilities[..., 0]).astype(np.int64)
        _, source = _posterior(np.einsum("efi,efij->efj", source, edges[tokens]))
    estimate = _episode_estimate(totals / horizon)
    dominant_tokens = np.flatnonzero(np.all(
        model.emission_matrix >= model.emission_matrix.max(axis=-1, keepdims=True), axis=0,
    ))
    analytic = None
    if len(dominant_tokens):
        token = int(dominant_tokens[0])
        unconditional = model.initial_distribution
        stationary_residual = float(np.abs(unconditional @ model.transition_matrix - unconditional).max())
        if stationary_residual > SIMPLEX_TOLERANCE:
            raise ArithmeticError("Wing uniform source prior must be stationary")
        exact_accuracy = float(unconditional @ model.emission_matrix[:, token]) ** 2
        exact_total = horizon * exact_accuracy
        analytic = {
            "status": "exact Bayes-optimal value from a statewise-dominant token, evaluated in float64",
            "joint_guess": [token, token],
            "fraction": exact_accuracy,
            "percentage": 100 * exact_accuracy,
            "expected_episode_return": exact_total,
            "emission_probabilities_of_dominant_token": model.emission_matrix[:, token].tolist(),
            "stationary_source_residual": stationary_residual,
            "justification": "one token maximizes probability in every hidden state, hence in every history-conditioned belief; the uniform source prior is stationary, so square its per-factor correctness probability at every decision",
        }
    return {
        "benchmark": "passive_delay_one_exact_bayes_rule_monte_carlo",
        "metric": "joint_token_guess_accuracy",
        "status": "Monte Carlo estimate of the exact Bayes-optimal prediction rule, not a certified maximum",
        **estimate,
        "expected_episode_return": horizon * estimate["mean"],
        "alpha": float(alpha),
        "x": float(x),
        "factor_count": 2,
        "delay": 1,
        "horizon": horizon,
        "n_episodes": n_episodes,
        "independent_chains": n_episodes,
        "episodes_per_chain": 1,
        "seed": seed,
        "reset_count": n_episodes,
        "warmup_actions": 0,
        "first_decision_accuracy": first_decision_accuracy,
        "analytic_cross_check": analytic,
        "reset": "uniform source prior; first decision has a blank observation and guesses the pending reset-edge token",
        "filter": "source <- normalize(source @ base_edge[revealed_token]); actions never alter the passive process",
        "policy": "argmax joint pending-token probability; independent-factor maximum probabilities are multiplied",
        "estimator": "mean conditional expected correctness before each pending token is revealed",
        "random_stream": "NumPy default_rng; step-major draws of shape (n_episodes, 2)",
        "episode_scope": "full fixed-length episodes; randomized first training-episode lengths excluded",
        "provenance": _provenance(edges),
    }


def simplex_grid(subdivisions: int) -> np.ndarray:
    subdivisions = _integer(subdivisions, "subdivisions", 1)
    first = np.repeat(np.arange(subdivisions + 1), np.arange(subdivisions + 1, 0, -1))
    second = np.concatenate([np.arange(subdivisions + 1 - index) for index in range(subdivisions + 1)])
    return np.column_stack((first, second, subdivisions - first - second)) / subdivisions


def simplex_stencil(beliefs: np.ndarray, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    subdivisions = _integer(subdivisions, "subdivisions", 1)
    beliefs = np.asarray(beliefs, dtype=np.float64)
    if (
        beliefs.ndim < 1 or beliefs.shape[-1] != 3 or not np.isfinite(beliefs).all()
        or np.any(beliefs < -SIMPLEX_TOLERANCE)
        or np.any(np.abs(beliefs.sum(axis=-1) - 1) > SIMPLEX_TOLERANCE)
    ):
        raise ValueError("beliefs must lie on the three-state probability simplex")
    normalized = np.maximum(beliefs, 0)
    normalized /= normalized.sum(axis=-1, keepdims=True)
    scaled = normalized[..., :2] * subdivisions
    first = np.minimum(np.floor(scaled[..., 0]).astype(np.int64), subdivisions - 1)
    second = np.minimum(np.floor(scaled[..., 1]).astype(np.int64), subdivisions - 1 - first)
    residual_first = scaled[..., 0] - first
    residual_second = scaled[..., 1] - second
    upper = (residual_first + residual_second > 1) & (first + second < subdivisions - 1)
    vertex_first = np.stack((first + upper, first + ~upper, first + upper), axis=-1)
    vertex_second = np.stack((second + upper, second + upper, second + ~upper), axis=-1)
    lower_weights = np.stack((1 - residual_first - residual_second, residual_first, residual_second), axis=-1)
    upper_weights = np.stack((residual_first + residual_second - 1, 1 - residual_first, 1 - residual_second), axis=-1)
    weights = np.where(upper[..., None], upper_weights, lower_weights)
    if np.any(weights < -SIMPLEX_TOLERANCE * subdivisions):
        raise ArithmeticError("negative barycentric interpolation weight")
    weights = np.maximum(weights, 0)
    weights /= weights.sum(axis=-1, keepdims=True)
    indices = vertex_first * (subdivisions + 1) - vertex_first * (vertex_first - 1) // 2 + vertex_second
    return indices, weights


def _interpolate(values: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return (values[indices] * weights).sum(axis=-1)
    return (values[indices] * weights[..., None]).sum(axis=-2)


class _WingGrid:
    def __init__(self, alpha: float, x: float, strength: float, reward_state: int, subdivisions: int):
        self.subdivisions = _integer(subdivisions, "subdivisions", 1)
        self.reward_state = _reward_state(reward_state)
        self.kernels = controlled_kernels(alpha=alpha, x=x, strength=strength)
        self.grid = simplex_grid(self.subdivisions)
        self.reward_scores = self.kernels.sum(axis=1)[:, :, self.reward_state].T
        self.rewards = self.grid @ self.reward_scores
        mass = np.einsum("bi,axij->baxj", self.grid, self.kernels)
        probabilities, posteriors = _posterior(mass)
        indices, weights = simplex_stencil(posteriors, self.subdivisions)
        reconstruction = (self.grid[indices] * weights[..., None]).sum(axis=-2)
        self.reconstruction_error = float(np.abs(reconstruction - posteriors).max())
        self.minimum_weight = float(weights.min())
        self.weight_sum_error = float(np.abs(weights.sum(axis=-1) - 1).max())
        count = len(self.grid)
        self.transition = csr_matrix((
            (probabilities[..., None] * weights).reshape(-1),
            indices.reshape(-1),
            np.arange(0, count * 3 * 6 + 1, 6),
        ), shape=(count * 3, count))
        self.transition.sum_duplicates()
        self.transition.eliminate_zeros()
        self.transition_row_sum_error = float(np.abs(np.asarray(self.transition.sum(axis=1)).ravel() - 1).max())
        reset_mass = np.einsum("i,xij->xj", np.full(3, 1 / 3), self.kernels[0])
        self.reset_probabilities, self.reset_beliefs = _posterior(reset_mass)
        self.reset_rewards = self.reset_beliefs @ self.reward_scores
        reset_next_mass = np.einsum("bi,axij->baxj", self.reset_beliefs, self.kernels)
        self.reset_next_probabilities, reset_next_beliefs = _posterior(reset_next_mass)
        self.reset_indices, self.reset_weights = simplex_stencil(reset_next_beliefs, self.subdivisions)

    def q_values(self, values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        return self.rewards + gamma * (self.transition @ values).reshape(-1, 3)

    def reset_value(self, continuation: np.ndarray) -> float:
        future = _interpolate(continuation, self.reset_indices, self.reset_weights)
        q_values = self.reset_rewards + (self.reset_next_probabilities * future).sum(axis=-1)
        return float(self.reset_probabilities @ q_values.max(axis=-1))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "subdivisions": self.subdivisions,
            "grid_points": len(self.grid),
            "sparse_transition_nonzeros": self.transition.nnz,
            "minimum_barycentric_weight": self.minimum_weight,
            "max_barycentric_sum_error": self.weight_sum_error,
            "max_posterior_reconstruction_error": self.reconstruction_error,
            "max_transition_row_sum_error": self.transition_row_sum_error,
        }


def controlled_bayes_bound(
    alpha: float = 0.94,
    x: float = 0.4,
    strength: float = 1.0,
    reward_state: int = 0,
    horizon: int = 1024,
    subdivisions: int = 200,
) -> dict[str, Any]:
    horizon = _integer(horizon, "horizon", 1)
    planner = _WingGrid(alpha, x, strength, reward_state, subdivisions)
    values = np.zeros(len(planner.grid))
    for _ in range(horizon - 1):
        values = planner.q_values(values).max(axis=-1)
    episode_return = planner.reset_value(values)
    return {
        "benchmark": "controlled_finite_horizon_belief_grid_upper_bound",
        "metric": "mean_rewarded_factor_arrival_state_indicator",
        "status": "numerical upper bound, not the exact Bayes optimum",
        "upper_bound_fraction": episode_return / horizon,
        "upper_bound_percentage": 100 * episode_return / horizon,
        "episode_return_upper_bound": episode_return,
        "alpha": float(alpha),
        "x": float(x),
        "strength": float(strength),
        "reward_state": planner.reward_state,
        "horizon": horizon,
        "discount": 1.0,
        "terminal_value": 0.0,
        "warmup_actions": 0,
        "reset": "uniform source prior -> neutral base edge -> visible reset token -> arrival posterior before first action; reset earns no reward",
        "reset_token_probabilities": planner.reset_probabilities.tolist(),
        "reset_beliefs": planner.reset_beliefs.tolist(),
        "reset_evaluation": "exact Bellman backup at reset posteriors, interpolating only the remaining-horizon continuation",
        "filter": "normalize(belief @ K[action, token]); no hidden-state or reward observations",
        "reward": "belief @ sum_token K[action, token] @ one_hot(reward_state)",
        "factor_normalization": FACTOR_NORMALIZATION,
        "bound_justification": "nonnegative barycentric posterior decomposition and convex finite-horizon POMDP values imply an information-relaxation upper bound by Bellman induction",
        "certification": "float64 numerical upper bound; no interval arithmetic, no certified discretization gap, no extrapolation",
        "episode_scope": "full fixed-length episodes; randomized first training-episode lengths excluded",
        "grid": planner.diagnostics(),
        "provenance": _provenance(planner.kernels),
    }


def controlled_bayes_refinements(
    alpha: float = 0.94,
    x: float = 0.4,
    strength: float = 1.0,
    reward_state: int = 0,
    horizon: int = 1024,
    subdivisions: tuple[int, ...] = (100, 200, 400),
) -> dict[str, Any]:
    levels = tuple(_integer(level, "subdivisions", 1) for level in subdivisions)
    if not levels or any(second <= first for first, second in zip(levels, levels[1:])):
        raise ValueError("subdivisions must be a nonempty strictly increasing sequence")
    bounds = [controlled_bayes_bound(alpha, x, strength, reward_state, horizon, level) for level in levels]
    fractions = [row["upper_bound_fraction"] for row in bounds]
    changes = [second - first for first, second in zip(fractions, fractions[1:])]
    best = min(bounds, key=lambda row: row["upper_bound_fraction"])
    return {
        "benchmark": "controlled_finite_horizon_grid_refinement",
        "status": best["status"],
        "metric": best["metric"],
        "alpha": best["alpha"],
        "x": best["x"],
        "strength": best["strength"],
        "reward_state": best["reward_state"],
        "horizon": best["horizon"],
        "factor_normalization": FACTOR_NORMALIZATION,
        "upper_bound_fraction": best["upper_bound_fraction"],
        "upper_bound_percentage": best["upper_bound_percentage"],
        "episode_return_upper_bound": best["episode_return_upper_bound"],
        "selected_subdivisions": best["grid"]["subdivisions"],
        "refinements": bounds,
        "convergence": {
            "successive_fraction_changes": changes,
            "last_absolute_fraction_change": abs(changes[-1]) if changes else None,
            "observed_nonincreasing": all(change <= 1e-12 for change in changes),
            "nested_triangulations": all(second % first == 0 for first, second in zip(levels, levels[1:])),
            "interpretation": "resolution stability is not a certified error bar or proof of equality to the continuous-belief optimum; minimum of computed upper bounds selected without extrapolation",
        },
    }


def controlled_belief_policy_reward(
    alpha: float = 0.94,
    x: float = 0.4,
    strength: float = 1.0,
    reward_state: int = 0,
    horizon: int = 1024,
    subdivisions: int = 200,
    n_episodes: int = 4096,
    seed: int = DEFAULT_SEED,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 10000,
) -> dict[str, Any]:
    horizon = _integer(horizon, "horizon", 1)
    n_episodes = _integer(n_episodes, "n_episodes", 2)
    seed = _integer(seed, "seed", 0)
    max_iterations = _integer(max_iterations, "max_iterations", 1)
    if isinstance(gamma, (bool, np.bool_)) or not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must lie in [0, 1)")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    planner = _WingGrid(alpha, x, strength, reward_state, subdivisions)
    values = np.zeros(len(planner.grid))
    for iteration in range(1, max_iterations + 1):
        updated = planner.q_values(values, gamma).max(axis=-1)
        residual_span = float(np.ptp(updated - values))
        values = updated - updated[0]
        if residual_span <= tolerance:
            break
    else:
        raise RuntimeError("centered discounted grid value iteration did not converge")
    policy_q = planner.q_values(values, gamma)
    rng = np.random.default_rng(seed)
    reset_tokens = (rng.random(n_episodes) >= planner.reset_probabilities[0]).astype(np.int64)
    beliefs = planner.reset_beliefs[reset_tokens].copy()
    totals = np.zeros(n_episodes)
    for _ in range(horizon):
        indices, weights = simplex_stencil(beliefs, subdivisions)
        actions = _interpolate(policy_q, indices, weights).argmax(axis=-1)
        totals += np.einsum("bi,bi->b", beliefs, planner.reward_scores.T[actions])
        mass = np.einsum("bi,bxij->bxj", beliefs, planner.kernels[actions])
        probabilities, posteriors = _posterior(mass)
        tokens = (rng.random(n_episodes) >= probabilities[:, 0]).astype(np.int64)
        beliefs = posteriors[np.arange(n_episodes), tokens]
    estimate = _episode_estimate(totals / horizon)
    return {
        "benchmark": "controlled_attainable_full_belief_policy_monte_carlo",
        "metric": "mean_rewarded_factor_arrival_state_indicator",
        "status": "Monte Carlo estimate of an attainable policy value, not the Bayes optimum or a rigorous lower confidence bound",
        **estimate,
        "expected_episode_return": horizon * estimate["mean"],
        "alpha": float(alpha),
        "x": float(x),
        "strength": float(strength),
        "reward_state": planner.reward_state,
        "horizon": horizon,
        "n_episodes": n_episodes,
        "independent_chains": n_episodes,
        "episodes_per_chain": 1,
        "reset_count": n_episodes,
        "warmup_actions": 0,
        "seed": seed,
        "policy": "stationary argmax barycentrically interpolated discounted grid Q; no artificial grid-state revelation",
        "filter": "exact action/token belief; no hidden-state or reward observations",
        "estimator": "conditional expected arrival-state reward before each controlled edge",
        "reset": "uniform source prior -> neutral base edge -> visible token -> arrival posterior before first action",
        "factor_normalization": FACTOR_NORMALIZATION,
        "simulated_factors": 1,
        "evaluation_discount": 1.0,
        "planning": {
            "gamma": float(gamma),
            "method": "centered discounted grid value iteration",
            "iterations": iteration,
            "bellman_increment_span": residual_span,
            "span_tolerance": float(tolerance),
            "max_iterations": max_iterations,
        },
        "random_stream": "NumPy default_rng; reset then step-major draws of shape (n_episodes,)",
        "grid": planner.diagnostics(),
        "provenance": _provenance(planner.kernels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("passive", "controlled", "all"), default="all")
    parser.add_argument("--alpha", type=float, default=0.94)
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--reward-state", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--with-policy", action="store_true")
    parser.add_argument("--policy-subdivisions", type=int, default=200)
    args = parser.parse_args()
    common = {"alpha": args.alpha, "x": args.x, "horizon": args.horizon}
    simulation = {"n_episodes": args.episodes, "seed": args.seed}
    controlled = common | {"strength": args.strength, "reward_state": args.reward_state}
    result = {}
    if args.study in ("passive", "all"):
        result["passive"] = token_guess_bayes_accuracy(**common, **simulation)
    if args.study in ("controlled", "all"):
        result["controlled"] = controlled_bayes_refinements(**controlled, subdivisions=tuple(args.subdivisions))
        if args.with_policy:
            result["controlled_policy"] = controlled_belief_policy_reward(
                **controlled, **simulation, subdivisions=args.policy_subdivisions,
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
