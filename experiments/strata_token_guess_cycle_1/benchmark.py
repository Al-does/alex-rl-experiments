from __future__ import annotations

from math import fsum
from typing import Any

import numpy as np

from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_1.process import (
    STRATA_ALPHA,
    STRATA_T0,
    STRATA_T1,
)


def _integer(value: int, name: str, minimum: int, maximum: int | None = None) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _model(alpha: float, t0: float, t1: float):
    model = strata_model(alpha=alpha, t0=t0, t1=t1)
    if not np.allclose(
        model.initial_distribution @ model.transition_matrix,
        model.initial_distribution,
        atol=1e-14,
        rtol=0.0,
    ):
        raise ValueError("the episode prior must be stationary for these bounds")
    return model


def _suffix_bounds(model, depth: int, batch_depth: int) -> dict[str, Any]:
    prior = model.initial_distribution
    emission = model.emission_matrix
    edges = model.edge_transition_matrices
    constant_guess = int(np.argmax(prior @ emission))
    block = emission[None].copy()
    inner_depth = min(depth, batch_depth)
    for _ in range(inner_depth):
        block = np.concatenate([edge @ block for edge in edges])
    totals = np.zeros(4)
    compensation = np.zeros(4)

    def accumulate(values: np.ndarray, remaining: int) -> None:
        nonlocal totals, compensation
        if remaining:
            for edge in edges:
                accumulate(edge @ values, remaining - 1)
            return
        joint = prior @ values
        masses = joint.sum(axis=1)
        other_guess = 1 - constant_guess
        switch = joint[:, other_guess] > joint[:, constant_guess]
        increment = np.array([
            joint.max(axis=1).sum(),
            (values.max(axis=2) @ prior).sum(),
            masses.sum(),
            masses[switch].sum(),
        ]) - compensation
        updated = totals + increment
        compensation = (updated - totals) - increment
        totals = updated

    accumulate(block, depth - inner_depth)
    lower, upper, mass, switch_mass = map(float, totals)
    return {
        "depth": depth,
        "contexts": 2 ** depth,
        "lower": lower,
        "upper": upper,
        "exact_finite_prefix_accuracy": lower,
        "gap": upper - lower,
        "context_probability_mass": mass,
        "constant_guess": constant_guess,
        "constant_guess_accuracy": float(np.max(prior @ emission)),
        "strict_switch_probability": switch_mass,
        "benefit_over_constant": lower - float(np.max(prior @ emission)),
    }


def exactly_stationary_suffix_bounds(
    depth: int,
    *,
    alpha: float = STRATA_ALPHA,
    t0: float = STRATA_T0,
    t1: float = STRATA_T1,
    batch_depth: int = 12,
) -> dict[str, Any]:
    depth = _integer(depth, "depth", 0, 24)
    batch_depth = _integer(batch_depth, "batch_depth", 0, 16)
    result = _suffix_bounds(_model(alpha, t0, t1), depth, batch_depth)
    return {
        **result,
        "method": "exhaustive_binary_suffix_enumeration",
        "lower_formula": "sum_w max_x (pi K_w M)_x",
        "upper_formula": "sum_w sum_s pi_s max_x (K_w M)_(s,x)",
        "upper_information": "genie reveals the source state before the suffix",
        "scope": "stationary infinite history and every finite prefix length >= depth",
        "numerics": "float64 enumeration; not interval-arithmetic certified",
        "batch_depth": batch_depth,
    }


def stratified_stationary_bounds(
    *,
    alpha: float = STRATA_ALPHA,
    t0: float = STRATA_T0,
    t1: float = STRATA_T1,
    grid_size: int = 2049,
    tail_tolerance: float = 1e-13,
    max_iterations: int = 10000,
) -> dict[str, Any]:
    from scipy.sparse import coo_matrix

    grid_size = _integer(grid_size, "grid_size", 3, 16385)
    max_iterations = _integer(max_iterations, "max_iterations", 1)
    tail_tolerance = _positive(tail_tolerance, "tail_tolerance")
    model = _model(alpha, t0, t1)
    if not 0.0 < alpha < 1.0 or t0 <= 0.0 or t1 <= 0.0:
        raise ValueError("renewal refinement requires 0 < alpha < 1 and t0,t1 > 0")
    zero, one = model.edge_transition_matrices
    emission = model.emission_matrix
    remaining_time = np.linalg.solve(np.eye(3) - one, np.ones(3))
    power = np.eye(3)
    token_coefficients = []
    return_coefficients = []
    for _ in range(10000):
        if np.max(power[:2].sum(axis=1)) <= tail_tolerance:
            break
        token_coefficients.append(power[:2] @ emission)
        return_coefficients.append((power[:2] @ zero)[:, :2])
        power = power @ one
    else:
        raise ValueError("the all-one tail did not reach tail_tolerance in 10000 steps")
    tokens = np.asarray(token_coefficients)
    returns = np.asarray(return_coefficients)
    requested_grid_size = grid_size
    reward_crossings = []
    for token in tokens:
        difference = token[:, 0] - token[:, 1]
        if difference[0] != difference[1]:
            root = difference[0] / (difference[0] - difference[1])
            if 0.0 < root < 1.0:
                reward_crossings.append(root)
    grid = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, grid_size), np.asarray(reward_crossings)
    ]))
    grid_size = len(grid)
    basis = np.column_stack([1.0 - grid, grid])
    cycle_time = basis @ remaining_time[:2]
    reward = np.zeros(grid_size)
    rows = []
    columns = []
    weights = []
    breakpoints = [grid]
    indices = np.arange(grid_size)
    for token, returned in zip(tokens, returns):
        reward += (basis @ token).max(axis=1)
        joint = basis @ returned
        probability = joint.sum(axis=1)
        posterior = joint[:, 1] / probability
        left = np.clip(np.searchsorted(grid, posterior) - 1, 0, grid_size - 2)
        right_weight = np.clip(
            (posterior - grid[left]) / (grid[left + 1] - grid[left]), 0.0, 1.0
        )
        rows.extend([indices, indices])
        columns.extend([left, left + 1])
        weights.extend([probability * (1.0 - right_weight), probability * right_weight])
        mass = returned.sum(axis=1)
        endpoints = returned[:, 1] / mass
        knots = grid[(grid > endpoints.min()) & (grid < endpoints.max())]
        denominator = returned[1, 1] - returned[0, 1] - knots * (mass[1] - mass[0])
        nonzero = denominator != 0.0
        roots = (knots[nonzero] * mass[0] - returned[0, 1]) / denominator[nonzero]
        breakpoints.append(roots[(roots > 0.0) & (roots < 1.0)])
        difference = token[:, 0] - token[:, 1]
        if difference[0] != difference[1]:
            root = difference[0] / (difference[0] - difference[1])
            if 0.0 < root < 1.0:
                breakpoints.append(np.array([root]))
    transition = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(grid_size, grid_size),
    ).tocsr()
    del rows, columns, weights
    stationary = np.full(grid_size, 1.0 / grid_size)
    stationary_iterations = 0
    for stationary_iterations in range(1, max_iterations + 1):
        updated = transition.T @ stationary
        updated /= updated.sum()
        error = np.max(np.abs(updated - stationary))
        stationary = updated
        if error < 1e-14:
            break
    estimate = float(stationary @ reward / (stationary @ cycle_time))
    bias = np.zeros(grid_size)
    bias_iterations = 0
    for bias_iterations in range(1, max_iterations + 1):
        updated = reward - estimate * cycle_time + transition @ bias
        updated -= updated[0]
        error = np.max(np.abs(updated - bias))
        bias = updated
        if error < 1e-12:
            break
    knots = np.unique(np.concatenate(breakpoints))
    lower = np.inf
    upper = -np.inf
    tail_mass_coefficients = power[:2].sum(axis=1)
    tail_time_coefficients = power[:2] @ remaining_time
    for start in range(0, len(knots), 4096):
        points = knots[start:start + 4096]
        point_basis = np.column_stack([1.0 - points, points])
        numerator = -np.interp(points, grid, bias)
        for token, returned in zip(tokens, returns):
            numerator += (point_basis @ token).max(axis=1)
            joint = point_basis @ returned
            probability = joint.sum(axis=1)
            numerator += probability * np.interp(joint[:, 1] / probability, grid, bias)
        time = point_basis @ remaining_time[:2]
        tail_mass = point_basis @ tail_mass_coefficients
        tail_time = point_basis @ tail_time_coefficients
        lower = min(lower, float(np.min(
            (numerator + 0.5 * tail_time + bias.min() * tail_mass) / time
        )))
        upper = max(upper, float(np.max(
            (numerator + tail_time + bias.max() * tail_mass) / time
        )))
    return {
        "lower": lower,
        "upper": upper,
        "gap": upper - lower,
        "method": "zero_return_renewal_piecewise_linear_poisson_residual_bounds",
        "scope": "stationary infinite-observation-history Bayes optimum",
        "grid_size": grid_size,
        "requested_grid_size": requested_grid_size,
        "all_one_terms": len(tokens),
        "tail_tolerance": tail_tolerance,
        "tail_probability_upper": float(tail_mass_coefficients.max()),
        "tail_expected_time_upper": float(tail_time_coefficients.max()),
        "residual_breakpoints": len(knots),
        "stationary_iterations": stationary_iterations,
        "bias_iterations": bias_iterations,
        "cycle_equation": "g tau(q) + h(q) = R(q) + P h(q), b(q)=(1-q,q,0)",
        "cycle_reward": "R(q)=sum_n max_x (b(q) E1^n M)_x",
        "cycle_time": "tau(q)=b(q) solve(I-E1,1)",
        "return_operator": "Ph(q)=sum_n p_n(q) h((b(q) E1^n E0)_1/p_n(q))",
        "proof": (
            "A zero puts the posterior on b(q)=(1-q,q,0). At successive zeros, "
            "the expected bias increments telescope. The ratio (R+Ph-h)/tau "
            "therefore bounds stationary reward by its infimum and supremum. "
            "For piecewise-linear h, every truncated numerator is piecewise "
            "affine; all grid knots, return-map preimages, and reward crossings "
            "are evaluated. A positive affine denominator has no interior "
            "extremum on a segment. Omitted rewards lie between half and all "
            "of the remaining expected time; omitted bias lies between min(h) "
            "and max(h) times the tail return probability. Grid solving only "
            "chooses h; its approximation and convergence are not assumptions "
            "of the residual bounds."
        ),
        "numerics": "float64 extremum evaluation with analytic tail bounds; not interval-arithmetic certified",
    }


def bayes_accuracy_bounds(
    *,
    alpha: float = STRATA_ALPHA,
    t0: float = STRATA_T0,
    t1: float = STRATA_T1,
    max_depth: int = 20,
    tolerance: float = 1e-5,
    batch_depth: int = 12,
    renewal_grid_size: int = 2049,
) -> dict[str, Any]:
    max_depth = _integer(max_depth, "max_depth", 0, 24)
    batch_depth = _integer(batch_depth, "batch_depth", 0, 16)
    tolerance = _positive(tolerance, "tolerance")
    model = _model(alpha, t0, t1)
    history = []
    for depth in range(max_depth + 1):
        history.append(_suffix_bounds(model, depth, batch_depth))
        if history[-1]["gap"] <= tolerance:
            break
    last = history[-1]
    lower, upper = last["lower"], last["upper"]
    renewal = None
    renewal_error = None
    if upper - lower > tolerance:
        try:
            renewal = stratified_stationary_bounds(
                alpha=alpha, t0=t0, t1=t1, grid_size=renewal_grid_size
            )
        except ValueError as error:
            renewal_error = str(error)
        if renewal is not None:
            lower = max(lower, renewal["lower"])
            upper = min(upper, renewal["upper"])
    constant = float(np.max(model.initial_distribution @ model.emission_matrix))
    return {
        "alpha": float(alpha),
        "t0": float(t0),
        "t1": float(t1),
        "stationary_prior": model.initial_distribution.tolist(),
        "lower": lower,
        "upper": upper,
        "gap": upper - lower,
        "tolerance": tolerance,
        "tolerance_met": bool(upper - lower <= tolerance),
        "actual_depth": last["depth"],
        "max_depth": max_depth,
        "suffix_history": history,
        "suffix_lower": last["lower"],
        "suffix_upper": last["upper"],
        "finite_prefix_lower": last["lower"],
        "finite_prefix_upper": upper,
        "finite_prefix_minimum_length": last["depth"],
        "finite_prefix_gap": upper - last["lower"],
        "renewal_refinement": renewal,
        "renewal_error": renewal_error,
        "constant_guess": int(np.argmax(model.initial_distribution @ model.emission_matrix)),
        "constant_guess_accuracy": constant,
        "source_state_genie_accuracy": float(
            model.initial_distribution @ model.emission_matrix.max(axis=1)
        ),
        "benefit_over_constant_lower": lower - constant,
        "benefit_over_constant_upper": upper - constant,
        "strict_suffix_switch_probability": last["strict_switch_probability"],
        "reference_accuracy": upper,
        "reference_label": "Bayes max (stationary upper bound; numerical bracket reported)",
        "reference_scope": "expected accuracy, not a ceiling on finite empirical measurements",
        "stationary_initialization_max_abs_error": float(np.max(np.abs(model.initial_distribution @ model.transition_matrix - model.initial_distribution))),
        "finite_prefix_transfer_proof": (
            "The stationary reset prior makes every reset-generated finite token block "
            "equal in law to a stationary block. Let V_t be expected optimal prediction "
            "accuracy conditioned on t previous tokens. Conditional Jensen for max "
            "implies V_t <= V_(t+1) <= V_infinity. Thus a valid stationary infinite-history "
            "upper bound also bounds every finite-prefix expected accuracy. The suffix "
            "quantity L_k is exactly V_k, so L_k <= V_t for t >= k. No latent-state "
            "mixing correction or stationary posterior assumption is used."
        ),
        "finite_prefix_scope": (
            "Stationary initialization; every prefix length >= actual_depth, "
            "including independently position-weighted post-warmup rows, lies "
            "between finite_prefix_lower and finite_prefix_upper. The tighter "
            "stationary lower bound is not asserted for finite histories."
        ),
        "numerics": "float64 deterministic bounds; not interval-arithmetic certified",
    }


def finite_episode_accuracy_bounds(
    benchmark: dict[str, Any],
    *,
    horizon: int = 1024,
    warmup: int = 32,
) -> dict[str, Any]:
    horizon = _integer(horizon, "horizon", 1)
    warmup = _integer(warmup, "warmup", 0, horizon - 1)
    depth = benchmark["actual_depth"]
    lower = []
    upper = []
    for observed in range(warmup, horizon):
        if observed < depth:
            prefix = benchmark["suffix_history"][observed]
            exact_accuracy = prefix.get("exact_finite_prefix_accuracy", prefix["lower"])
            lower.append(exact_accuracy)
            upper.append(exact_accuracy)
        else:
            lower.append(benchmark["finite_prefix_lower"])
            upper.append(benchmark["finite_prefix_upper"])
    return {
        "horizon": horizon,
        "warmup": warmup,
        "scored_steps_per_episode": horizon - warmup,
        "lower": fsum(lower) / len(lower),
        "upper": fsum(upper) / len(upper),
        "gap": (fsum(upper) - fsum(lower)) / len(lower),
        "prefix_length_at_reset": 0,
        "scope": "fixed-horizon episodes with a stationary reset prior and all post-warmup positions",
    }


def pending_token_geometry(
    *,
    alpha: float = STRATA_ALPHA,
    t0: float = STRATA_T0,
    t1: float = STRATA_T1,
) -> dict[str, Any]:
    model = _model(alpha, t0, t1)
    transition = model.transition_matrix
    emission = model.emission_matrix
    pending_map = np.linalg.solve(transition, emission)
    direction = np.cross(np.ones(3), pending_map[:, 0])
    norm = np.linalg.norm(direction)
    if norm <= 1e-14:
        raise ValueError("pending-token probabilities have no unique simplex-null direction")
    direction /= norm
    source_direction = np.cross(np.ones(3), emission[:, 0])
    source_direction /= np.linalg.norm(source_direction)
    return {
        "transition_matrix": transition.tolist(),
        "emission_matrix": emission.tolist(),
        "arrival_to_pending_token_map": pending_map.tolist(),
        "arrival_null_direction": direction.tolist(),
        "source_null_direction": source_direction.tolist(),
        "simplex_null_residual": float(abs(direction.sum())),
        "pending_null_residual": float(np.max(np.abs(direction @ pending_map))),
        "map_identity_residual": float(np.max(np.abs(transition @ pending_map - emission))),
        "source_posterior": "posterior after conditioning on every delivered token",
        "arrival_belief": "source_posterior @ transition_matrix = info.belief_current",
        "pending_token_probabilities": "source_posterior @ emission_matrix = arrival_belief @ arrival_to_pending_token_map",
        "map_warning": "the inverse-transition map need not be stochastic outside the reachable arrival-belief set",
        "strata": "after zero the source posterior has state-2 mass zero; after n subsequent ones it lies on the projective image of that segment under E1^n",
    }


def monte_carlo_bayes_accuracy(
    *,
    alpha: float = STRATA_ALPHA,
    t0: float = STRATA_T0,
    t1: float = STRATA_T1,
    episodes: int = 8192,
    horizon: int = 1024,
    warmup: int = 32,
    seed: int = 20260909,
) -> dict[str, Any]:
    episodes = _integer(episodes, "episodes", 2)
    horizon = _integer(horizon, "horizon", 1)
    warmup = _integer(warmup, "warmup", 0, horizon - 1)
    model = _model(alpha, t0, t1)
    rng = np.random.default_rng(seed)
    edges = model.edge_transition_matrices
    emission = model.emission_matrix
    source = np.broadcast_to(model.initial_distribution, (episodes, 3)).copy()
    states = rng.choice(3, size=episodes, p=model.initial_distribution)
    constant_guess = int(np.argmax(model.initial_distribution @ emission))
    conditional = np.zeros(episodes)
    realized = np.zeros(episodes)
    constant_realized = np.zeros(episodes)
    switches = np.zeros(episodes)
    benefits = np.zeros(episodes)
    rows = np.arange(episodes)
    for observed in range(horizon):
        probabilities = source @ emission
        guess = np.where(
            probabilities[:, 1 - constant_guess] > probabilities[:, constant_guess],
            1 - constant_guess,
            constant_guess,
        )
        tokens = (rng.random(episodes) >= emission[states, 0]).astype(np.int64)
        if observed >= warmup:
            conditional += probabilities[rows, guess]
            realized += guess == tokens
            constant_realized += constant_guess == tokens
            switches += guess != constant_guess
            benefits += probabilities[rows, guess] - probabilities[:, constant_guess]
        destinations = edges[tokens, states, :]
        destinations /= destinations.sum(axis=1, keepdims=True)
        states = np.minimum(
            (rng.random(episodes)[:, None] >= np.cumsum(destinations, axis=1)).sum(axis=1),
            model.n_states - 1,
        )
        posterior = np.einsum("bi,bij->bj", source, edges[tokens])
        source = posterior / posterior.sum(axis=1, keepdims=True)
    count = horizon - warmup

    def summarize(values: np.ndarray) -> dict[str, float]:
        means = values / count
        mean = float(means.mean())
        standard_error = float(means.std(ddof=1) / np.sqrt(episodes))
        return {
            "mean": mean,
            "episode_cluster_standard_error": standard_error,
            "normal_95_lower": mean - 1.96 * standard_error,
            "normal_95_upper": mean + 1.96 * standard_error,
        }

    return {
        "episodes": episodes,
        "horizon": horizon,
        "warmup": warmup,
        "seed": seed,
        "scored_steps": episodes * count,
        "conditional_bayes_accuracy": summarize(conditional),
        "constant_control_variate_bayes_accuracy": summarize(
            benefits + count * float(np.max(model.initial_distribution @ emission))
        ),
        "control_variate_identity": "E[max_x p(x|history)] = stationary_constant_accuracy + E[max_x p(x|history) - p(constant_guess|history)]",
        "realized_bayes_accuracy": summarize(realized),
        "realized_constant_accuracy": summarize(constant_realized),
        "strict_switch_probability": summarize(switches),
        "conditional_benefit_over_constant": summarize(benefits),
        "paired_realized_benefit_over_constant": summarize(realized - constant_realized),
        "scope": "independent fixed-horizon stationary-reset episodes, exact full-history filter",
        "uncertainty": "normal approximation over independent episode means; token rows are not treated as independent",
        "warning": "Monte Carlo crosscheck, not an exact Bayes reference or deterministic bound",
    }
