"""Unit tests for critic-aware belief probe targets."""

from __future__ import annotations

import numpy as np

from envs.mess3.model import passive_model
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import (
    action_transitions,
    emission_matrix,
)
from experiments.critic_aware_probes_2026_11.targets import (
    alpha_geometry_summary,
    linear_residual_from_belief,
    load_solution,
    mess3_passive_targets,
    save_solution,
    simplex_grid_3,
    solve_pbvi,
)


def _ladder_matrices(variant: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transitions = np.asarray(action_transitions(variant), dtype=np.float64)
    emission = np.asarray(emission_matrix(), dtype=np.float64)
    reward = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=np.float64), (transitions.shape[0], 3)).copy()
    return transitions, emission, reward


def test_trivial_quotient_pbvi_is_exact_and_purely_linear(tmp_path) -> None:
    transitions, emission, reward = _ladder_matrices(1)
    grid = simplex_grid_3(15)
    sol = solve_pbvi(transitions, emission, reward, gamma=0.99, belief_points=grid, max_iters=1500, tol=1e-7)
    assert sol.bellman_residual_max < 1e-4

    rng = np.random.default_rng(0)
    b = rng.dirichlet([1.0, 1.0, 1.0], size=1000)
    geom = alpha_geometry_summary(sol, b)
    # Trivial Quotient: D_ctrl = {0} (c_plan_evals == 0), D_V = span{rho} (1 nonzero m_v_eval), Delta V_nl == 0
    assert geom["c_plan_evals"][0] < 1e-10
    assert geom["m_v_evals"][0] > 100.0
    assert geom["m_v_evals"][1] < 1e-8
    assert geom["v_nl_var_frac"] < 1e-12

    out = tmp_path / "v1.npz"
    save_solution(out, sol)
    reloaded = load_solution(out)
    np.testing.assert_allclose(reloaded.all_alphas, sol.all_alphas)
    assert reloaded.gamma == sol.gamma


def test_mess3_passive_v_star_is_nearly_purely_nonlinear() -> None:
    model = passive_model(alpha=0.85)
    E = np.asarray(model.emission_matrix, dtype=np.float64)
    rng = np.random.default_rng(7)
    b_fit = rng.dirichlet([0.5, 0.5, 0.5], size=4000)
    b_test = rng.dirichlet([0.5, 0.5, 0.5], size=4000)
    tf = mess3_passive_targets(b_fit, E)
    tt = mess3_passive_targets(b_test, E)

    # Q* is exact affine function of b, so nonlinear residual is ~0
    _, _, q_nl_frac = linear_residual_from_belief(b_fit, tf["Q_star"], b_test, tt["Q_star"])
    assert q_nl_frac < 1e-12

    # V* = max_a (b E)_a is symmetric across states, so ~100% nonlinear
    _, _, v_nl_frac = linear_residual_from_belief(b_fit, tf["V_star"], b_test, tt["V_star"])
    assert v_nl_frac > 0.99
