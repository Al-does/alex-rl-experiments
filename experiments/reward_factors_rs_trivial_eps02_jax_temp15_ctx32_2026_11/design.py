"""Analytic/numerical verification of the control and value geometry.

Analytic facts (checked numerically below):

* Each factor's hidden chain is strongly lumpable onto L={0,1} vs R={2}: rows
  0/1 of every action's transition put identical mass on state 2, and
  P(token 2 | 0) = P(token 2 | 1). The L-substate tokens 0/1 are unequal,
  however, so the exact posterior rho_f = P(s_f = 2) is NOT a scalar
  recursion: updating rho_f uses delta_f = P(0) - P(1) (a filtering
  coordinate with no direct reward or control role). By the 0<->1 symmetry
  V_f(rho, delta) = V_f(rho, -delta), so delta_f has zero linear value weight.
* Factors are independent, observed by separate tokens and rewarded additively,
  so the Bellman operator separates: Q*((b1, b2), (a1, a2)) = Q1(b1, a1) + Q2(b2, a2)
  and V* = V1(b1) + V2(b2).
* In the Trivial factor action 1 is optimal in every fully observed state, so a
  constant action is POMDP-optimal and V2 is exactly linear (alpha_2 symmetric in
  0/1, i.e. a pure rho_2 direction). The optimal control therefore reads only
  factor 1, while V* has a second, rho_2, direction.

Numerics: full-simplex value iteration per factor on a (rho, u) grid,
u = delta / (1 - rho); a joint 9-state Bellman-residual check of V1 + V2;
exact-filter simulations of the Bayes policy and references; visited-belief
value regressions, local alpha-vector geometry and decoding references.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.process import (
    ACTION_PAIRS, REWARD_STATE, factor_matrices, rs_trivial_process,
)

GAMMA = 0.99
GRID_RHO = 301
GRID_U = 101
CONTEXT_LEN = 32
EPISODE_LENGTH = 1024
WARMUP = 64
RESULTS = Path(__file__).with_name("results")


def to_coords(belief):
    belief = np.asarray(belief, float)
    rho = belief[..., 2]
    mass = belief[..., 0] + belief[..., 1]
    u = np.where(mass > 1e-12, (belief[..., 0] - belief[..., 1]) / np.maximum(mass, 1e-12), 0.)
    return np.clip(rho, 0, 1), np.clip(u, -1, 1)


def from_coords(rho, u):
    return np.stack([(1 - rho) * (1 + u) / 2, (1 - rho) * (1 - u) / 2, rho], -1)


def interp2(table, rho, u):
    """Bilinear interpolation of ``table[GRID_RHO, GRID_U]`` on the (rho, u) grid."""
    x = np.clip(rho, 0, 1) * (table.shape[0] - 1)
    y = (np.clip(u, -1, 1) + 1) / 2 * (table.shape[1] - 1)
    i = np.clip(np.floor(x).astype(int), 0, table.shape[0] - 2)
    j = np.clip(np.floor(y).astype(int), 0, table.shape[1] - 2)
    fx, fy = x - i, y - j
    return ((1 - fx) * (1 - fy) * table[i, j] + fx * (1 - fy) * table[i + 1, j]
            + (1 - fx) * fy * table[i, j + 1] + fx * fy * table[i + 1, j + 1])


def successors(transitions, emission, belief, action):
    prior = np.einsum("...s,...st->...t", belief, transitions[action])
    joint = prior[..., :, None] * emission[None]  # (..., state, token)
    probs = joint.sum(-2)
    return probs, joint / np.maximum(probs[..., None, :], 1e-300)


def value_iteration(transitions, emission, tol=1e-9, max_iter=20_000):
    rho, u = np.meshgrid(np.linspace(0, 1, GRID_RHO), np.linspace(-1, 1, GRID_U), indexing="ij")
    belief = from_coords(rho, u)
    nexts = []
    for a in range(transitions.shape[0]):
        probs, posts = successors(transitions, emission, belief, a)
        nexts.append([(probs[..., o], *to_coords(posts[..., o])) for o in range(emission.shape[1])])
    value = np.zeros_like(rho)
    for iteration in range(max_iter):
        q = np.stack([rho + GAMMA * sum(p * interp2(value, r, v) for p, r, v in nxt)
                      for nxt in nexts])
        new = q.max(0)
        delta = np.abs(new - value).max()
        value = new
        if delta < tol:
            break
    return {"value": value, "q": q, "iterations": iteration + 1, "residual": float(delta)}


@lru_cache(maxsize=None)
def factor_solutions():
    first, second, emission = factor_matrices()
    return tuple(value_iteration(t, emission) for t in (first, second))


def factor_q(factor, belief):
    rho, u = to_coords(belief)
    return np.stack([interp2(q, rho, u) for q in factor_solutions()[factor]["q"]], -1)


def factor_value(factor, belief):
    rho, u = to_coords(belief)
    return interp2(factor_solutions()[factor]["value"], rho, u)


def value_star(first, second):
    """(1 - gamma) V*(b) from the factor-marginal beliefs (product belief)."""
    return (1 - GAMMA) * (factor_value(0, first) + factor_value(1, second))


def scalar_rho_update(transitions, emission, rho, action, merged_token):
    """Merged-token (token 2 vs not) scalar filter: ignores the 0/1 token identity."""
    prior = (1 - rho) * transitions[action, 0, 2] + rho * transitions[action, 2, 2]
    c_l, c_r = emission[0, 2], emission[2, 2]
    like_r = np.where(merged_token, c_r, 1 - c_r)
    like_l = np.where(merged_token, c_l, 1 - c_l)
    return prior * like_r / (prior * like_r + (1 - prior) * like_l)


def _filter(belief, transitions, emission, actions, tokens):
    prior = np.einsum("ns,nst->nt", belief, transitions[actions])
    post = prior * emission[:, tokens].T
    return post / post.sum(1, keepdims=True)


def simulate(policy, *, episodes=512, length=EPISODE_LENGTH, seed=0):
    """Exact-filter rollouts; ``policy(rng, beliefs, history) -> (a1, a2)``."""
    rng = np.random.default_rng(seed)
    first, second, emission = factor_matrices()
    trans = (first, second)
    states = rng.integers(0, 3, (2, episodes))
    tokens = np.minimum((rng.random((2, episodes, 1)) > np.cumsum(emission[states], -1)).sum(-1), 2)
    beliefs = [emission[:, tokens[f]].T / emission[:, tokens[f]].T.sum(1, keepdims=True)
               for f in range(2)]
    history = {"tokens": [], "actions": []}
    out = {k: [] for k in ("b1", "b2", "token_1", "token_2", "a1", "a2", "r1", "r2")}
    for t in range(length):
        history["tokens"].append(tokens.copy())
        a1, a2 = policy(rng, beliefs, history)
        for f in range(2):
            out[f"b{f + 1}"].append(beliefs[f])
            out[f"token_{f + 1}"].append(tokens[f].copy())
            out[f"r{f + 1}"].append(states[f] == REWARD_STATE)
        out["a1"].append(a1)
        out["a2"].append(a2)
        history["actions"].append(np.stack([a1, a2]))
        for f, action in enumerate((a1, a2)):
            cdf = np.cumsum(trans[f][action, states[f]], 1)
            states[f] = np.minimum((rng.random((episodes, 1)) > cdf).sum(1), 2)
            cdf = np.cumsum(emission[states[f]], 1)
            tokens[f] = np.minimum((rng.random((episodes, 1)) > cdf).sum(1), 2)
            beliefs[f] = _filter(beliefs[f], trans[f], emission, action, tokens[f])
    return {k: np.stack(v, 1) for k, v in out.items()}


def window_beliefs(tokens, actions, transitions, emission, context_len=CONTEXT_LEN):
    """Exact filter restarted from the uniform prior over only the last ``context_len``
    tokens (``actions[:, t]`` is the action after token t). Returns (n, T, 3)."""
    n, length = tokens.shape
    out = np.empty((n, length, 3))
    for t in range(length):
        start = max(0, t - context_len + 1)
        b = emission[:, tokens[:, start]].T / emission[:, tokens[:, start]].T.sum(1, keepdims=True)
        for k in range(start + 1, t + 1):
            b = _filter(b, transitions, emission, actions[:, k - 1], tokens[:, k])
        out[:, t] = b
    return out


def _affine_r2(x, y):
    x = np.column_stack([np.asarray(x).reshape(len(y), -1), np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(1 - (y - x @ coef).var() / y.var())


def _report(sim):
    sl = slice(WARMUP, None)
    r1, r2 = sim["r1"][:, sl].mean(1), sim["r2"][:, sl].mean(1)
    total = r1 + r2
    return {"reward_per_step": float(total.mean()),
            "reward_per_step_episode_se": float(total.std(ddof=1) / np.sqrt(len(total))),
            "reward_components_per_step": [float(r1.mean()), float(r2.mean())],
            "episodes": int(len(total)), "steps_per_episode": int(sim["r1"].shape[1]),
            "warmup_excluded": WARMUP}


def bayes_policy(rng, beliefs, history):
    return tuple(factor_q(f, beliefs[f]).argmax(-1) for f in range(2))


def verify(episodes=512, seed=0):
    first, second, emission = factor_matrices()
    trans = (first, second)
    s1, s2 = factor_solutions()
    rho_grid = np.linspace(0, 1, GRID_RHO)

    # Control: factor-2 argmax everywhere; factor-1 dependence on u at fixed rho.
    a1_grid, a2_grid = s1["q"].argmax(0), s2["q"].argmax(0)
    q2_sorted = np.sort(s2["q"], 0)
    rows_varying = [i for i in range(GRID_RHO) if len(set(a1_grid[i])) > 1]
    switch_by_u = []
    for j in range(GRID_U):
        col = a1_grid[:, j]
        switch_by_u.append(rho_grid[1:][np.diff(col) != 0].tolist())
    # Trivial factor fully observed MDP: action 1 optimal in every state?
    mdp_value = np.zeros(3)
    for _ in range(20_000):
        mdp_q = np.stack([(np.arange(3) == REWARD_STATE) + GAMMA * second[a] @ mdp_value for a in range(3)])
        mdp_value = mdp_q.max(0)

    # Joint 9-state Bellman residual of V1 + V2 at random product beliefs.
    process = rs_trivial_process()
    rng = np.random.default_rng(seed)
    b1, b2 = rng.dirichlet(np.ones(3), 2000), rng.dirichlet(np.ones(3), 2000)
    joint_b = np.einsum("ni,nj->nij", b1, b2).reshape(-1, 9)
    T, E, R = (np.asarray(process.transitions), np.asarray(process.emission), np.asarray(process.reward))
    joint_q = []
    for a in range(9):
        probs, posts = successors(T, E, joint_b, a)
        cont = 0.
        for o in range(9):
            post = posts[..., o].reshape(-1, 3, 3)
            cont = cont + probs[:, o] * (factor_value(0, post.sum(2)) + factor_value(1, post.sum(1)))
        joint_q.append(joint_b @ R + GAMMA * cont)
    joint_q = np.stack(joint_q, -1)
    additive = factor_value(0, b1) + factor_value(1, b2)
    pairs = np.array(ACTION_PAIRS)
    joint_best = pairs[joint_q.argmax(-1)]

    # Simulations.
    sim = simulate(bayes_policy, episodes=episodes, seed=seed)
    sl = slice(WARMUP, None)
    vb1, vb2 = sim["b1"][:, sl].reshape(-1, 3), sim["b2"][:, sl].reshape(-1, 3)
    v1 = (1 - GAMMA) * factor_value(0, vb1)
    v2 = (1 - GAMMA) * factor_value(1, vb2)
    v = v1 + v2
    rho1, rho2 = vb1[:, 2], vb2[:, 2]
    d1, d2 = vb1[:, 0] - vb1[:, 1], vb2[:, 0] - vb2[:, 1]

    # Local alpha vectors from finite differences in (rho, delta) belief coordinates.
    def alpha(factor, b):
        eps = 1e-3
        rho, dl = b[:, 2], b[:, 0] - b[:, 1]
        def val(r, d):
            m = 1 - r
            return factor_value(factor, np.stack([(m + d) / 2, (m - d) / 2, r], -1))
        g_r = (val(np.minimum(rho + eps, 1), dl) - val(np.maximum(rho - eps, 0), dl)) / (
            np.minimum(rho + eps, 1) - np.maximum(rho - eps, 0))
        span = np.maximum(1 - rho - np.abs(dl), 0)
        h = np.minimum(eps, np.maximum(span / 2, 1e-6))
        g_d = (val(rho, dl + h) - val(rho, dl - h)) / (2 * h)
        base = val(rho, dl)
        e_r = np.array([0, 0, 1.])
        e_d = np.array([1, -1, 0.])
        return (base[:, None] + g_r[:, None] * (e_r - rho[:, None])
                + g_d[:, None] * (e_d - dl[:, None]))
    sample = slice(0, 200_000)
    al = (1 - GAMMA) * (alpha(0, vb1[sample])[:, :, None] + alpha(1, vb2[sample])[:, None, :]).reshape(-1, 9)
    center = np.eye(9) - 1 / 9
    centered = al @ center
    evals, evecs = np.linalg.eigh(np.cov(centered.T))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    ind = lambda f, vec: center @ (np.repeat(vec, 3) if f == 0 else np.tile(vec, 3))
    dirs = {"rho_1": ind(0, np.array([0, 0, 1.])), "delta_1": ind(0, np.array([1, -1, 0.])),
            "rho_2": ind(1, np.array([0, 0, 1.])), "delta_2": ind(1, np.array([1, -1, 0.]))}
    unit = lambda x: x / np.linalg.norm(x)
    mean_alpha = centered.mean(0)
    top = evecs[:, :2]
    residual = mean_alpha - top @ (top.T @ mean_alpha)

    # Performance references.
    def constant(a1, a2):
        return lambda rng, b, h: (np.full(len(b[0]), a1), np.full(len(b[0]), a2))
    def rho_threshold(rng, b, h):  # factor-1 action from rho_1 only (u = 0 column), a2 = 1
        q = np.stack([np.interp(b[0][:, 2], rho_grid, s1["q"][a][:, GRID_U // 2]) for a in range(3)], -1)
        return q.argmax(-1), np.ones(len(b[0]), int)
    def window_policy(rng, b, h):
        tokens = np.stack(h["tokens"][-CONTEXT_LEN:], -1)  # (2, n, k)
        acts = np.stack(h["actions"][-(tokens.shape[-1] - 1):], -1) if tokens.shape[-1] > 1 else None
        wb = []
        for f in range(2):
            a = acts[f] if acts is not None else np.zeros((tokens.shape[1], 0), int)
            a = np.concatenate([a, np.zeros((tokens.shape[1], 1), int)], 1)
            wb.append(window_beliefs(tokens[f], a, trans[f], emission,
                                     context_len=tokens.shape[-1])[:, -1])
        return bayes_policy(rng, wb, h)
    def uniform(rng, b, h):
        return rng.integers(0, 3, len(b[0])), rng.integers(0, 3, len(b[0]))
    perf = {"bayes_optimal": _report(sim)}
    refs = {"rho_1_threshold_a2_1": rho_threshold, "uniform_random": uniform,
            **{f"constant_a1_{a}_a2_1": constant(a, 1) for a in range(3)},
            "constant_a1_0_a2_0": constant(0, 0), "constant_a1_0_a2_2": constant(0, 2)}
    for name, policy in refs.items():
        perf[name] = _report(simulate(policy, episodes=256, seed=seed + 1))
    perf["bayes_optimal_paired_seed"] = _report(simulate(bayes_policy, episodes=256, seed=seed + 1))
    perf["context_32_exact_filter_bayes_actions"] = _report(
        simulate(window_policy, episodes=64, length=320, seed=seed + 2))

    # Decoding references under the Bayes policy (first 128 episodes).
    n = 128
    refs_decode = {}
    for f in range(2):
        tok, acts = sim[f"token_{f + 1}"][:n], sim[f"a{f + 1}"][:n]
        exact = sim[f"b{f + 1}"][:n]
        wb = window_beliefs(tok, acts, trans[f], emission)
        scalar = np.empty(tok.shape)
        r = exact[:, 0, 2]
        scalar[:, 0] = r
        for t in range(1, tok.shape[1]):
            r = scalar_rho_update(trans[f], emission, r, acts[:, t - 1], tok[:, t] == REWARD_STATE)
            scalar[:, t] = r
        e_rho, e_d = exact[:, sl, 2].ravel(), (exact[:, sl, 0] - exact[:, sl, 1]).ravel()
        merged = (tok == REWARD_STATE).astype(float)
        counts = np.stack([np.convolve(m, np.ones(CONTEXT_LEN))[:tok.shape[1]] for m in merged])
        onehot = np.eye(3)[tok[:, sl].ravel()]
        def r2(pred, target):
            return float(1 - ((pred - target) ** 2).mean() / target.var())
        refs_decode[f"factor_{f + 1}"] = {
            "rho_context_32_exact_filter_r_squared": r2(wb[:, sl, 2].ravel(), e_rho),
            "delta_context_32_exact_filter_r_squared": r2((wb[:, sl, 0] - wb[:, sl, 1]).ravel(), e_d),
            "rho_full_history_merged_token_scalar_filter_r_squared": r2(scalar[:, sl].ravel(), e_rho),
            "rho_current_token_affine_r_squared": _affine_r2(onehot, e_rho),
            "delta_current_token_affine_r_squared": _affine_r2(onehot, e_d),
            "rho_context_32_token2_count_affine_r_squared": _affine_r2(counts[:, sl].ravel(), e_rho),
            "rho_visited_std": float(e_rho.std()), "delta_visited_std": float(e_d.std()),
        }

    features = {"rho_1": rho1, "rho_2": rho2, "rho_1_rho_2": np.column_stack([rho1, rho2]),
                "rho_1_rho_2_delta_1_delta_2": np.column_stack([rho1, rho2, d1, d2]),
                "factor_1_marginal": vb1, "factor_2_marginal": vb2,
                "joint_belief_9d": np.einsum("ni,nj->nij", vb1, vb2).reshape(-1, 9)}
    sub = slice(0, 200_000)
    return {
        "gamma": GAMMA, "grid": {"rho": GRID_RHO, "u": GRID_U},
        "value_iteration": {
            "iterations": [s1["iterations"], s2["iterations"]],
            "residual": [s1["residual"], s2["residual"]],
            "joint_9_state_bellman_max_abs_residual_of_v1_plus_v2": float(
                np.abs(joint_q.max(-1) - additive).max()),
            "joint_value_range": [float(additive.min()), float(additive.max())],
            "joint_greedy_a2_fraction": {str(a): float((joint_best[:, 1] == a).mean()) for a in range(3)},
        },
        "control": {
            "factor_2_argmax_fraction_full_simplex": {str(a): float((a2_grid == a).mean()) for a in range(3)},
            "factor_2_min_q_gap": float((q2_sorted[-1] - q2_sorted[-2]).min()),
            "factor_2_fully_observed_mdp_argmax_by_state": mdp_q.argmax(0).tolist(),
            "factor_1_argmax_fraction_full_simplex": {str(a): float((a1_grid == a).mean()) for a in range(3)},
            "factor_1_rho_rows_where_argmax_varies_with_u": len(rows_varying),
            "factor_1_switch_rho_range_over_u": [
                float(min(min(s) for s in switch_by_u if s)), float(max(max(s) for s in switch_by_u if s))],
            "factor_1_switch_rho_at_u0": switch_by_u[GRID_U // 2],
        },
        "value": {
            "visited_variance_share": {"factor_1": float(v1.var() / v.var()), "factor_2": float(v2.var() / v.var())},
            "visited_affine_r_squared_of_scaled_v_star": {
                k: _affine_r2(x[sub], v[sub]) for k, x in features.items()},
            "factor_2_value_affine_in_rho_2_r_squared": _affine_r2(rho2[sub], v2[sub]),
            "factor_1_value_nonlinearity_1_minus_affine_r2_in_belief": 1 - _affine_r2(vb1[sub], v1[sub]),
        },
        "alpha_vector_geometry": {
            "eigenvalues": evals.tolist(),
            "top_2_abs_cosines": [{k: float(abs(unit(evecs[:, i]) @ unit(d))) for k, d in dirs.items()}
                                  for i in range(2)],
            "mean_alpha_residual_off_top_2": float(np.linalg.norm(residual)),
            "mean_alpha_residual_abs_cosines": {k: float(abs(unit(residual) @ unit(d))) for k, d in dirs.items()},
            "note": ("covariance of local supporting alpha vectors (value gradients) at visited beliefs, "
                     "centered on the 9-state simplex; its span is the factor-1 directions along which "
                     "the optimal plan's value gradient changes; the mean alpha off that span is the "
                     "constant factor-2 value slope"),
        },
        "performance_references_per_step": perf,
        "decoding_references_under_bayes_policy": refs_decode,
        "visited_rows": int(len(rho1)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--output", type=Path, default=RESULTS / "design.json")
    args = parser.parse_args(argv)
    report = verify(args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "performance_references_per_step"}, indent=1))
    print(json.dumps({k: (v["reward_per_step"], v["reward_components_per_step"])
                      for k, v in report["performance_references_per_step"].items()}, indent=1))


if __name__ == "__main__":
    main()
