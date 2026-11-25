"""Exact POMDP alpha-vector PBVI solver and critic-aware probe target builders."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class POMDPSolution:
    """Exact (point-based) alpha-vector solution for a finite-state POMDP."""
    gamma: float
    transitions: np.ndarray  # (A, S, S)
    emission: np.ndarray     # (S, K)
    reward: np.ndarray       # (S,) or (A, S)
    alpha_by_action: tuple[np.ndarray, ...]  # A arrays of shape (M_a, S)
    all_alphas: np.ndarray   # (M, S)
    all_actions: np.ndarray  # (M,)
    bellman_residual_max: float
    bellman_residual_mean: float

    def q_values(self, beliefs: np.ndarray) -> np.ndarray:
        """Return Q*(b, a) of shape (N, A) for beliefs of shape (N, S)."""
        b = np.asarray(beliefs, dtype=np.float64)
        return np.stack([(b @ alphas.T).max(axis=1) for alphas in self.alpha_by_action], axis=1)

    def v_values(self, beliefs: np.ndarray) -> np.ndarray:
        """Return V*(b) of shape (N,) for beliefs of shape (N, S)."""
        b = np.asarray(beliefs, dtype=np.float64)
        return (b @ self.all_alphas.T).max(axis=1)

    def selected_alphas(self, beliefs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (alpha_{p*(b)} [N, S], a*(b) [N]) for beliefs of shape (N, S)."""
        b = np.asarray(beliefs, dtype=np.float64)
        idx = (b @ self.all_alphas.T).argmax(axis=1)
        return self.all_alphas[idx], self.all_actions[idx]


def simplex_grid_3(step_count: int = 30) -> np.ndarray:
    """Uniform triangular grid on the 3-state simplex."""
    pts = []
    for i in range(step_count + 1):
        for j in range(step_count + 1 - i):
            k = step_count - i - j
            pts.append((i / step_count, j / step_count, k / step_count))
    return np.asarray(pts, dtype=np.float64)


def _prune_alphas(alphas: np.ndarray, beliefs: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Keep alpha-vectors that are optimal at least at one belief point, deduplicated."""
    vals = beliefs @ alphas.T  # (B, M)
    best = vals.argmax(axis=1)
    kept = alphas[np.unique(best)]
    # Fast vectorized deduplication by rounding to tol
    scale = 1.0 / tol
    _, idx = np.unique(np.round(kept * scale).astype(np.int64), axis=0, return_index=True)
    return kept[np.sort(idx)]


def solve_pbvi(
    transitions: np.ndarray,
    emission: np.ndarray,
    reward: np.ndarray,
    gamma: float = 0.99,
    belief_points: np.ndarray | None = None,
    max_iters: int = 2000,
    tol: float = 1e-6,
) -> POMDPSolution:
    """Solve infinite-horizon POMDP via exact alpha-vector PBVI.

    Timing matches `env.py`: reward r(s) is received pre-transition, then action a
    transitions S' ~ T[a, s, :] and emits O' ~ E[S', :].
    """
    T = np.asarray(transitions, dtype=np.float64)
    E = np.asarray(emission, dtype=np.float64)
    r = np.asarray(reward, dtype=np.float64)
    A, S, _ = T.shape
    K = E.shape[1]
    if r.ndim == 1:
        r_as = np.broadcast_to(r[None, :], (A, S)).copy()
    else:
        r_as = r.copy()

    if belief_points is None:
        if S == 3:
            belief_points = simplex_grid_3(25)  # 351 points covering simplex
        else:
            raise ValueError("Provide explicit belief_points for S != 3")
    B = np.asarray(belief_points, dtype=np.float64)
    N = B.shape[0]

    # Precompute K_ao[a, o, s, s'] = T[a, s, s'] * E[s', o]
    K_ao = T[:, None, :, :] * E.T[None, :, None, :]  # (A, K, S, S)

    # Precompute next beliefs phi(b, a, o) for all (b, a, o)
    next_b = np.einsum("ns,aost->aont", B, K_ao)  # (A, K, N, S)
    probs = next_b.sum(axis=-1, keepdims=True)
    next_b = np.where(probs > 1e-15, next_b / np.maximum(probs, 1e-15), 1.0 / S)

    # Initialize with 1-step reward vectors
    alphas = np.unique(np.round(r_as, 12), axis=0)
    prev_v = (B @ alphas.T).max(axis=1)

    for it in range(max_iters):
        # K_ao_alpha[a, o, m, s] = sum_t K_ao[a, o, s, t] * alphas[m, t]
        K_ao_alpha = np.einsum("aost,mt->aoms", K_ao, alphas)  # (A, K, M, S)
        # Best alpha index for every (a, o, n) in one matmul:
        # scores[a, o, n, m] = sum_s next_b[a, o, n, s] * alphas[m, s]
        scores = np.einsum("aons,ms->aonm", next_b, alphas)
        best_idx = scores.argmax(axis=-1)  # (A, K, N)
        backed = np.broadcast_to(r_as[:, None, :], (A, N, S)).copy()
        for a in range(A):
            for o in range(K):
                backed[a] += gamma * K_ao_alpha[a, o, best_idx[a, o]]
        # For each belief point n, pick the best action's backed-up alpha
        vals = np.einsum("ns,ans->an", B, backed)  # (A, N)
        best_a = vals.argmax(axis=0)  # (N,)
        best_alphas = backed[best_a, np.arange(N)]  # (N, S)
        all_pruned = _prune_alphas(np.vstack([alphas, best_alphas]), B)
        new_v = (B @ all_pruned.T).max(axis=1)
        diff = float(np.max(np.abs(new_v - prev_v)))
        alphas = all_pruned
        prev_v = new_v
        if diff < tol:
            break

    # One final exact per-action backup so Q*(b, a) has exact per-action alpha sets
    K_ao_alpha = np.einsum("aost,mt->aoms", K_ao, alphas)
    scores = np.einsum("aons,ms->aonm", next_b, alphas)
    best_idx = scores.argmax(axis=-1)
    alpha_by_action = []
    for a in range(A):
        backed_a = np.broadcast_to(r_as[a][None, :], (N, S)).copy()
        for o in range(K):
            backed_a += gamma * K_ao_alpha[a, o, best_idx[a, o]]
        alpha_by_action.append(_prune_alphas(backed_a, B))

    all_alphas = np.concatenate(alpha_by_action, axis=0)
    all_actions = np.concatenate([
        np.full(arr.shape[0], a, dtype=np.int32) for a, arr in enumerate(alpha_by_action)
    ])

    res = bellman_residual(B, T, E, r_as, gamma, all_alphas)
    return POMDPSolution(
        gamma=gamma,
        transitions=T,
        emission=E,
        reward=r_as,
        alpha_by_action=tuple(alpha_by_action),
        all_alphas=all_alphas,
        all_actions=all_actions,
        bellman_residual_max=float(np.max(res)),
        bellman_residual_mean=float(np.mean(res)),
    )


def bellman_residual(
    beliefs: np.ndarray,
    transitions: np.ndarray,
    emission: np.ndarray,
    reward_as: np.ndarray,
    gamma: float,
    alphas: np.ndarray,
) -> np.ndarray:
    """Return |V(b) - (TV)(b)| for each belief b."""
    B = np.asarray(beliefs, dtype=np.float64)
    A, S, _ = transitions.shape
    K = emission.shape[1]
    v_curr = (B @ alphas.T).max(axis=1)
    q_backup = np.zeros((B.shape[0], A), dtype=np.float64)
    for a in range(A):
        q_a = B @ reward_as[a]
        b_next_prior = B @ transitions[a]  # (N, S)
        for o in range(K):
            unnorm = b_next_prior * emission[:, o][None, :]
            p_o = unnorm.sum(axis=1)
            b_o = np.where(p_o[:, None] > 1e-15, unnorm / np.maximum(p_o[:, None], 1e-15), 1.0 / S)
            v_o = (b_o @ alphas.T).max(axis=1)
            q_a = q_a + gamma * p_o * v_o
        q_backup[:, a] = q_a
    return np.abs(v_curr - q_backup.max(axis=1))


def linear_residual_from_belief(
    b_fit: np.ndarray,
    y_fit: np.ndarray,
    b_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit best affine function of belief b to y on fit split; return (res_fit, res_test, nl_var_frac)."""
    bf = np.asarray(b_fit, dtype=np.float64)
    bt = np.asarray(b_test, dtype=np.float64)
    yf = np.asarray(y_fit, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)
    squeeze = yf.ndim == 1
    if squeeze:
        yf = yf[:, None]
        yt = yt[:, None]
    xf = np.concatenate([bf, np.ones((bf.shape[0], 1))], axis=1)
    xt = np.concatenate([bt, np.ones((bt.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(xf, yf, rcond=1e-12)
    res_f = yf - xf @ coef
    res_t = yt - xt @ coef
    total_var = float(np.mean(np.sum((yt - yt.mean(axis=0, keepdims=True)) ** 2, axis=1)))
    res_var = float(np.mean(np.sum((res_t - res_t.mean(axis=0, keepdims=True)) ** 2, axis=1)))
    frac = res_var / total_var if total_var > 1e-15 else 0.0
    if squeeze:
        return res_f[:, 0], res_t[:, 0], frac
    return res_f, res_t, frac


def alpha_geometry_summary(solution: POMDPSolution, beliefs: np.ndarray) -> dict:
    """Compute D_ctrl (centered plan covariance) vs D_V (mean alpha + uncentered M_V) diagnostics."""
    b = np.asarray(beliefs, dtype=np.float64)
    S = b.shape[1]
    proj = np.eye(S) - np.ones((S, S)) / S
    alphas, actions = solution.selected_alphas(b)
    tangents = alphas @ proj  # (N, S)
    mean_alpha = tangents.mean(axis=0)
    centered = tangents - mean_alpha[None, :]
    c_plan = (centered.T @ centered) / centered.shape[0]
    m_v = (tangents.T @ tangents) / tangents.shape[0]
    evals_plan = np.sort(np.linalg.eigvalsh(c_plan))[::-1]
    evals_mv = np.sort(np.linalg.eigvalsh(m_v))[::-1]
    # Nonlinear variance fraction of V*(b) on beliefs
    v = solution.v_values(b)
    _, res, nl_frac = linear_residual_from_belief(b, v, b, v)
    return {
        "n_alphas": int(solution.all_alphas.shape[0]),
        "action_fractions": np.bincount(actions, minlength=len(solution.alpha_by_action)).astype(float).tolist(),
        "bellman_residual_max": solution.bellman_residual_max,
        "c_plan_evals": evals_plan.tolist(),
        "m_v_evals": evals_mv.tolist(),
        "mean_alpha_tangent": mean_alpha.tolist(),
        "mean_alpha_norm": float(np.linalg.norm(mean_alpha)),
        "v_std": float(np.std(v)),
        "dv_nl_std": float(np.std(res)),
        "v_nl_var_frac": float(nl_frac),
    }


def mess3_passive_targets(beliefs: np.ndarray, emission: np.ndarray) -> dict[str, np.ndarray]:
    """Exact gamma=0 optimal V*(b) and Q*(b, a) = (b @ E) for passive Mess3 token-guessing."""
    b = np.asarray(beliefs, dtype=np.float64)
    E = np.asarray(emission, dtype=np.float64)
    q = b @ E  # (N, 3)
    v = q.max(axis=1)  # (N,)
    return {"V_star": v, "Q_star": q, "A_star": q - v[:, None]}


def save_solution(path, sol: POMDPSolution) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "gamma": np.asarray(sol.gamma, dtype=np.float64),
        "transitions": sol.transitions,
        "emission": sol.emission,
        "reward": sol.reward,
        "all_alphas": sol.all_alphas,
        "all_actions": sol.all_actions,
        "bellman_residual_max": np.asarray(sol.bellman_residual_max, dtype=np.float64),
        "bellman_residual_mean": np.asarray(sol.bellman_residual_mean, dtype=np.float64),
        "n_actions": np.asarray(len(sol.alpha_by_action), dtype=np.int32),
    }
    for a, arr in enumerate(sol.alpha_by_action):
        arrays[f"alpha_action_{a}"] = arr
    np.savez(path, **arrays)


def load_solution(path) -> POMDPSolution:
    with np.load(path) as data:
        n_actions = int(data["n_actions"])
        alpha_by_action = tuple(data[f"alpha_action_{a}"].copy() for a in range(n_actions))
        return POMDPSolution(
            gamma=float(data["gamma"]),
            transitions=data["transitions"].copy(),
            emission=data["emission"].copy(),
            reward=data["reward"].copy(),
            alpha_by_action=alpha_by_action,
            all_alphas=data["all_alphas"].copy(),
            all_actions=data["all_actions"].copy(),
            bellman_residual_max=float(data["bellman_residual_max"]),
            bellman_residual_mean=float(data["bellman_residual_mean"]),
        )

