"""Near-Bayes-optimal reward references for the v2 env via belief-grid value iteration.

The belief-MDP (filtered belief after the current token) is solved by discounted value
iteration on a triangulated simplex grid with barycentric interpolation; the resulting
one-step-lookahead policy is then simulated with the exact filter using the probe protocol
(uniform reset, 1,024-step episodes, 64-step warmup excluded). Grid policies are not certified
optimal; the grid/discount sweep shows convergence. Simple references are also reported.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as env

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/reward_reference.json"
EPISODE, WARMUP = 1024, 64


def grid(n):
    ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)])
    index = -np.ones((n + 2, n + 2), dtype=int)
    index[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
    points = np.column_stack([ij / n, 1 - ij.sum(1) / n])
    return points, index


def interpolate(beliefs, n, index):
    x, y = beliefs[..., 0] * n, beliefs[..., 1] * n
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    i, j = np.clip(i, 0, n), np.clip(j, 0, n)
    fx, fy = x - i, y - j
    upper = (fx + fy > 1) & (i + j + 1 <= n - 0)
    upper &= (i + j + 2 <= n)
    lower_ok = ~upper
    v0 = np.where(lower_ok, index[i, j], index[np.minimum(i + 1, n), np.minimum(j + 1, n)])
    v1 = index[np.minimum(i + 1, n), j]
    v2 = index[i, np.minimum(j + 1, n)]
    w0 = np.where(lower_ok, 1 - fx - fy, fx + fy - 1)
    w1 = np.where(lower_ok, fx, 1 - fy)
    w2 = np.where(lower_ok, fy, 1 - fx)
    v1 = np.where(v1 < 0, v0, v1)
    v2 = np.where(v2 < 0, v0, v2)
    w = np.clip(np.stack([w0, w1, w2], -1), 0, None)
    w /= w.sum(-1, keepdims=True)
    return np.stack([v0, v1, v2], -1), w


def successors(beliefs, kernels, emission):
    predicted = np.einsum("ns,ast->nat", beliefs, kernels)
    joint = predicted[..., None] * emission[None, None]
    prob = joint.sum(2)
    post = joint / np.where(prob[:, :, None, :] > 0, prob[:, :, None, :], 1)
    return prob, np.moveaxis(post, 2, 3)


def solve(variant, n, gamma, tol=1e-9):
    kernels, emission = env.action_transitions(variant), env.emission_matrix()
    points, index = grid(n)
    prob, post = successors(points, kernels, emission)
    verts, weights = interpolate(post, n, index)
    value = np.zeros(len(points))
    for _ in range(100000):
        q = points[:, 2, None] + gamma * (prob * (value[verts] * weights).sum(-1)).sum(-1)
        new = q.max(1)
        if np.abs(new - value).max() < tol * (1 - gamma):
            value = new
            break
        value = new
    return value, index


def lookahead_policy(value, n, index, kernels, emission, gamma):
    def act(beliefs):
        prob, post = successors(beliefs, kernels, emission)
        verts, weights = interpolate(post, n, index)
        return (prob * (value[verts] * weights).sum(-1)).sum(-1).argmax(1)
    return act


def simulate(variant, policy, episodes=4096, seed=20261115, history=False):
    rng = np.random.default_rng(seed)
    kernels, emission = env.action_transitions(variant), env.emission_matrix()
    state = rng.choice(3, size=episodes, p=env.INITIAL_STATE_DISTRIBUTION)
    cumulative_e = emission.cumsum(1)
    token = (rng.random(episodes)[:, None] > cumulative_e[state]).sum(1)
    belief = env.INITIAL_STATE_DISTRIBUTION * emission[:, token].T
    belief /= belief.sum(1, keepdims=True)
    tokens = np.full((episodes, 3), -1)
    tokens[:, -1] = token
    totals = np.zeros(episodes)
    for t in range(EPISODE):
        action = policy(tokens) if history else policy(belief)
        if t >= WARMUP:
            totals += state == env.REWARD_STATE
        state = (rng.random(episodes)[:, None] > kernels[action, state].cumsum(1)).sum(1)
        token = (rng.random(episodes)[:, None] > cumulative_e[state]).sum(1)
        belief = np.einsum("ns,nst->nt", belief, kernels[action]) * emission[:, token].T
        belief /= belief.sum(1, keepdims=True)
        tokens = np.column_stack([tokens[:, 1:], token])
    per_episode = totals / (EPISODE - WARMUP)
    boot = np.random.default_rng(seed + 1).choice(per_episode, (2000, episodes)).mean(1)
    return {"mean": float(per_episode.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "episode_sd": float(per_episode.std(ddof=1)), "episodes": episodes}


def reference(variant):
    kernels, emission = env.action_transitions(variant), env.emission_matrix()
    out = {"constant_action": {}, "belief_grid": []}
    for a in range(3):
        out["constant_action"][str(a)] = simulate(variant, lambda b, a=a: np.full(len(b), a))
    best = None
    for policy in itertools.product(range(3), repeat=3):
        result = simulate(variant, lambda tok, p=np.array(policy): p[tok[:, -1]], episodes=1024,
                          history=True)
        if best is None or result["mean"] > best[1]["mean"]:
            best = (policy, result)
    out["current_token"] = {"policy": list(best[0]),
                            **simulate(variant, lambda tok, p=np.array(best[0]): p[tok[:, -1]],
                                       history=True)}
    out["myopic_belief"] = simulate(
        variant, lambda b: np.einsum("ns,ast->nat", b, kernels)[..., env.REWARD_STATE].argmax(1))
    for n, gamma in ((60, .99), (120, .99), (120, .995), (200, .995), (200, .998)):
        value, index = solve(variant, n, gamma)
        result = simulate(variant, lookahead_policy(value, n, index, kernels, emission, gamma))
        out["belief_grid"].append({"grid": n, "gamma": gamma, **result})
        print(variant, n, gamma, result["mean"], flush=True)
    out["best_belief_grid"] = max(out["belief_grid"], key=lambda r: r["mean"])
    return out


def main():
    report = {
        "protocol": {"episode_length": EPISODE, "warmup": WARMUP, "reset": "uniform",
                     "reward": "pre-transition reward-state occupancy, warmup excluded",
                     "caveat": "grid value-iteration lookahead policy; near-optimal, not a certified Bayes bound"},
        "variants": {str(v): reference(v) for v in (2, 3)},
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    for v, r in report["variants"].items():
        print(v, {k: round(r[k]["mean"], 4) for k in ("current_token", "myopic_belief", "best_belief_grid")},
              {a: round(c["mean"], 4) for a, c in r["constant_action"].items()})


if __name__ == "__main__":
    main()
