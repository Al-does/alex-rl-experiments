"""Belief-MDP theory for the v2 ladder family: Q-gaps, value of the nuisance coordinate.

Family: Tilt X = (+1, lam, -1), Tilt Y = (-1, -lam, -1) log-odds directions (lam=1 is the
Reward-State Quotient, lam=-1 the Full-State Quotient), optional emission asymmetry kappa
(P(C|S1) = .85 - kappa, P(C|S2) = .85 + kappa), and the Trivial Quotient (variant 1).
delta = b1 - b2 (zero-based b[0]-b[1]) and rho = b3 are the within-class coordinates.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_persistent_nuisance_jax_2026_11.shared import tilt_reward_state
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
from experiments.within_class_exploration_sweep_2026_11.reward_reference import (
    grid, interpolate, successors,
)

HERE = Path(__file__).resolve().parent
EPISODE, WARMUP = 1024, 64


def directions(lam):
    return np.array([[0., 1., -1.], [0., lam, -lam], [0., -1., -1.]])


def kernels_for(lam=None, variant=None):
    if variant is not None:
        return v2.action_transitions(variant)
    d = directions(lam)
    return np.stack([v2.BASELINE_TRANSITION_MATRIX] + [
        tilt_reward_state(v2.BASELINE_TRANSITION_MATRIX, d[:, a], effect_size=v2.EFFECT_SIZE)
        for a in (1, 2)])


def emission_for(kappa=0.):
    return np.array([[(.15 + kappa) * .75, (.15 + kappa) * .25, .85 - kappa],
                     [(.15 - kappa) * .25, (.15 - kappa) * .75, .85 + kappa], [0., 0., 1.]])


def solve(kernels, emission, n=160, gamma=.995, tol=1e-9):
    points, index = grid(n)
    prob, post = successors(points, kernels, emission)
    verts, weights = interpolate(post, n, index)
    value = np.zeros(len(points))
    for _ in range(100000):
        new = (points[:, 2, None] + gamma * (prob * (value[verts] * weights).sum(-1)).sum(-1)).max(1)
        done = np.abs(new - value).max() < tol * (1 - gamma)
        value = new
        if done:
            break

    def q(beliefs):
        prob, post = successors(beliefs, kernels, emission)
        verts, weights = interpolate(post, n, index)
        return beliefs[:, 2, None] + gamma * (prob * (value[verts] * weights).sum(-1)).sum(-1)
    return q


def simulate(kernels, emission, policy, episodes=1024, seed=20261115, keep=True):
    rng = np.random.default_rng(seed)
    cumulative = emission.cumsum(1)
    state = rng.choice(3, size=episodes, p=v2.INITIAL_STATE_DISTRIBUTION)
    token = (rng.random(episodes)[:, None] > cumulative[state]).sum(1)
    belief = v2.INITIAL_STATE_DISTRIBUTION * emission[:, token].T
    belief /= belief.sum(1, keepdims=True)
    beliefs, actions, reward = [], [], np.zeros(episodes)
    for t in range(EPISODE):
        action = policy(belief)
        if t >= WARMUP:
            reward += state == 2
            if keep:
                beliefs.append(belief), actions.append(action)
        state = (rng.random(episodes)[:, None] > kernels[action, state].cumsum(1)).sum(1)
        token = (rng.random(episodes)[:, None] > cumulative[state]).sum(1)
        belief = np.einsum("ns,nst->nt", belief, kernels[action]) * emission[:, token].T
        belief /= belief.sum(1, keepdims=True)
    out = {"reward": float(reward.mean() / (EPISODE - WARMUP)),
           "reward_se": float(reward.std(ddof=1) / (EPISODE - WARMUP) / np.sqrt(episodes))}
    if keep:
        out["belief"], out["action"] = np.stack(beliefs, 1), np.stack(actions, 1)
    return out


def lumped_filter_reward(kernels, emission, q, episodes=1024, seed=20261115):
    """Reward of the belief-MDP policy when the filter never distinguishes states 1 and 2.

    The filter tracks only rho = P(state 3) under the two-state model obtained by giving states
    1 and 2 equal weight (averaged transitions and emissions); the policy is the optimal Q at
    the symmetric belief with that rho. Exact for the Reward-State Quotient; elsewhere the gap
    to the optimal reward is what tracking b1 - b2 buys (filtering plus action selection).
    """
    rng = np.random.default_rng(seed)
    cumulative = emission.cumsum(1)
    lumped_emission = np.stack([emission[:2].mean(0), emission[2]])
    to_reward = np.stack([kernels[:, :2, 2].mean(1), kernels[:, 2, 2]], axis=1)  # (action, lumped state)
    state = rng.choice(3, size=episodes, p=v2.INITIAL_STATE_DISTRIBUTION)
    token = (rng.random(episodes)[:, None] > cumulative[state]).sum(1)
    prior = np.array([2 / 3, 1 / 3]) * lumped_emission[:, token].T
    rho = prior[:, 1] / prior.sum(1)
    reward = np.zeros(episodes)
    for t in range(EPISODE):
        action = q(np.column_stack([(1 - rho) / 2, (1 - rho) / 2, rho])).argmax(1)
        if t >= WARMUP:
            reward += state == 2
        state = (rng.random(episodes)[:, None] > kernels[action, state].cumsum(1)).sum(1)
        token = (rng.random(episodes)[:, None] > cumulative[state]).sum(1)
        ahead = (1 - rho) * to_reward[action, 0] + rho * to_reward[action, 1]
        joint = np.column_stack([1 - ahead, ahead]) * lumped_emission[:, token].T
        rho = joint[:, 1] / joint.sum(1)
    return {"reward": float(reward.mean() / (EPISODE - WARMUP)),
            "reward_se": float(reward.std(ddof=1) / (EPISODE - WARMUP) / np.sqrt(episodes))}


def cells(x, classes, bins):
    """Quantile-bin index of x computed separately inside each class."""
    out = np.zeros(len(x), dtype=np.int64)
    for k in np.unique(classes):
        m = classes == k
        edges = np.unique(np.quantile(x[m], np.linspace(0, 1, bins + 1)))
        out[m] = np.clip(np.searchsorted(edges, x[m], side="right") - 1, 0, max(len(edges) - 2, 0))
    return out


def residual_share(y, classes, *codes):
    """Within-class variance share of y left after conditioning on discrete codes."""
    y = np.asarray(y, dtype=float).reshape(len(classes), -1)
    key = np.unique(np.column_stack([classes, *codes]), axis=0, return_inverse=True)[1].reshape(-1)
    group = np.unique(classes, return_inverse=True)[1].reshape(-1)

    def within(index):
        count = np.bincount(index).astype(float)
        total = 0.
        for column in y.T:
            mean = np.bincount(index, column) / count
            total += ((column - mean[index]) ** 2).sum()
        return total
    sst = within(group)
    return float(within(key) / sst) if sst > 1e-20 else None


def analyse(kernels, emission, episodes=1024):
    q = solve(kernels, emission)
    optimal = simulate(kernels, emission, lambda b: q(b).argmax(1), episodes)
    b, a = optimal["belief"].reshape(-1, 3), optimal["action"].reshape(-1)
    values = q(b)
    gaps = values - values.max(1, keepdims=True)
    delta, rho = b[:, 0] - b[:, 1], b[:, 2]

    def quotient_policy(beliefs):
        r = beliefs[:, 2]
        return q(np.column_stack([(1 - r) / 2, (1 - r) / 2, r])).argmax(1)
    quotient = simulate(kernels, emission, quotient_policy, episodes, keep=False)
    lumped = lumped_filter_reward(kernels, emission, q, episodes)
    # future optimal action: does delta predict it beyond class and rho?
    actions = optimal["action"]
    future = {}
    for horizon in (1, 8, 32):
        now, later = actions[:, :-horizon].reshape(-1), actions[:, horizon:].reshape(-1)
        target = np.eye(3)[later]
        d = (optimal["belief"][:, :-horizon, 0] - optimal["belief"][:, :-horizon, 1]).reshape(-1)
        r = optimal["belief"][:, :-horizon, 2].reshape(-1)
        rho_cell = cells(r, now, 20)
        delta_cell = cells(d, now * 100 + rho_cell, 10)
        left_rho = residual_share(target, now, rho_cell)
        left_both = residual_share(target, now, rho_cell, delta_cell)
        future[str(horizon)] = {"left_after_rho": left_rho, "left_after_rho_and_delta": left_both,
                                "delta_increment": None if left_rho is None else left_rho - left_both}
    rho_cell, delta_cell = cells(rho, a, 20), cells(delta, a, 20)
    return {
        "optimal_reward": optimal["reward"], "optimal_reward_se": optimal["reward_se"],
        "quotient_policy_reward": quotient["reward"], "quotient_policy_reward_se": quotient["reward_se"],
        "nuisance_value": optimal["reward"] - quotient["reward"],
        "lumped_filter_reward": lumped["reward"], "lumped_filter_reward_se": lumped["reward_se"],
        "lumped_filter_loss": optimal["reward"] - lumped["reward"],
        "action_fractions": (np.bincount(a, minlength=3) / len(a)).tolist(),
        "within_class_sd": {"delta": float(np.sqrt(sum(((delta[a == k] - delta[a == k].mean()) ** 2).sum() for k in np.unique(a)) / len(a))),
                            "rho": float(np.sqrt(sum(((rho[a == k] - rho[a == k].mean()) ** 2).sum() for k in np.unique(a)) / len(a)))},
        "qgap_within_class_sd": float(np.sqrt(sum(((gaps[a == k] - gaps[a == k].mean(0)) ** 2).sum() for k in np.unique(a)) / len(a))),
        "qgap_share_needing_delta": residual_share(gaps, a, rho_cell),
        "qgap_share_needing_rho": residual_share(gaps, a, delta_cell),
        "future_action": future,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lams", type=float, nargs="*", default=[1, .75, .5, 0, -.5, -1])
    parser.add_argument("--kappas", type=float, nargs="*", default=[.05, .1])
    parser.add_argument("--episodes", type=int, default=1024)
    args = parser.parse_args()
    report = {"trivial": analyse(kernels_for(variant=1), emission_for(), args.episodes), "lam": {}, "kappa": {}}
    print("trivial", json.dumps(report["trivial"]), flush=True)
    for lam in args.lams:
        report["lam"][str(lam)] = analyse(kernels_for(lam), emission_for(), args.episodes)
        print("lam", lam, json.dumps(report["lam"][str(lam)]), flush=True)
    for kappa in args.kappas:
        report["kappa"][str(kappa)] = analyse(kernels_for(1.), emission_for(kappa), args.episodes)
        print("kappa", kappa, json.dumps(report["kappa"][str(kappa)]), flush=True)
    (HERE / "results/theory.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
