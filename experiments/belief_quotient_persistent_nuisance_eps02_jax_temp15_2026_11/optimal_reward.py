"""Finite-episode belief bounds and exact-evaluation reactive policy search.

Run with --help. Interpolation bounds are not exact Bayes values, and nonlinear
reactive-policy search is not a global optimality certificate.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t

from experiments.belief_quotient_history_jax_2026_11.shared import EPISODE_LENGTH
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment as eps02
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import design as historical
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
from experiments.reward_factors_history_jax_2026_11.process import ACTION_PAIRS

LADDER = "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11"
FACTORS = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11"


def power_gradient(matrix, exponent, left, right):
    """Scalar left @ matrix**exponent @ right and its matrix derivative."""
    nodes = [matrix, np.eye(len(matrix))]
    tape = []
    base, result = 0, 1
    while exponent:
        if exponent & 1:
            tape.append((len(nodes), result, base))
            nodes.append(nodes[result] @ nodes[base])
            result = len(nodes) - 1
        exponent >>= 1
        if exponent:
            tape.append((len(nodes), base, base))
            nodes.append(nodes[base] @ nodes[base])
            base = len(nodes) - 1
    adjoints = [np.zeros_like(matrix) for _ in nodes]
    adjoints[result] = np.outer(left, right)
    for out, a, b in reversed(tape):
        adjoints[a] += adjoints[out] @ nodes[b].T
        adjoints[b] += nodes[a].T @ adjoints[out]
    return float(left @ nodes[result] @ right), adjoints[0]


def reactive_value(policy, kernels, emission, initial, reward, horizon):
    """Exact complete-episode mean and derivative w.r.t. token/action probabilities."""
    transition = np.einsum("so,oa,asj->sj", emission, policy, kernels)
    size = len(initial)
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size], augmented[:size, size] = transition, reward
    augmented[size, size] = 1
    left, right = np.r_[initial, 0.], np.eye(size + 1)[size]
    value, derivative = power_gradient(augmented, horizon, left, right)
    gradient = np.einsum("so,asj,sj->oa", emission, kernels, derivative[:size, :size])
    return value / horizon, gradient / horizon


def reactive_search(kernels, emission, initial, reward, horizon, starts=24, seed=0,
                    extra_policies=()):
    """Simplex-constrained multistart search, including deterministic policies."""
    tokens, actions = emission.shape[1], len(kernels)
    rng = np.random.default_rng(seed)
    policies = list(extra_policies)
    if tokens == actions == 3:
        policies += [np.eye(actions)[list(p)] for p in itertools.product(range(actions), repeat=tokens)]
    policies += [rng.dirichlet(np.ones(actions), tokens) for _ in range(starts)]
    best, best_policy, successes = -np.inf, None, 0
    def objective(x):
        raw = x.reshape(tokens, actions)
        sums = raw.sum(1, keepdims=True)
        normalized = np.divide(raw, sums, out=np.full_like(raw, 1/actions), where=sums > 0)
        value, gradient = reactive_value(normalized, kernels, emission,
                                         initial, reward, horizon)
        gradient = (gradient - (gradient * normalized).sum(1, keepdims=True)) / np.maximum(sums, 1e-100)
        return -value, -gradient.ravel()
    constraint = {"type": "eq", "fun": lambda x: x.reshape(tokens, actions).sum(1) - 1,
                  "jac": lambda x: np.kron(np.eye(tokens), np.ones((1, actions)))}
    for policy in policies:
        baseline = -objective(policy.ravel())[0]
        result = minimize(objective, policy.ravel(), jac=True, method="SLSQP",
                          bounds=[(0., 1.)] * (tokens * actions), constraints=constraint,
                          options={"ftol": 1e-12, "maxiter": 500})
        successes += int(result.success)
        candidate = np.clip(result.x.reshape(tokens, actions), 0, 1)
        candidate /= candidate.sum(1, keepdims=True)
        value = -objective(candidate.ravel())[0]
        if baseline > value:
            value, candidate = baseline, policy
        if value > best:
            best, best_policy = value, candidate
    return {"best_known": best, "policy": best_policy.tolist(), "starts": len(policies),
            "successful_optimizations": successes, "globally_certified": False}


class SimplexGrid:
    def __init__(self, divisions):
        if divisions < 1:
            raise ValueError("grid divisions must be positive")
        self.n = divisions
        self.points = np.array([(i, j, divisions-i-j) for i in range(divisions+1)
                                for j in range(divisions+1-i)], dtype=float) / divisions

    def stencil(self, belief):
        xy = np.clip(np.asarray(belief)[..., :2] * self.n, 0, self.n)
        i = np.minimum(np.floor(xy[..., 0]).astype(int), self.n - 1)
        j = np.minimum(np.floor(xy[..., 1]).astype(int), self.n - i - 1)
        x, y = xy[..., 0] - i, xy[..., 1] - j
        upper = (x + y > 1) & (i + j < self.n - 1)
        def index(a, b):
            return a * (self.n + 1) - a * (a - 1) // 2 + b
        indices = np.stack([index(i, j + upper), index(i + 1, j),
                            index(i + upper, j + 1)], -1)
        weights = np.stack([np.where(upper, 1-x, 1-x-y),
                            np.where(upper, 1-y, x), np.where(upper, x+y-1, y)], -1)
        weights = np.maximum(weights, 0)
        weights /= weights.sum(-1, keepdims=True)
        return indices, weights

    def interpolate(self, values, belief):
        indices, weights = self.stencil(belief)
        return (values[indices] * weights).sum(-1)


def successors(beliefs, kernels, emission):
    predictive = np.einsum("...s,asj->...aj", beliefs, kernels)
    joint = predictive[..., :, None, :] * emission.T
    probabilities = joint.sum(-1)
    posterior = joint / np.maximum(probabilities[..., None], 1e-300)
    return probabilities, posterior


def belief_dp(kernels, emission, horizon, divisions):
    """Convex interpolation gives an upper bound on finite-horizon POMDP value."""
    grid = SimplexGrid(divisions)
    probabilities, posterior = successors(grid.points, kernels, emission)
    indices, weights = grid.stencil(posterior)
    values = np.zeros((horizon + 1, len(grid.points)))
    for remaining in range(1, horizon + 1):
        continuation = (values[remaining-1][indices] * weights).sum(-1)
        values[remaining] = grid.points[:, 2] + (probabilities * continuation).sum(-1).max(-1)
    return grid, values


def belief_reference(kernels, emission, horizon, divisions, episodes, seed):
    grid, values = belief_dp(kernels, emission, horizon, divisions)
    initial_joint = np.full(3, 1/3)[:, None] * emission
    token_probability = initial_joint.sum(0)
    reset_beliefs = (initial_joint / token_probability).T
    upper = token_probability @ grid.interpolate(values[horizon], reset_beliefs) / horizon
    rng = np.random.default_rng(seed)
    tokens = rng.choice(3, episodes, p=token_probability)
    beliefs = reset_beliefs[tokens]
    totals, late_totals = np.zeros(episodes), np.zeros(episodes)
    for step in range(horizon):
        totals += beliefs[:, 2]
        if step >= 64:
            late_totals += beliefs[:, 2]
        probabilities, posterior = successors(beliefs, kernels, emission)
        q = (probabilities * grid.interpolate(values[horizon-step-1], posterior)).sum(-1)
        actions = q.argmax(-1)
        p = probabilities[np.arange(episodes), actions]
        tokens = (rng.random(episodes)[:, None] > p.cumsum(-1)).sum(-1)
        beliefs = posterior[np.arange(episodes), actions, np.minimum(tokens, 2)]
    def estimate(samples):
        mean = float(samples.mean())
        half = float(student_t.ppf(.975, episodes-1) * samples.std(ddof=1) / np.sqrt(episodes))
        return {"mean": mean, "ci95": [mean-half, mean+half]}
    return {"grid_divisions": divisions, "upper_bound": float(upper),
            "feasible_policy": estimate(totals/horizon),
            "post_warmup64_policy": estimate(late_totals/(horizon-64)) if horizon > 64 else None,
            "episodes": episodes, "seed": seed,
            "certification": "upper bound up to floating-point error; CI estimates a feasible policy, not exact optimum"}


def product_policy(first, second):
    return np.stack([np.outer(first[:, a], second[:, b]).ravel()
                     for a, b in ACTION_PAIRS], -1)


def source_comparisons(root):
    records = []
    studies = (LADDER, FACTORS, "belief_quotient_persistent_nuisance_eps02_jax_temp15_cont4b_2026_11")
    for study in studies:
        for path in sorted((root / "experiments" / study).glob("*/results/*/condition_summary.json")):
            summary = json.loads(path.read_text())
            if summary.get("smoke"):
                continue
            final = max(summary["checkpoint_reports"], key=lambda report: report["checkpoint_step"])
            curve_path = path.with_name("training_curves.jsonl")
            curve = json.loads(curve_path.read_text().splitlines()[-1])
            records.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "curves_sha256": hashlib.sha256(curve_path.read_bytes()).hexdigest(),
                            "study": study, "leaf": path.parts[-4], "seed": summary["seed"],
                            "architecture": summary["architecture"], "condition": summary["condition"],
                            "final_steps": final["checkpoint_step"],
                            "train_reward_per_step": curve["reward_per_step"],
                            "summary_train_reward_per_step": summary["training"]["final_reward_per_step"],
                            "greedy_post_warmup64": final["policy"]["reward_per_step"]})
    return records


def sanity_check():
    old = historical.controller_references()
    rewards = []
    for policy in itertools.product(range(3), repeat=3):
        transition = sum(v2.EMISSION_MATRIX[:, token, None] * v2.action_transitions(2)[action]
                         for token, action in enumerate(policy))
        rewards.append(historical.stationary_distribution(transition)[2])
    return {"pr208_v1": old, "v2_variant2_best_deterministic_stationary": float(max(rewards)),
            "v2_variant2_policy": list(itertools.product(range(3), repeat=3))[int(np.argmax(rewards))],
            "note": "PR208 references are v1 feasible controllers, not v2 Bayes optima"}


def build_report(horizon=1024, grids=(80, 160), episodes=4096, starts=24, seed=208):
    tasks, memoryless = {}, {}
    initial, reward = np.full(3, 1/3), np.array([0., 0., 1.])
    for environment_name, environment in (("eps02", eps02), ("v2", v2)):
        for variant in (1, 2, 3):
            kernels, emission = environment.action_transitions(variant), environment.emission_matrix()
            reactive = reactive_search(kernels, emission, initial, reward, horizon, starts, seed)
            references = [belief_reference(kernels, emission, horizon, n, episodes, seed)
                          for n in grids]
            tasks[f"{environment_name}/variant_{variant}"] = {
                "bayes": references, "memoryless": reactive}
            if variant == 1:
                policy = np.eye(3)[[1, 1, 1]]
                exact = reactive_value(policy, kernels, emission, initial, reward, horizon)[0]
                tasks[f"{environment_name}/variant_{variant}"]["exact_bayes_and_memoryless"] = exact
                tasks[f"{environment_name}/variant_{variant}"]["proof"] = (
                    "S1/S2 have identical reward-state transition probabilities. "
                    "Action 1 maximizes P(next S3) in every state; its S3 row "
                    "has greater P(next S3) than its nonreward rows. "
                    "Backward induction on monotone two-state values makes always-1 optimal, "
                    "even with full state information.")
                reactive.update(best_known=exact, policy=policy.tolist(), globally_certified=True)
            if environment_name == "eps02":
                memoryless[variant] = np.array(reactive["policy"])
    kernels = eps02.action_transitions(3)
    joint_kernels = np.stack([np.kron(kernels[a], kernels[b]) for a, b in ACTION_PAIRS])
    joint_emission = np.kron(eps02.emission_matrix(), eps02.emission_matrix())
    first, second = np.divmod(np.arange(9), 3)
    for condition in ("reward_both", "reward_factor_1"):
        joint_reward = (first == 2).astype(float) + ((second == 2) if condition == "reward_both" else 0)
        reactive = reactive_search(joint_kernels, joint_emission, np.full(9, 1/9),
            joint_reward, horizon, starts, seed,
            [product_policy(memoryless[3], memoryless[3])])
        multiplier = 2 if condition == "reward_both" else 1
        tasks[f"eps02/{condition}"] = {
            "bayes_from": "eps02/variant_3", "bayes_multiplier": multiplier,
            "memoryless": reactive,
            "note": "Bayes value is additive by independence/unrestricted full histories and Cartesian actions. Joint reactive search is NOT restricted to product policies."}
    root = Path(__file__).resolve().parents[2]
    achieved = source_comparisons(root)
    for record in achieved:
        if horizon != EPISODE_LENGTH:
            continue
        task = tasks[f"eps02/{record['condition']}"]
        bayes = tasks[task["bayes_from"]] if "bayes_from" in task else task
        multiplier = task.get("bayes_multiplier", 1)
        upper = multiplier * bayes.get("exact_bayes_and_memoryless", bayes["bayes"][-1]["upper_bound"])
        record["bayes_upper_minus_train"] = upper - record["train_reward_per_step"]
        record["train_minus_memoryless_best_known"] = record["train_reward_per_step"] - task["memoryless"]["best_known"]
    return {"schema": 1, "horizon": horizon, "objective": "undiscounted complete-episode expected reward / horizon",
            "conventions": {"reward": "pre-transition S3 occupancy; reward_both SUMS two factors",
                "initial": [1/3]*3, "reset": "emit before first action", "warmup": 0,
                "training_gamma": .99, "benchmark_gamma": 1.,
                "temperature": "1.5 does not constrain distributions; entropy=0, epsilon-greedy=0",
                "memoryless": "stationary stochastic current-token only; no previous action",
                "bayes": "finite horizon, time-dependent full exact token/executed-action posterior, no rewards",
                "comparison": "train rewards include 1024 reset steps; greedy probe rewards exclude first 64; neither has saved episode CIs"
                    + (f"; achieved-vs-reference gaps omitted because benchmark horizon {horizon} differs from training episode length {EPISODE_LENGTH}"
                       if horizon != EPISODE_LENGTH else ""),
                "limitations": "Bayes values bounded/estimated, not exact. Reactive search lower bounds only: no global certificate, especially joint 9-token policies."},
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sanity": sanity_check(), "tasks": tasks, "achieved": achieved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--grids", type=int, nargs="+", default=[80, 160])
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=208)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if args.horizon < 1 or args.episodes < 2 or args.starts < 1:
        parser.error("horizon/starts must be positive and episodes >= 2")
    report = build_report(args.horizon, args.grids, args.episodes, args.starts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output}; {len(report['achieved'])} achieved-run records")


if __name__ == "__main__":
    main()
