"""Model-based design diagnostics: how far each process is from exact lumpability,
what acting on the approximate quotient costs, and how far the probe targets separate.

Controllers are simulated in float64 NumPy with exact filters. Each plays Tilt X
while its statistic is below a threshold and Noop otherwise; the statistic is
either the equal-weight quotient filter or the reward marginal of the full
belief. These are feasible reference controllers, not Bayes-optimality bounds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11 import environment

STATISTICS = ("quotient", "reward_marginal")
THRESHOLDS = tuple(np.round(np.arange(0.06, 0.31, 0.02), 2).tolist())
HORIZON, WARMUP = 1024, 128
EPISODES, SMOKE_EPISODES = 8192, 64
NOOP, TILT_X = 0, 1


def _sample(rng, cumulative):
    draws = rng.random(len(cumulative))[:, None]
    return np.minimum((draws > cumulative).sum(1), cumulative.shape[1] - 1)


def episode_estimate(values):
    return {"mean": float(values.mean()),
            "episode_mc_se": float(values.std(ddof=1) / np.sqrt(len(values)))}


def simulate(full, statistic, threshold, *, episodes, seed, horizon=HORIZON, warmup=WARMUP):
    """Per-episode outcomes and post-warmup filter rows of one threshold controller.

    Expected reward averages the exact reward-state posterior, which has the
    realized reward's mean under any history-measurable policy.
    """

    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}")
    rng = np.random.default_rng(seed)
    quotient = environment.quotient_model(full)
    transitions, emission = np.cumsum(full["transitions"], -1), np.cumsum(full["emission"], -1)
    state = _sample(rng, np.broadcast_to(np.cumsum(full["initial"]), (episodes, 3)))
    belief = np.broadcast_to(full["initial"], (episodes, 3))
    coarse = np.broadcast_to(quotient["initial"], (episodes, 2))
    rows = {name: np.empty((episodes, horizon)) for name in ("rho", "quotient", "identity")}
    reward, actions = np.empty((episodes, horizon)), np.empty((episodes, horizon), dtype=int)
    for t in range(horizon):
        token = _sample(rng, emission[state])
        belief = belief * full["emission"][:, token].T
        belief = belief / belief.sum(1, keepdims=True)
        coarse = coarse * quotient["emission"][:, (token == 2).astype(int)].T
        coarse = coarse / coarse.sum(1, keepdims=True)
        rows["rho"][:, t], rows["quotient"][:, t] = belief[:, 2], coarse[:, 1]
        rows["identity"][:, t] = belief[:, 0] - belief[:, 1]
        value = coarse[:, 1] if statistic == "quotient" else belief[:, 2]
        action = np.where(value < threshold, TILT_X, NOOP)
        reward[:, t], actions[:, t] = state == 2, action
        state = _sample(rng, transitions[action, state])
        belief = np.einsum("bi,bij->bj", belief, full["transitions"][action])
        coarse = np.einsum("bi,bij->bj", coarse, quotient["transitions"][action])
    return {
        "expected_reward": rows["rho"].mean(1),
        "post_warmup_expected_reward": rows["rho"][:, warmup:].mean(1),
        "realized_reward": reward.mean(1),
        "tilt_x_fraction": float((actions == TILT_X).mean()),
        "rows": {name: value[:, warmup:].ravel() for name, value in rows.items()},
    }


def target_separation(rows):
    """Post-warmup variances and the share of reward-marginal variance that an
    affine function of the quotient filter does not explain."""

    rho, quotient = rows["rho"], rows["quotient"]
    return {
        "reward_marginal_variance": float(rho.var()),
        "quotient_variance": float(quotient.var()),
        "identity_variance": float(rows["identity"].var()),
        "gap_mean_square": float(np.square(rho - quotient).mean()),
        "reward_marginal_unexplained_by_quotient": float(1 - np.corrcoef(rho, quotient)[0, 1] ** 2),
    }


def reference_controllers(full, *, episodes=EPISODES, seed=0, thresholds=THRESHOLDS,
                          horizon=HORIZON, warmup=WARMUP):
    """Select each controller's threshold on seed ``seed`` and confirm both on
    ``seed + 1``; the regret pairs confirmation episodes that share that seed."""

    report = {"thresholds": list(thresholds), "episodes": episodes, "horizon": horizon,
              "warmup": warmup, "selection_seed": seed, "confirmation_seed": seed + 1}
    confirmed = {}
    for statistic in STATISTICS:
        scan = [float(simulate(full, statistic, threshold, episodes=episodes, seed=seed,
                               horizon=horizon, warmup=warmup)["expected_reward"].mean())
                for threshold in thresholds]
        selected = thresholds[int(np.argmax(scan))]
        confirmed[statistic] = simulate(full, statistic, selected, episodes=episodes,
                                        seed=seed + 1, horizon=horizon, warmup=warmup)
        report[statistic] = {
            "selection_expected_reward": scan, "selected_threshold": selected,
            "tilt_x_fraction": confirmed[statistic]["tilt_x_fraction"],
            **{name: episode_estimate(confirmed[statistic][name]) for name in (
                "expected_reward", "post_warmup_expected_reward", "realized_reward")},
        }
    regret = (confirmed["reward_marginal"]["expected_reward"]
              - confirmed["quotient"]["expected_reward"])
    report["quotient_regret"] = {
        **episode_estimate(regret),
        "relative": float(regret.mean() / confirmed["reward_marginal"]["expected_reward"].mean()),
    }
    report["target_separation"] = target_separation(confirmed["reward_marginal"]["rows"])
    return report


def design_summary(*, episodes=EPISODES, seed=0):
    levels = {}
    for level, kappa in {"000": 0.0, **environment.LEVELS}.items():
        full = environment.model(kappa)
        levels[level] = {
            "kappa": kappa,
            "not_c_rate_states_1_2_3": full["emission"][:, :2].sum(1).tolist(),
            "lumpability_defect": environment.lumpability_defect(full),
            **reference_controllers(full, episodes=episodes, seed=seed),
        }
    return {"source_study": environment.SOURCE_STUDY, "variant": environment.VARIANT,
            "controllers": "Tilt X below the threshold, Noop otherwise; simulated references, "
                           "not Bayes-optimality bounds",
            "levels": levels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(design_summary(episodes=SMOKE_EPISODES if args.smoke else EPISODES,
                                     seed=args.seed), indent=2) + "\n"
    if args.output:
        with open(args.output, "x") as handle:
            handle.write(text)
    print(text, end="")
