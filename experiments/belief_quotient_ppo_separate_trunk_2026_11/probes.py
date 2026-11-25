"""Matched last-layer probes on each checkpoint's greedy policy histories."""
import jax
import numpy as np

from experiments.algorithm_generality_probes_eps02_2026_11.paper_probe import (
    fit_score, flatten, runtime,
)
from experiments.belief_quotient_history_jax_2026_11.probe import probe_budget

from .reference import value_target

_, rollout, replay = runtime()


def belief_targets(belief):
    return {"belief": belief, "rho": belief[..., 2:3],
            "delta": belief[..., :1] - belief[..., 1:2]}


def probe_checkpoint(spec, process, params, key, *, smoke, episode_length,
                     representation, targets, values, divisions, reward_reference):
    budget = probe_budget(smoke)
    counts = budget["train_rows"], budget["test_rows"]
    warmup = budget["warmup"]
    usable = episode_length - warmup
    with jax.default_matmul_precision("highest"):
        records = [
            rollout(spec, process, params, stream, -(-rows // usable), episode_length)
            for stream, rows in zip(jax.random.split(key), counts)
        ]
        features = {("actor" if "critic" in params else "shared"): [
            flatten(r[representation], warmup, n) for r, n in zip(records, counts)
        ]}
        if "critic" in params:
            features["critic"] = [
                flatten(replay(spec, params["critic"], r["obs"], representation), warmup, n)
                for r, n in zip(records, counts)
            ]
    labels = []
    for r, n in zip(records, counts):
        beliefs = np.asarray(r["belief"])
        labels.append({
            **targets(flatten(beliefs, warmup, n)),
            "v_star": flatten(value_target(values, beliefs, divisions), warmup, n),
        })
    test = records[1]
    actions = flatten(test["action"], warmup, counts[1])
    episode_rewards = np.asarray(test["reward"], dtype=float).mean(1)
    reward = float(episode_rewards.mean())
    return {
        **budget, "representation": representation, "policy_mode": "greedy", "probe_ridge": 1e-6,
        "policy": {
            "full_episode_reward_per_step": reward,
            "full_episode_reward_se": float(episode_rewards.std(ddof=1) / np.sqrt(len(episode_rewards))),
            "full_episodes": len(episode_rewards),
            "post_warmup_reward_per_step": float(flatten(test["reward"], warmup, counts[1]).mean()),
            "action_fractions": (np.bincount(actions, minlength=spec.n_actions) / len(actions)).tolist(),
            "gap_to_full_episode_bayes_upper": reward_reference["full_episode_bayes_upper"] - reward,
        },
        "trunks": {
            trunk: {name: fit_score(fit, test, train_target, labels[1][name])[0]
                    for name, train_target in labels[0].items()}
            for trunk, (fit, test) in features.items()
        },
    }
