"""Final greedy reward of every sweep arm against the belief-grid reference (reward_reference.py).

Each final policy is rolled out greedily for 4,096 fresh 1,024-step episodes (64-step warmup
excluded), the same protocol as the reference. All arms share one rollout key, so arm-vs-arm
differences use common random numbers and a paired episode bootstrap.
"""
import json
import sys
from pathlib import Path

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout
from experiments.within_class_belief_probes_2026_11.temp15 import restore
from .analyze import HERE, RAW

REFERENCE = HERE / "results/reward_reference.json"
TRAJECTORY = HERE / "results/within_class_exploration_sweep_trajectory.json"
OUTPUT = HERE / "results/reward_gate.json"
EPISODES, CHUNK, EPISODE, WARMUP = 4096, 512, 1024, 64
CHECKPOINT_EPISODES = 1024
GATE = 0.95
# Checkpoints whose 84-episode probe-stream reward is below this fraction of the reference
# (>5 SE under the gate) fail without re-evaluation.
SCREEN = 0.80


def episode_rewards(arm, step=None, episodes=EPISODES):
    run = Path(arm["run_dir"])
    run = run if run.is_absolute() else Path.cwd() / run
    step = arm["final_reference_step"] if step is None else step
    cache = RAW / arm["study"] / run.parent.parent.name / f"{step}_reward_{episodes}.npy"
    if cache.exists():
        return np.load(cache)
    spec, process, params, _ = restore(run, step, raw_root=RAW / arm["study"])
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42), 3), episodes // CHUNK)
    out = np.concatenate([np.asarray(_probe_rollout(spec, process, params, k, CHUNK, EPISODE)["reward"])
                          [:, WARMUP:].mean(1) for k in keys])
    np.save(cache, out)
    return out


def interval(x, n=2000, seed=20261115):
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return np.quantile(x[idx].mean(1), [.025, .975]).tolist()


def main():
    reference = json.loads(REFERENCE.read_text())["variants"]
    trajectory = json.loads(TRAJECTORY.read_text())
    arms, rewards = [], {}
    only = [int(v) for v in sys.argv[1:]]
    for arm in trajectory["arms"]:
        if only and arm["variant"] not in only:
            continue
        key = (arm["variant"], arm["training_epsilon"], arm["training_entropy"])
        rewards[key] = episode_rewards(arm)
        ref = reference[str(arm["variant"])]["best_belief_grid"]
        x = rewards[key]
        ratio = x.mean() / ref["mean"]
        ref_sd = ref["episode_sd"] / np.sqrt(ref["episodes"])
        boot = np.random.default_rng(1).integers(0, len(x), (2000, len(x)))
        ratios = x[boot].mean(1) / (ref["mean"] + ref_sd * np.random.default_rng(2).standard_normal(2000))
        checkpoints = []
        for p in arm["points"]:
            screen = p["greedy_reward"]["mean"] / ref["mean"]
            point = {"step": p["step"], "greedy_reward_84ep": p["greedy_reward"]["mean"],
                     "screen_fraction": screen,
                     "fixed_history_full_h_r2": p["fixed_history"]["metrics"]["full_h"]["belief"]["r2"],
                     "own_history_full_h_r2": p["own_history"]["metrics"]["full_h"]["belief"]["r2"]}
            if screen >= SCREEN:
                y = episode_rewards(arm, p["step"], CHECKPOINT_EPISODES)
                point.update(reward_mean=float(y.mean()), reward_ci95=interval(y),
                             episodes=len(y), fraction_of_reference=float(y.mean() / ref["mean"]))
            else:
                point.update(fraction_of_reference=screen, episodes=84)
            point["passes_gate"] = bool(point["fraction_of_reference"] >= GATE)
            checkpoints.append(point)
        passing = [c for c in checkpoints if c["passes_gate"]]
        final_r2 = checkpoints[-1]["fixed_history_full_h_r2"]
        gated_peak = max(passing, key=lambda c: c["fixed_history_full_h_r2"]) if passing else None
        arms.append({"variant": arm["variant"], "training_epsilon": arm["training_epsilon"],
                     "training_entropy": arm["training_entropy"], "run_id": arm["run_id"],
                     "final_step": arm["final_reference_step"], "episodes": len(x),
                     "reward_mean": float(x.mean()), "reward_ci95": interval(x),
                     "reference_mean": ref["mean"], "fraction_of_reference": float(ratio),
                     "fraction_ci95": np.quantile(ratios, [.025, .975]).tolist(),
                     "passes_gate": bool(ratio >= GATE), "checkpoints": checkpoints,
                     "first_gated_step": passing[0]["step"] if passing else None,
                     "gated_peak": None if gated_peak is None else {
                         "step": gated_peak["step"], "fixed_history_full_h_r2": gated_peak["fixed_history_full_h_r2"],
                         "minus_final": gated_peak["fixed_history_full_h_r2"] - final_r2}})
    for arm in arms:
        peers = [a for a in arms if a["variant"] == arm["variant"]]
        best = max(peers, key=lambda a: a["reward_mean"])
        x = rewards[(arm["variant"], arm["training_epsilon"], arm["training_entropy"])]
        y = rewards[(best["variant"], best["training_epsilon"], best["training_entropy"])]
        arm["minus_best_arm"] = {"best": [best["training_epsilon"], best["training_entropy"]],
                                 "mean": float((x - y).mean()), "ci95": interval(x - y)}
    report = {"protocol": {"episodes": EPISODES, "episode_length": EPISODE, "warmup": WARMUP,
                           "policy": "greedy (argmax logits), as in the probe rollouts",
                           "gate": f"final greedy reward >= {GATE:.0%} of the belief-grid reference",
                           "rollout_key": "fold_in(PRNGKey(42), 3), shared across arms (common random numbers)",
                           "checkpoint_rewards": (f"checkpoints screened with 84-episode probe-stream rewards; those >= {SCREEN:.0%} of "
                                                  f"reference re-evaluated with {CHECKPOINT_EPISODES} fresh episodes (same key)"),
                           "gated_peak": "max fixed-history full-h R² over gate-passing sampled checkpoints; exploratory (selected after inspection)"},
              "arms": arms}
    if not only:
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    for a in arms:
        print(a["variant"], a["training_epsilon"], a["training_entropy"], round(a["reward_mean"], 4),
              [round(v, 4) for v in a["reward_ci95"]], round(a["fraction_of_reference"], 3),
              round(a["minus_best_arm"]["mean"], 4), [round(v, 4) for v in a["minus_best_arm"]["ci95"]],
              a["first_gated_step"], a["gated_peak"])


if __name__ == "__main__":
    main()
