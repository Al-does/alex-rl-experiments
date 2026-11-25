"""Aggregate per-seed training curves, training-time probes and post-hoc geometry.

Reads ``<arm>_transformer/results/<arm>-seed<seed>/{condition_summary.json,
training_curves.jsonl}`` and optional ``results/posthoc.json`` and writes
``results/summary.json`` plus a Markdown table ``results/summary.md``.
Uncertainty is the across-seed mean +/- standard error (n = number of seeds)
and paired per-seed differences against ``actor_only``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.algorithms import ARMS

ROOT = Path(__file__).parent
TARGETS = ("rho_1", "rho_2", "delta_1", "delta_2", "factor_1", "factor_2", "joint", "value_star")
REWARD_GATE = 0.95  # fraction of the Bayes-optimal reward per step


def _load(arm, seed):
    run = ROOT / f"{arm}_transformer" / "results" / f"{arm}-seed{seed}"
    summary = json.loads((run / "condition_summary.json").read_text())
    curves = [json.loads(line) for line in (run / "training_curves.jsonl").read_text().splitlines()]
    return summary, curves


def _stats(values):
    values = np.asarray([v for v in values if v is not None], float)
    if len(values) == 0:
        return None
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return {"mean": float(values.mean()), "se": se, "n": int(len(values)), "values": values.tolist()}


def _first_step(curve, key, threshold):
    for row in curve:
        if row[key] >= threshold:
            return row["env_steps"]
    return None


def per_run(arm, seed, bayes):
    summary, curves = _load(arm, seed)
    reports = summary["checkpoint_reports"]
    final = reports[-1]
    tail = curves[-10:]
    gate = REWARD_GATE * bayes
    return {
        "final_train_reward_per_step_last10_mean": float(np.mean([c["reward_per_step"] for c in tail])),
        "final_train_reward_per_step_last10_slope_per_1e8_steps": float(np.polyfit(
            [c["env_steps"] / 1e8 for c in tail], [c["reward_per_step"] for c in tail], 1)[0]),
        "steps_to_train_reward_gate": _first_step(curves, "reward_per_step", gate),
        "final_greedy_probe_reward_per_step": final["policy"]["reward_per_step"],
        "final_greedy_reward_components": final["policy"]["reward_components_per_step"],
        "passes_reward_gate": float(np.mean([c["reward_per_step"] for c in tail])) >= gate,
        "final_probe_r_squared": {t: final["targets"][t]["r_squared"] for t in TARGETS},
        "untrained_probe_r_squared": {t: reports[0]["targets"][t]["r_squared"] for t in TARGETS},
        "probe_trajectory": [{"step": r["checkpoint_step"],
                              "greedy_reward": r["policy"]["reward_per_step"],
                              **{t: r["targets"][t]["r_squared"] for t in ("rho_1", "rho_2", "value_star")}}
                             for r in reports],
        "final_explained_variance": tail[-1].get("explained_variance"),
        "train_seconds": summary["training"].get("train_seconds"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args(argv)
    design = json.loads((ROOT / "results" / "design.json").read_text())
    bayes = design["performance_references_per_step"]["bayes_optimal"]["reward_per_step"]
    runs = {arm: {seed: per_run(arm, seed, bayes) for seed in args.seeds
                  if (ROOT / f"{arm}_transformer" / "results" / f"{arm}-seed{seed}" / "condition_summary.json").exists()}
            for arm in args.arms}
    posthoc_path = ROOT / "results" / "posthoc.json"
    posthoc = json.loads(posthoc_path.read_text()) if posthoc_path.exists() else {}
    aggregate = {}
    for arm, seeds in runs.items():
        entry = {
            "seeds": sorted(seeds),
            "train_reward": _stats([r["final_train_reward_per_step_last10_mean"] for r in seeds.values()]),
            "greedy_reward": _stats([r["final_greedy_probe_reward_per_step"] for r in seeds.values()]),
            "probe_r_squared": {t: _stats([r["final_probe_r_squared"][t] for r in seeds.values()]) for t in TARGETS},
        }
        if "actor_only" in runs and arm != "actor_only":
            paired = [seeds[s]["final_probe_r_squared"]["rho_2"] - runs["actor_only"][s]["final_probe_r_squared"]["rho_2"]
                      for s in seeds if s in runs["actor_only"]]
            entry["paired_rho_2_r_squared_minus_actor_only"] = _stats(paired)
        final_steps = {}
        for name, report in posthoc.items():
            a, seed, step = name.split("/")
            if a == arm:
                final_steps.setdefault(step, {})[seed] = report
        entry["posthoc"] = {}
        for step, by_seed in final_steps.items():
            sampled = [r["sampled"] for r in by_seed.values()]
            block = {
                "seeds": sorted(by_seed),
                "sampled_reward": _stats([r["reward_per_step"] for r in sampled]),
                "factor_2_action_1_fraction": _stats([r["factor_2_action_fractions"][1] for r in sampled]),
                "probes_residual": {t: _stats([r["probes"][t]["residual"] for r in sampled]) for t in sampled[0]["probes"]},
                "probes_context_32_window_filter": {t: _stats([r["probes"][t]["context_32_window_filter"] for r in sampled]) for t in sampled[0]["probes"]},
                "log_policy_affine_r_squared": {k: _stats([r["log_policy_affine_r_squared"][k] for r in sampled]) for k in sampled[0]["log_policy_affine_r_squared"]},
                "steering_tv": {f: {d: _stats([np.mean([r["steering_one_visited_std"][f][f"{d}_plus"]["mean_tv_tempered_policy"],
                                                         r["steering_one_visited_std"][f][f"{d}_minus"]["mean_tv_tempered_policy"]]) for r in sampled])
                                    for d in ("probe", "random", "probe_norm_matched_to_rho_1")} for f in ("rho_1", "rho_2", "delta_1", "delta_2")},
            }
            if "value_head" in sampled[0]:
                block["value_head"] = {k: _stats([r["value_head"][k] for r in sampled])
                                       for k in ("r_squared_vs_value_star", "affine_r_squared_vs_value_star", "affine_r_squared_from_rho_1",
                                                 "affine_r_squared_from_rho_1_rho_2")}
                block["value_head"]["rho_2_over_rho_1_coefficient_ratio"] = _stats(
                    [r["value_head"]["coefficients_rho_1_rho_2"][1] / r["value_head"]["coefficients_rho_1_rho_2"][0] for r in sampled])
            entry["posthoc"][step] = block
        aggregate[arm] = entry
    out = {"bayes_reward_per_step": bayes, "reward_gate": REWARD_GATE * bayes, "runs": runs, "aggregate": aggregate}
    (ROOT / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    fmt = lambda s: "n/a" if s is None else (f"{s['mean']:.3f}" + (f" ± {s['se']:.3f}" if s["se"] is not None else ""))
    lines = ["| arm | seeds | train reward/step | greedy reward/step | " + " | ".join(f"R² {t}" for t in TARGETS) + " |",
             "|" + "---|" * (4 + len(TARGETS))]
    for arm, e in aggregate.items():
        lines.append(f"| {arm} | {len(e['seeds'])} | {fmt(e['train_reward'])} | {fmt(e['greedy_reward'])} | "
                     + " | ".join(fmt(e["probe_r_squared"][t]) for t in TARGETS) + " |")
    lines += ["", "| arm | seed | train reward | steps to gate | slope/1e8 | R² rho_1 | R² rho_2 | R² delta_1 | R² delta_2 | R² V* |",
              "|" + "---|" * 10]
    for arm, seeds in runs.items():
        for seed, r in seeds.items():
            p = r["final_probe_r_squared"]
            lines.append(f"| {arm} | {seed} | {r['final_train_reward_per_step_last10_mean']:.4f} | {r['steps_to_train_reward_gate']} | "
                         f"{r['final_train_reward_per_step_last10_slope_per_1e8_steps']:+.4f} | "
                         + " | ".join(f"{p[t]:.3f}" for t in ("rho_1", "rho_2", "delta_1", "delta_2", "value_star")) + " |")
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
