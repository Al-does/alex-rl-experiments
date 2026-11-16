"""Compact cross-agent table of block-4 confirmation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = {
    "unsteered": lambda c: c["intact"],
    "uniform_random": lambda c: c["random"],
    "exact_belief_next_step_map": lambda c: c["exact_belief_myopic_map"],
    "exact_belief_long_run_map": lambda c: c["exact_belief_discounted_map"],
    **{f"{t}_steering_{m}_map": (lambda c, k=f"{km}_{kt}_block_4": c["comparisons"][k])
       for km, m in (("myopic", "next_step"), ("discounted", "long_run"))
       for kt, t in (("marginal", "factored"), ("joint", "joint"))},
    **{f"null_{t}_{m}_map": (lambda c, k=f"{km}_{kt}": c["matched_null"][k])
       for km, m in (("myopic", "next_step"), ("discounted", "long_run"))
       for kt, t in (("marginal", "factored"), ("joint", "joint"))},
}
SETUP_ROWS = ("unsteered", "uniform_random", "exact_belief_long_run_map",
              "factored_steering_long_run_map", "joint_steering_long_run_map",
              "null_factored_long_run_map", "null_joint_long_run_map")


def setup_of(stem: str) -> str:
    return stem.split("__")[0]


def final_probe_error(report: dict) -> dict:
    summary = json.loads((Path(report["source"]["results"]) / "condition_summary.json").read_text())
    final = summary["checkpoint_reports"][-1]
    return {"checkpoint_step": final["checkpoint_step"],
            **{f"{k}_1_minus_r2": round(1 - v["r_squared"], 4) for k, v in final["targets"].items()}}


def by_setup(agents: dict, errors: dict) -> dict:
    out = {}
    for setup in sorted({setup_of(s) for s in agents}):
        stems = [s for s in agents if setup_of(s) == setup]
        rows = {}
        for name in SETUP_ROWS:
            values = np.array([agents[s][name]["total_reward"] for s in stems])
            components = np.array([agents[s][name]["component_reward"] for s in stems])
            rows[name] = {"total_reward_mean": round(float(values.mean()), 4),
                          "total_reward_sd": round(float(values.std(ddof=1)), 4) if len(stems) > 1 else None,
                          "total_reward_min": round(float(values.min()), 4),
                          "total_reward_max": round(float(values.max()), 4),
                          "component_reward_mean": components.mean(0).round(4).tolist()}
        gaps = {}
        unsteered = np.array([agents[s]["unsteered"]["total_reward"] for s in stems])
        exact = np.array([agents[s]["exact_belief_long_run_map"]["total_reward"] for s in stems])
        joint_error = np.array([errors[s]["joint_1_minus_r2"] for s in stems])
        for kind in ("factored", "joint"):
            steered = np.array([agents[s][f"{kind}_steering_long_run_map"]["total_reward"] for s in stems])
            gain = steered - unsteered
            closed = gain / (exact - unsteered)
            gaps[kind] = {"gain_mean": round(float(gain.mean()), 4),
                          "gain_sd": round(float(gain.std(ddof=1)), 4) if len(stems) > 1 else None,
                          "agents_with_positive_gain": int((gain > 0).sum()),
                          "gap_closed_mean": round(float(closed.mean()), 4),
                          "gap_closed_sd": round(float(closed.std(ddof=1)), 4) if len(stems) > 1 else None}
            if len(stems) > 2:
                gaps[kind]["pearson_r_joint_1_minus_r2_vs_steered_reward"] = round(
                    float(np.corrcoef(joint_error, steered)[0, 1]), 4)
                ranks = lambda x: np.argsort(np.argsort(x))
                gaps[kind]["spearman_r_joint_1_minus_r2_vs_steered_reward"] = round(
                    float(np.corrcoef(ranks(joint_error), ranks(steered))[0, 1]), 4)
        out[setup] = {"n_agents": len(stems), "agents": sorted(stems), "rows": rows,
                      "long_run_steering_vs_unsteered": gaps}
    return out


FIELDS = ("total_reward", "total_reward_se", "component_reward", "phase_component_reward",
          "factor_action_agrees_with_exact_belief_map", "factor_action_change_fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(args.results.glob("ctx32*__seed*.json"))}
    out = {"site": "block_4", "agents": sorted(reports), "rewards": {}}
    errors = {stem: final_probe_error(r) for stem, r in reports.items()}
    out["final_checkpoint_probe_error"] = errors
    for reward in ("xor", "moving"):
        agents = {}
        for stem, report in reports.items():
            confirmation = report["rewards"][reward]["confirmation"]
            agents[stem] = {name: {f: pick(confirmation)[f] for f in FIELDS}
                            | ({"strength": pick(confirmation)["strength"]}
                               if "strength" in pick(confirmation) else {})
                            for name, pick in ROWS.items()}
        means = {name: {f: np.mean([a[name][f] for a in agents.values()], 0).round(4).tolist()
                        for f in ("total_reward", "component_reward", "phase_component_reward",
                                  "factor_action_agrees_with_exact_belief_map")}
                 for name in ROWS}
        out["rewards"][reward] = {"agents": agents, "mean_over_agents": means,
                                  "by_setup": by_setup(agents, errors)}
        print(f"\n{reward} (n={len(agents)}): total | components | phase x component | agree")
        for name, m in means.items():
            print(f"  {name:34s} {m['total_reward']:.3f} {m['component_reward']} "
                  f"{m['phase_component_reward']} {m['factor_action_agrees_with_exact_belief_map']}")
    (args.results / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
