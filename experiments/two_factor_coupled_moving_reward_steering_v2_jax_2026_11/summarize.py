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
FIELDS = ("total_reward", "total_reward_se", "component_reward", "phase_component_reward",
          "factor_action_agrees_with_exact_belief_map", "factor_action_change_fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(args.results.glob("ctx32*__seed*.json"))}
    out = {"site": "block_4", "agents": sorted(reports), "rewards": {}}
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
        out["rewards"][reward] = {"agents": agents, "mean_over_agents": means}
        print(f"\n{reward} (n={len(agents)}): total | components | phase x component | agree")
        for name, m in means.items():
            print(f"  {name:34s} {m['total_reward']:.3f} {m['component_reward']} "
                  f"{m['phase_component_reward']} {m['factor_action_agrees_with_exact_belief_map']}")
    (args.results / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
