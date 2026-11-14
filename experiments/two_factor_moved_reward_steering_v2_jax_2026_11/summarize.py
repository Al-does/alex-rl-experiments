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
    "factored_steering_next_step_map": lambda c: c["comparisons"]["myopic_marginal_block_4"],
    "joint_steering_next_step_map": lambda c: c["comparisons"]["myopic_joint_block_4"],
    "factored_steering_long_run_map": lambda c: c["comparisons"]["discounted_marginal_block_4"],
    "joint_steering_long_run_map": lambda c: c["comparisons"]["discounted_joint_block_4"],
    "null_factored_next_step_map": lambda c: c["matched_null"]["myopic_marginal"],
    "null_joint_next_step_map": lambda c: c["matched_null"]["myopic_joint"],
    "null_factored_long_run_map": lambda c: c["matched_null"]["discounted_marginal"],
    "null_joint_long_run_map": lambda c: c["matched_null"]["discounted_joint"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    agents, grouped = {}, {}
    for path in sorted(args.results.glob("ctx32*__seed*.json")):
        report = json.loads(path.read_text())
        rows = {}
        for name, pick in ROWS.items():
            entry = pick(report["confirmation"])
            rows[name] = {"factor_reward": entry["factor_reward"], "factor_reward_se": entry["factor_reward_se"],
                          **({"strength": entry["strength"]} if "strength" in entry else {})}
        agents[path.stem] = {"condition": path.stem.split("__")[1], "rows": rows,
                             "both_factor_probe_r2_block_4": report["probe_r2"]["block_4"],
                             "factor_2_anchor_state_by_action_noop_x_y":
                                 report["factor_2_anchor_state_by_action_noop_x_y"]}
        grouped.setdefault(agents[path.stem]["condition"], []).append(rows)
    means = {condition: {name: np.mean([r[name]["factor_reward"] for r in runs], 0).round(4).tolist()
                         for name in ROWS} for condition, runs in grouped.items()}
    output = args.results / "summary.json"
    output.write_text(json.dumps({"site": "block_4", "agents": agents,
                                  "mean_over_agents_factor_reward": means}, indent=1) + "\n")
    for condition, table in means.items():
        print(f"\n{condition} (n={len(grouped[condition])})")
        for name, (first, second) in table.items():
            print(f"  {name:34s} {first:.3f} {second:.3f}")


if __name__ == "__main__":
    main()
