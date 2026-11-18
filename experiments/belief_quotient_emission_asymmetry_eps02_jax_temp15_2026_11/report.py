"""Aggregate ``analysis`` reports into one Markdown row per condition.

Cells are the mean [minimum, maximum] over training seeds; the per-report
episode-bootstrap intervals describe evaluation uncertainty only and are not
aggregated here.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _reference(statistic):
    def value(report):
        references = report["behavior"]["references"]
        return references and references[statistic]["expected_reward"]["mean"]

    return value


COLUMNS = {
    "emission defect": lambda r: r["lumpability_defect"]["emission"],
    "transition defect": lambda r: r["lumpability_defect"]["transition"],
    "reward marginal unexplained by quotient":
        lambda r: r["target_separation"]["reward_marginal_unexplained_by_quotient"],
    "quotient R2": lambda r: r["probes"]["quotient"]["scores"]["trained"]["r_squared"],
    "reward-marginal R2":
        lambda r: r["probes"]["reward_marginal"]["scores"]["trained"]["r_squared"],
    "quotient minus reward-marginal R2, change from init":
        lambda r: r["which_target"]["quotient"]["training_change"][
            "quotient_minus_reward_marginal_r_squared"],
    "identity R2, change from init":
        lambda r: r["probes"]["identity"]["trained_minus_initialization"]["delta_r_squared"],
    "quotient-gap R2, change from init":
        lambda r: r["probes"]["quotient_gap"]["trained_minus_initialization"]["delta_r_squared"],
    "action agreement gained from identity":
        lambda r: r["action_rules"]["identity_beyond_quotient"]["difference"],
    "follows reward marginal where rules disagree":
        lambda r: r["action_rules"]["agent_follows_reward_marginal_when_rules_disagree"],
    "agent reward": lambda r: r["behavior"]["agent"]["expected_reward"]["mean"],
    "quotient controller": _reference("quotient"),
    "reward-marginal controller": _reference("reward_marginal"),
}


def _cell(values):
    values = [value for value in values if value is not None]
    if not values:
        return "n/a"
    return f"{np.mean(values):.4f} [{min(values):.4f}, {max(values):.4f}]"


def summarize(paths):
    groups = defaultdict(dict)
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report["smoke"]:
            raise ValueError(f"{path} is a smoke report")
        seeds = groups[report["study"], report["condition"]]
        if report["seed"] in seeds:
            raise ValueError(f"duplicate seed {report['seed']} for {report['condition']}")
        seeds[report["seed"]] = report
    rows = sorted(groups.items(), key=lambda item: (
        max(report["lumpability_defect"]["transition"] for report in item[1].values()),
        max(report["lumpability_defect"]["emission"] for report in item[1].values())))
    header = ["study", "condition", "seeds", *COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for (study, condition), seeds in rows:
        cells = [_cell([column(report) for report in seeds.values()]) for column in COLUMNS.values()]
        lines.append("| " + " | ".join(
            [study, condition, ", ".join(map(str, sorted(seeds))), *cells]) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = summarize(args.reports)
    if args.output:
        with open(args.output, "x") as handle:
            handle.write(table)
    print(table, end="")
