"""Across-agent means and seed SEs, with the complete per-seed table."""

import argparse
import json
from pathlib import Path

import numpy as np

from .steering import STUDY_DIR, TARGETS


def summarize(reports):
    if not reports:
        raise ValueError("no reports supplied")
    seeds = [r["source"]["seed"] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate agent seed")
    conditions = set(reports[0]["confirmation"])
    if any(set(r["confirmation"]) != conditions for r in reports):
        raise ValueError("inconsistent confirmation conditions")
    fixed = reports[0].get("fixed_joint_strength")
    if any(r.get("fixed_joint_strength") != fixed for r in reports):
        raise ValueError("inconsistent fixed strengths")
    if len({r.get("checkpoint_step") for r in reports}) != 1:
        raise ValueError("inconsistent checkpoint steps")
    per_seed, aggregate = {}, {}
    for r in reports:
        per_seed[str(r["source"]["seed"])] = {
            c: dict(zip(("F1", "F2", "total"),
                        [*r["confirmation"][c]["factor_reward"],
                         r["confirmation"][c]["total_reward"]])) for c in sorted(conditions)}
    for c in sorted(conditions):
        values = np.array([[per_seed[str(s)][c][m] for m in ("F1", "F2", "total")] for s in seeds])
        metrics = ("F1", "F2", "total")
        if fixed is not None:
            denominators = np.array([[per_seed[str(s)]["random"]["F2"],
                                      per_seed[str(s)]["intact"]["F1"]] for s in seeds])
            if np.any(denominators == 0):
                raise ValueError("zero ratio denominator")
            ratios = values[:, [1, 0]] / denominators
            values = np.column_stack((values, ratios))
            metrics += ("F2/random", "F1/intact")
            for i, s in enumerate(seeds):
                per_seed[str(s)][c].update(zip(metrics[3:], ratios[i].tolist()))
        aggregate[c] = {m: {"mean": float(values[:, i].mean()),
                            "seed_se": float(values[:, i].std(ddof=1) / np.sqrt(len(seeds)))
                            if len(seeds) > 1 else None}
                        for i, m in enumerate(metrics)}
    return {"n_agents": len(seeds), "agent_seeds": sorted(seeds),
            **({"reached_random_baseline": {t: sum(r["strength_tuning"]["selected"][t][
                "reached_random_baseline"] for r in reports) for t in TARGETS}}
               if fixed is None else {"fixed_joint_strength": fixed,
                   "checkpoint_step": reports[0]["checkpoint_step"],
                   "ratio_aggregation": "mean and seed SE of per-agent ratios",
                   "joint_block_4_below_random": sum(
                       r["confirmation"]["joint_block_4"]["factor_reward"][1] <
                       r["confirmation"]["random"]["factor_reward"][1] for r in reports)}),
            "across_agent": aggregate, "per_seed": per_seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=STUDY_DIR / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_text()) for p in sorted(
        args.results.glob("eps02_factor2_suppression__seed*.json"))])
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
