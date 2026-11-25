"""Build the held-out learning figure and extract per-seed/mean endpoints."""
import argparse
import csv
from itertools import product
from pathlib import Path
import json

import numpy as np

from .projection import PROJECTION, TARGETS, load_projection
from .reference import REWARD_REFERENCES
from .seed_queue import LEAVES

SEEDS = (42, 43, 44)
FINAL = 2_012_217_344
TOTAL_CHECKPOINTS = 102
STUDY_DIR = Path(__file__).parent
CONTROLS = (("shared", "shared", "Shared PPO", "tab:blue"),
            ("separate", "actor", "Separate actor", "tab:orange"),
            ("separate", "critic", "Separate critic", "tab:red"),
            ("shared_wide", "shared", "Wide shared PPO", "tab:green"))


def load_runs(root):
    runs = {}
    for leaf in LEAVES:
        by_seed = {}
        for path in sorted((root / leaf / "results").glob("*/run_manifest.json")):
            manifest = json.loads(path.read_text())
            if manifest["status"] != "completed":
                continue
            seed = manifest["runtime"]["seed"]
            if seed not in SEEDS or manifest["runtime"]["smoke"]:
                continue
            projection = path.with_name(PROJECTION)
            summary = (load_projection(projection) if projection.exists()
                       else json.loads(path.with_name("condition_summary.json").read_text()))
            reports = summary["checkpoint_reports"]
            batch = summary["training"]["batch_steps"]
            steps = [row["checkpoint_step"] for row in reports]
            if (seed in by_seed or summary["seed"] != seed or
                    summary["training"]["env_steps"] != FINAL or
                    batch != FINAL // (TOTAL_CHECKPOINTS - 1) or
                    steps != [i * batch for i in range(TOTAL_CHECKPOINTS)]):
                raise ValueError(f"duplicate or incomplete training trajectory: {path}")
            by_seed[seed] = (manifest, summary)
        if set(by_seed) != set(SEEDS):
            raise ValueError(f"{leaf} has only these completed seeds: {sorted(by_seed)}")
        runs[leaf] = by_seed
    return runs


def endpoint_rows(runs):
    for condition, variant in (("trivial", 1), ("full_state", 3)):
        refs = REWARD_REFERENCES[variant]
        ref = refs["full_episode_bayes_upper"]
        attainable = refs["full_episode_feasible"] if variant == 3 else ref
        for arm, trunk, _, _ in CONTROLS:
            rows = []
            for seed in SEEDS:
                manifest, run = runs[f"{condition}_{arm}"][seed]
                end = run["checkpoint_reports"][-1]
                policy = end["policy"]
                errors = end["trunks"][trunk]
                row = dict(
                    sampled=run["training"]["final_reward_per_step"],
                    greedy=policy["full_episode_reward_per_step"],
                    greedy_episode_se=policy["full_episode_reward_se"],
                    attainable=attainable, bayes_upper=ref,
                    gap_to_attainable=attainable - policy["full_episode_reward_per_step"],
                    gap_to_upper=ref - policy["full_episode_reward_per_step"],
                    **{key: errors[key]["error"] for key in TARGETS},
                )
                rows.append(row)
                yield dict(condition=condition, arm=arm, trunk=trunk, seed=seed,
                           run_id=manifest["run_id"], **row)
            mean = {key: float(np.mean([row[key] for row in rows]))
                    for key in rows[0] if key != "greedy_episode_se"}
            yield dict(condition=condition, arm=arm, trunk=trunk, seed="mean",
                       run_id="", greedy_episode_se="", **mean)


def band(values):
    values = np.asarray(values)
    draws = np.array(list(product(range(len(SEEDS)), repeat=len(SEEDS))))
    lo, hi = np.percentile(values[draws].mean(axis=1), [2.5, 97.5], axis=0)
    return values.mean(axis=0), lo, hi


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, panels = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True, layout="constrained")
    for condition, variant, target, title, ax in (
        ("trivial", 1, "rho", "Trivial Quotient", panels[0]),
        ("full_state", 3, "delta", "Full-State Quotient", panels[1]),
    ):
        reward_ax = ax.twinx()
        for arm, trunk, label, color in CONTROLS:
            reports = [runs[f"{condition}_{arm}"][s][1]["checkpoint_reports"] for s in SEEDS]
            steps = [point["checkpoint_step"] / 1e9 for point in reports[0]]
            scores = [[point["trunks"][trunk][target]["error"] for point in r] for r in reports]
            mean, low, high = band(scores)
            if not np.isfinite(scores).all() or np.min(scores) <= 0:
                raise ValueError(f"nonpositive or nonfinite log-plot score: {condition}/{arm}/{trunk}")
            ax.plot(steps, mean, color=color, lw=1.5, label=label)
            ax.fill_between(steps, low, high, color=color, alpha=.12)
            if trunk != "critic":
                rewards = [[point["policy"]["full_episode_reward_per_step"]
                            for point in r] for r in reports]
                mean, low, high = band(rewards)
                reward_ax.plot(steps, mean, "--", color=color, lw=1.1, alpha=.8)
                reward_ax.fill_between(steps, low, high, color=color, alpha=.06)
        refs = REWARD_REFERENCES[variant]
        reward_ax.axhline(refs["full_episode_bayes_upper"], color="0.35", ls=":", lw=1)
        if variant == 3:
            reward_ax.axhline(refs["full_episode_feasible"], color="0.60", ls="-.", lw=1)
        reward_ax.set_ylabel("Complete-episode greedy reward / step")
        ax.set_title(f"{title} ({'ρ' if target == 'rho' else 'δ'} probe)")
        ax.set_xlabel("Training env steps (billions)")
        ax.set_xscale("symlog", linthresh=steps[1])
        ax.set_xticks([0, steps[1], .1, 1, 2], ["0", ".02", ".1", "1", "2"])
        ax.set_xlim(0, steps[-1] * 1.05)
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", alpha=.2)
    panels[0].set_ylabel("Held-out 1 − R²")
    lines, labels = panels[0].get_legend_handles_labels()
    fig.legend(lines, labels, fontsize=9, frameon=False, loc="outside upper center", ncol=4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(STUDY_DIR)
    rows = list(endpoint_rows(runs))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "final_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(runs, args.output / "learning_curves.pdf")


if __name__ == "__main__":
    main()
