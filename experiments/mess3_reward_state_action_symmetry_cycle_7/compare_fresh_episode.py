"""Compare checkpoint_probe_curve.json between cycle-6 (reseeded) and cycle-7
(fresh-episode runner) runs for the same seeds, plus the seed-to-seed spread
reference from seeds 47-56.

Usage: uv run python -m experiments.mess3_reward_state_action_symmetry_cycle_7.compare_fresh_episode --out compare_fresh_episode.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

STUDY = "mess3_reward_state_action_symmetry_cycle_7"
OLD_STUDY = "mess3_reward_state_action_symmetry_cycle_6"
NEW_SEEDS = [42, 43, 44, 45, 46]
REF_SEEDS = list(range(47, 57))
CONDITIONS = ["variant_2_ctx32_ent", "variant_3_ctx32_ent"]

ROOT = Path(__file__).resolve().parent


def load_curve(condition: str, study: str, seed: int):
    pattern = ROOT / condition / "results" / f"{study}-{condition}-seed{seed}-10m" / "checkpoint_probe_curve.json"
    hits = glob.glob(str(pattern))
    if not hits:
        return None
    return json.loads(Path(hits[0]).read_text())


def curve_points(curve):
    return [
        {"agent_steps": c["agent_steps"], "global_mse_ratio": c["global_mse_ratio"], "mse": c["mse"]}
        for c in curve["checkpoints"]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Write JSON report to this path (default: stdout)")
    args = ap.parse_args()

    report = {"conditions": {}}
    for cond in CONDITIONS:
        cond_out = {"per_seed": {}, "reference_seeds_47_56": {}, "spread": {}}
        ref_finals = []
        for seed in REF_SEEDS:
            c = load_curve(cond, OLD_STUDY, seed)
            if c is None:
                continue
            final = c["checkpoints"][-1]["global_mse_ratio"]
            ref_finals.append(final)
            cond_out["reference_seeds_47_56"][seed] = {"final_global_mse_ratio": final}
        for seed in NEW_SEEDS:
            old = load_curve(cond, OLD_STUDY, seed)
            new = load_curve(cond, STUDY, seed)
            entry = {}
            if old:
                entry["cycle6"] = curve_points(old)
                entry["cycle6_final"] = old["checkpoints"][-1]["global_mse_ratio"]
            if new:
                entry["cycle7"] = curve_points(new)
                entry["cycle7_final"] = new["checkpoints"][-1]["global_mse_ratio"]
            if old and new:
                entry["final_delta"] = new["checkpoints"][-1]["global_mse_ratio"] - old["checkpoints"][-1]["global_mse_ratio"]
                entry["final_ratio"] = new["checkpoints"][-1]["global_mse_ratio"] / old["checkpoints"][-1]["global_mse_ratio"]
            cond_out["per_seed"][seed] = entry
        if ref_finals:
            cond_out["spread"] = {
                "mean": statistics.mean(ref_finals),
                "stdev": statistics.stdev(ref_finals) if len(ref_finals) > 1 else 0.0,
                "min": min(ref_finals),
                "max": max(ref_finals),
                "n": len(ref_finals),
            }
        report["conditions"][cond] = cond_out

    # Cross-condition gap summary (Reward-State v2 vs Full-State v3)
    gap = {}
    for seed in NEW_SEEDS:
        e2 = report["conditions"]["variant_2_ctx32_ent"]["per_seed"].get(seed, {})
        e3 = report["conditions"]["variant_3_ctx32_ent"]["per_seed"].get(seed, {})
        row = {}
        if "cycle6_final" in e2 and "cycle6_final" in e3:
            row["cycle6_gap"] = e2["cycle6_final"] / e3["cycle6_final"]
        if "cycle7_final" in e2 and "cycle7_final" in e3:
            row["cycle7_gap"] = e2["cycle7_final"] / e3["cycle7_final"]
        gap[seed] = row
    report["reward_state_vs_full_state_gap"] = gap

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
