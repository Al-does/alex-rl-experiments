"""Assemble seed-level comparisons without running models."""

import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 44, 46, 49, 52)
NAMES = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}


def metrics(run):
    result = {f"error_{k}": v["error"] for k, v in run["scalars"].items()}
    for stage in ("trained", "matched_init"):
        for feature, targets in run[stage]["metrics"].items():
            for target, value in targets.items():
                if value["excluded_rows"] and run["variant"] != 1:
                    raise ValueError(f"unrepresented action classes: {run['run_id']}")
                result[f"within_{stage}_{feature}_{target}"] = value["r2"]
    return result


def main():
    runs = [json.loads(p.read_text()) for p in sorted((ROOT / "results/agents").glob("*.json"))]
    if len(runs) != 30 or len({(r['algorithm'], r['variant'], r['seed']) for r in runs}) != 30:
        raise ValueError("expected 30 unique PPO/SAC agents")
    reference = json.loads((ROOT / "results/reinforce_reference.json").read_text())
    groups = {}
    for name, source in (("REINFORCE (15)", reference["runs"]),
                         ("REINFORCE (5)", [r for r in reference["runs"] if r["seed"] in SEEDS]),
                         ("PPO (5)", [r for r in runs if r["algorithm"] == "ppo"]),
                         ("SAC auto-alpha (5)", [r for r in runs if r["algorithm"] == "sac_autoalpha"])):
        for variant in (1, 2, 3):
            selected = sorted([r for r in source if r["variant"] == variant], key=lambda r: r["seed"])
            values = [metrics(r) for r in selected]
            common = set.intersection(*(set(v) for v in values))
            means = {k: float(np.mean([v[k] for v in values])) for k in sorted(common)}
            groups[f"{name}: {NAMES[variant]}"] = dict(algorithm=name, variant=variant, n=len(selected), means=means,
                per_seed={str(r["seed"]): v for r, v in zip(selected, values)})
    summary = {"coverage_exceptions": [{"algorithm": r["algorithm"], "variant": r["variant"], "seed": r["seed"],
        "fit_counts": r["trained"]["fit_counts"], "test_counts": r["trained"]["test_counts"],
        "excluded_rows": r["trained"]["metrics"]["full_h"]["belief"]["excluded_rows"]}
        for r in runs if r["variant"] == 1 and np.count_nonzero(r["trained"]["test_counts"]) > 1],
        "protocol": {"fit_rows": 60000, "test_rows": 80000, "warmup": 64, "ridge": 1e-6,
        "streams": "split(fold_in(PRNGKey(training_seed), 2))", "policy": "greedy", "matmul_precision": "highest",
        "within": "class-specific affine fits; test-class-centered pooled SST; final-policy histories replayed through actual step-0 network",
        "skipped": ["A/B mixing", "frozen shuffled/random histories", "within-class perturbation and label permutation controls", "own-policy init rollouts"],
        "constraints": "If raw global error >1, report support-constrained prediction error, preserving raw score; retrospective screening caveat applies",
        "full_state_coarse": "merged-token state-3 posterior, NOT an exact quotient"},
        "reference_sources": reference["sources"], "groups": groups,
        "constraint_events": [{"run_id": r["run_id"], "target": k, "raw": v["raw"]["error"], "constrained": v["error"]}
                              for r in runs for k, v in r["scalars"].items() if "constrained" in v]}
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    lines = ["# Algorithm generality: eps02 ladder probes", "", "CPU-only, frozen final actors at step 2,012,217,344. Five-seed cohort: 42/44/46/49/52.", "",
        "## Global normalized probe error (1−R²)", "", "| Algorithm | Condition | b | ρ | δ | c[B] / merged target |", "|---|---|---:|---:|---:|---:|"]
    for g in groups.values():
        m = g["means"]; coarse = m.get("error_coarse_b", m.get("error_merged_token_rho")) if g["variant"] != 1 else None
        lines.append(f"| {g['algorithm']} | {NAMES[g['variant']]} | {m['error_belief']:.4f} | {m['error_rho']:.4f} | {m['error_delta']:.4f} | {coarse:.4f} |" if coarse is not None else
                     f"| {g['algorithm']} | {NAMES[g['variant']]} | {m['error_belief']:.4f} | {m['error_rho']:.4f} | {m['error_delta']:.4f} | — |")
    lines += ["", "The last column is an exact quotient only for Reward-State; for Full-State it is the merged-token filter. Each target has its own variance denominator.", "", "## Within-action-class R²: matched init → trained", "", "| Algorithm | Condition | b | ρ | δ |", "|---|---|---:|---:|---:|"]
    for g in groups.values():
        m=g["means"]
        cells=[f"{m[f'within_matched_init_full_h_{t}']:.4f} → {m[f'within_trained_full_h_{t}']:.4f}" for t in ("belief","rho","delta")]
        lines.append(f"| {g['algorithm']} | {NAMES[g['variant']]} | " + " | ".join(cells)+" |")
    lines += ["", "Trivial Quotient is descriptive, not evidence for task-required action regions: most policies use one class. SAC seed 44 has one held-out rare action absent from fit (1/80,000 rows excluded); seed 46 has 4 fit / 11 test rare actions. All Reward-State and Full-State within-class scores have complete coverage.", "", "## Per-seed results", "", "Each triple is global error b/ρ/δ; within-class cells are matched-init → trained R² for b/ρ/δ.", "", "| Algorithm | Condition | Seed | Global errors | Coarse error | Within-class R² |", "|---|---|---:|---|---:|---|"]
    for g in groups.values():
        if g["n"] == 15:
            continue
        for seed,m in g["per_seed"].items():
            e=" / ".join(f"{m['error_'+t]:.4f}" for t in ("belief","rho","delta"))
            c=m.get("error_coarse_b",m.get("error_merged_token_rho")) if g['variant']!=1 else None
            w=" / ".join(f"{m['within_matched_init_full_h_'+t]:.3f}→{m['within_trained_full_h_'+t]:.3f}" for t in ("belief","rho","delta"))
            lines.append(f"| {g['algorithm']} | {NAMES[g['variant']]} | {seed} | {e} | {f'{c:.4f}' if c is not None else '—'} | {w} |")
    lines += ["", "Head-blind and logit-only controls, all per-class scores and excluded-row counts are retained in agent JSONs; group means are in summary.json.", "",
              f"Global constrained-replacement events: {len(summary['constraint_events'])}. Raw global fits are retained in every agent JSON.", ""]
    (ROOT / "results/tables.md").write_text("\n".join(lines))
    buffer=io.StringIO(); writer=csv.writer(buffer, lineterminator="\n");writer.writerow(["algorithm","variant","seed","metric","value"])
    for g in groups.values():
        for seed, values in g["per_seed"].items():
            for key,value in values.items():writer.writerow([g["algorithm"],g["variant"],seed,key,value])
    (ROOT / "results/per_seed.csv").write_text(buffer.getvalue())
    print("Assembled 30 agents, 45 REINFORCE references; summary.json, tables.md, per_seed.csv")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
