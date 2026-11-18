"""Extend the frozen Stage 1 analysis through training seed 56 on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes as probe

import numpy as np

OUTPUT = probe.STUDY / "results/five_site_probes_seeds42_56"
NAMES = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}
SEEDS = list(range(42, 57))
STEPS = (0, 2012217344)
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 100000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage1_hashes():
    return {p.name: digest(p) for p in sorted(probe.OUTPUT.iterdir()) if p.is_file()}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = probe.OUTPUT
    hashes = stage1_hashes()
    paths = sorted(original.glob("variant_*.json"))
    if len(paths) != 18:
        raise ValueError("expected 18 complete Stage 1 reports")
    OUTPUT.mkdir()
    for path in paths:
        shutil.copy2(path, OUTPUT / path.name)
    (OUTPUT / "stage1_snapshot.json").write_text(json.dumps(hashes, indent=2) + "\n")
    base, arch_hash = probe.frozen_module(probe.ARCH_REV, probe.ARCH_PATH)
    core, core_hash = probe.frozen_module(probe.STEER_REV, probe.CORE_PATH)
    try:
        probe.OUTPUT = OUTPUT
        for variant in (1, 2, 3):
            for seed in range(45, 57):
                for index in (0, -1):
                    probe.run_one(base, core, variant, seed, index,
                                  {probe.ARCH_PATH: arch_hash, probe.CORE_PATH: core_hash})
    finally:
        probe.OUTPUT = original
    if stage1_hashes() != hashes:
        raise ValueError("Stage 1 changed")
    print("Stage 2 complete: 72 new CPU checkpoint reports; Stage 1 unchanged.", flush=True)


def bootstrap_mean_ci(values, indices):
    values = np.asarray(values, dtype=float)
    if values.shape != (15,) or not np.isfinite(values).all():
        raise ValueError("expected 15 finite seed measurements")
    replicates = values[indices].mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975], method="linear")
    return {"mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def summarize():
    outputs = [OUTPUT / name for name in ("summary.json", "tables.md", "per_seed.csv",
                                          "reward_state_paired.csv", "hashes.json")]
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
    snapshot = json.loads((OUTPUT / "stage1_snapshot.json").read_text())
    if stage1_hashes() != snapshot:
        raise ValueError("Stage 1 changed")
    paths = sorted(OUTPUT.glob("variant_*.json"))
    reports = [json.loads(p.read_text()) for p in paths]
    expected = {(v, s, t) for v in (1, 2, 3) for s in SEEDS for t in STEPS}
    actual = {(d["variant"], d["seed"], d["checkpoint_step"]) for d in reports}
    if len(reports) != 90 or actual != expected:
        raise ValueError("expected the complete 90-report grid")
    protocols = [d["protocol"] for d in reports]
    if any(p != protocols[0] for p in protocols):
        raise ValueError("inconsistent protocols")
    for path, report in zip(paths, reports):
        if report["seed"] <= 44 and digest(path) != snapshot[path.name]:
            raise ValueError("copied Stage 1 report changed")
        assert report["runtime"]["devices"] == "[CpuDevice(id=0)]"
        assert max(report["replay_final_norm_max_absolute_difference"]) == 0
        assert all(c["absolute_difference"] == 0 for c in report["final_norm_comparison"].values())
        if report["seed"] == 42:
            assert all(c["archived_absolute_difference"] == 0
                       for c in report["final_norm_comparison"].values())
    lookup = {(d["variant"], d["seed"], d["checkpoint_step"]): d for d in reports}
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 15, size=(BOOTSTRAP_REPLICATES, 15))
    aggregates, rows, pairs = [], [], []
    text = ["# Fifteen-seed five-site probes", "",
            "Cells: mean normalized error (1−R²) [95% seed-bootstrap percentile CI].",
            "100,000 resamples of 15 training seeds with replacement; RNG seed 42.",
            "The same seed-index resamples are used across sites, targets, conditions and checkpoints.",
            "Full-State Quotient c is the merged-token control, not an exact quotient.", ""]
    for step in STEPS:
        text += [f"## {'Initialization' if step == 0 else 'Final'}", "",
                 "| Condition | Site | b | c | ρ | δ |", "|---|---|---:|---:|---:|---:|"]
        for variant in (1, 2, 3):
            for site in probe.SITES:
                cells = []
                for target in ("b", "c", "rho", "delta"):
                    values = []
                    for seed in SEEDS:
                        score = lookup[variant, seed, step]["scores"][site][target]
                        assert abs(score["error"] + score["r_squared"] - 1) < 1e-12
                        values.append(score["error"])
                        rows.append(dict(condition=NAMES[variant], seed=seed, step=step,
                                         site=site, target=target, **score))
                    cell = bootstrap_mean_ci(values, indices)
                    aggregates.append(dict(condition=NAMES[variant], step=step, site=site,
                                           target=target, n=15, **cell))
                    cells.append(f"{cell['mean']:.4f} [{cell['ci95_low']:.4f}, {cell['ci95_high']:.4f}]")
                text.append("| " + " | ".join([NAMES[variant], site, *cells]) + " |")
        text.append("")
    text += ["## Reward-State Quotient: is final b error below initialization?", "",
             "Strict inequality, computed separately within each training seed and site.", "",
             "| Seed | Block 1 | Block 2 | Block 3 | Block 4 | Final norm |",
             "|---:|---|---|---|---|---|"]
    for seed in SEEDS:
        cells = []
        for site in probe.SITES:
            init, final = [lookup[2, seed, step]["scores"][site]["b"]["error"] for step in STEPS]
            below = final < init
            pairs.append(dict(seed=seed, site=site, init_b_error=init, final_b_error=final,
                              final_minus_init=final-init, final_b_below_init=below))
            cells.append("Yes" if below else "No")
        text.append("| " + " | ".join([str(seed), *cells]) + " |")
    counts = {site: sum(p["final_b_below_init"] for p in pairs if p["site"] == site) for site in probe.SITES}
    text.append("| **Below-init count / 15** | " + " | ".join(f"**{counts[s]}/15**" for s in probe.SITES) + " |")
    outputs[0].write_text(json.dumps({"bootstrap": {"unit": "training_seed", "method": "percentile",
                            "replicates": BOOTSTRAP_REPLICATES, "rng_seed": BOOTSTRAP_SEED,
                            "quantile_method": "linear", "shared_indices": True, "seeds": SEEDS},
                            "aggregates": aggregates, "reward_state_pairs": pairs,
                            "reward_state_below_init_counts": counts}, indent=2) + "\n")
    outputs[1].write_text("\n".join(text) + "\n")
    for path, data in [(outputs[2], rows), (outputs[3], pairs)]:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
        path.write_text(buffer.getvalue())
    sources = paths + outputs[:4] + [OUTPUT / "stage1_snapshot.json", Path(__file__), Path(probe.__file__)]
    outputs[4].write_text(json.dumps({str(p.relative_to(probe.STUDY)): digest(p)
                                     for p in sources}, indent=2) + "\n")
    print("Validated 90 CPU reports, 1,800 scores, 180 exact final-norm comparisons; Stage 1 unchanged.")
    print(outputs[1].read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize() if args.summarize else run()


if __name__ == "__main__":
    main()
