"""Add medians and requested seed-51 sensitivity means from corrected CSV."""
import csv
import json
from pathlib import Path
from statistics import mean, median
import math

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes_stage2 as stage2
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_hashes

OUTPUT = stage2.OUTPUT
TARGETS = ("b", "c", "rho", "delta")


def robust_statistics(rows):
    grouped = {}
    for row in rows:
        key = (row["condition"], int(row["step"]), row["site"], row["target"])
        seed, value = int(row["seed"]), float(row["error"])
        group = grouped.setdefault(key, {})
        if seed in group or not math.isfinite(value):
            raise ValueError("duplicate seed or nonfinite error")
        group[seed] = value
    expected = {(name, step, site, target) for name in stage2.NAMES.values()
                for step in stage2.STEPS for site in stage2.probe.SITES for target in TARGETS}
    if set(grouped) != expected or any(set(g) != set(stage2.SEEDS) for g in grouped.values()):
        raise ValueError("expected complete corrected 15-seed grid")
    medians = {key: median(group.values()) for key, group in grouped.items()}
    sensitivity = []
    for variant in (1, 2):
        key = (stage2.NAMES[variant], 0, "block_1", "c")
        values = [v for seed, v in grouped[key].items() if seed != 51]
        sensitivity.append(dict(condition=key[0], step=0, site="block_1", target="c",
                                excluded_seed=51, n=14, seeds=[s for s in stage2.SEEDS if s != 51],
                                mean=mean(values)))
    return medians, sensitivity


def main():
    summary_path = OUTPUT / "corrected_summary.json"
    summary = json.loads(summary_path.read_text())
    if "sensitivity_excluding_seed51" in summary:
        raise FileExistsError("robust statistics already added")
    manifest_path = OUTPUT / "corrected_hashes.json"
    study = OUTPUT.parent.parent
    five_site_hashes.verify_manifest(manifest_path, study)
    with (OUTPUT / "corrected_per_seed.csv").open() as handle:
        medians, sensitivity = robust_statistics(list(csv.DictReader(handle)))
    for cell in summary["aggregates"]:
        cell["median"] = medians[cell["condition"], cell["step"], cell["site"], cell["target"]]
    summary["sensitivity_excluding_seed51"] = sensitivity
    text = ["", "## Medians across all 15 seeds", "", "Cells are median normalized errors (1−R²).", ""]
    for step in stage2.STEPS:
        text += [f"### {'Initialization' if step == 0 else 'Final'}", "",
                 "| Condition | Site | b | c | ρ | δ |", "|---|---|---:|---:|---:|---:|"]
        for name in stage2.NAMES.values():
            for site in stage2.probe.SITES:
                values = [f"{medians[name, step, site, target]:.4f}" for target in TARGETS]
                text.append("| " + " | ".join([name, site, *values]) + " |")
        text.append("")
    text += ["## Sensitivity: initialization block-1 c, excluding seed 51", "",
             "These 14-seed means do not replace the primary 15-seed means or CIs.", "",
             "| Condition | n | Mean 1−R² |", "|---|---:|---:|"]
    for cell in sensitivity:
        text.append(f"| {cell['condition']} | 14 | {cell['mean']:.6f} |")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table_path = OUTPUT / "corrected_tables.md"
    table_path.write_text(table_path.read_text() + "\n".join(text) + "\n")
    five_site_hashes.write_manifest(manifest_path, (summary_path, table_path, Path(__file__)), study)
    print("Added 120 medians and two 14-seed sensitivity means; primary means/CIs and per-seed data unchanged.")
    print("\n".join(text))


if __name__ == "__main__":
    main()
