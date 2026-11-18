"""Validate and summarize the three-seed, two-checkpoint five-site reports."""
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean

STUDY = Path(__file__).resolve().parent
OUTPUT = STUDY / "results/five_site_probes_stage1_seeds42_44"
NAMES = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}


def main():
    outputs = [OUTPUT / name for name in ("summary.json", "tables.md", "per_seed.csv", "hashes.json")]
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
    paths = sorted(OUTPUT.glob("variant_*.json"))
    reports = [json.loads(path.read_text()) for path in paths]
    expected = {(v, s, t) for v in (1, 2, 3) for s in (42, 43, 44) for t in (0, 2012217344)}
    assert {(d["variant"], d["seed"], d["checkpoint_step"]) for d in reports} == expected
    assert len(reports) == 18
    assert all(max(d["replay_final_norm_max_absolute_difference"]) == 0 for d in reports)
    rows, comparisons, aggregates = [], [], []
    for d in reports:
        for site, scores in d["scores"].items():
            for target, score in scores.items():
                assert abs(score["error"] + score["r_squared"] - 1) < 1e-12
                rows.append(dict(condition=NAMES[d["variant"]], seed=d["seed"],
                                 step=d["checkpoint_step"], site=site, target=target, **score))
        for target, comparison in d["final_norm_comparison"].items():
            assert comparison["absolute_difference"] == 0
            if d["seed"] == 42:
                assert comparison["archived_absolute_difference"] == 0
            comparisons.append(dict(condition=NAMES[d["variant"]], seed=d["seed"],
                                    step=d["checkpoint_step"], target=target, **comparison))
    text = ["# Stage 1 five-site probes", "", "Cells are mean normalized error / mean R² across training seeds 42–44.",
            "Full-State Quotient c is the merged-token control, not an exact quotient.", ""]
    for step in (0, 2012217344):
        text += [f"## {'Initialization' if step == 0 else 'Final'} (step {step})", "",
                 "| Condition | Site | b | c | ρ | δ |", "|---|---|---:|---:|---:|---:|"]
        for variant in (1, 2, 3):
            group = [d for d in reports if d["variant"] == variant and d["checkpoint_step"] == step]
            for site in group[0]["scores"]:
                cells = []
                for target in ("b", "c", "rho", "delta"):
                    values = [d["scores"][site][target]["error"] for d in group]
                    avg = mean(values)
                    aggregates.append(dict(condition=NAMES[variant], step=step, site=site, target=target,
                                           mean_error=avg, mean_r_squared=1-avg,
                                           min_error=min(values), max_error=max(values), seeds=[42, 43, 44]))
                    cells.append(f"{avg:.4f} / {1-avg:.4f}")
                text.append("| " + " | ".join([NAMES[variant], site, *cells]) + " |")
        text.append("")
    text += ["## Final-normalization validation", "",
             "Each row gives five-site b/c errors; the original final-norm-only reproduction agrees exactly.",
             "For seed 42 the archived reports also agree exactly (both checkpoints, all conditions).", "",
             "| Condition | Seed | Init b | Init c | Final b | Final c |", "|---|---:|---:|---:|---:|---:|"]
    for variant in (1, 2, 3):
        for seed in (42, 43, 44):
            pair = sorted([d for d in reports if d["variant"] == variant and d["seed"] == seed],
                          key=lambda d: d["checkpoint_step"])
            values = [d["scores"]["post_final_norm"][target]["error"] for d in pair for target in ("b", "c")]
            text.append("| " + " | ".join([NAMES[variant], str(seed), *[f"{x:.6f}" for x in values]]) + " |")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    outputs[0].write_text(json.dumps({"aggregates": aggregates, "final_norm_comparisons": comparisons}, indent=2) + "\n")
    outputs[1].write_text("\n".join(text) + "\n")
    outputs[2].write_text(buffer.getvalue())
    sources = paths + outputs[:3] + [STUDY / "five_site_probes.py", Path(__file__)]
    outputs[3].write_text(json.dumps({str(p.relative_to(STUDY)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sources}, indent=2) + "\n")
    print("Validated 18 reports, 360 site/target scores, 36 final-norm comparisons; all exact.")
    print(outputs[1].read_text())


if __name__ == "__main__":
    main()
