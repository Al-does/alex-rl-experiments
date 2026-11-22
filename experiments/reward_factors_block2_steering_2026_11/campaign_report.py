"""Three-seed paired summaries, encoder R² and strict B2/B4 sign counts."""

import json
import math
from pathlib import Path
import statistics

from . import report

LOWER = {"ladder_occupancy", "ladder_mirror", "moving_f1_changed", "full_occupancy"}
METRICS = ("ladder_occupancy", "ladder_mirror", "moving_reward", "moving_gap", "moving_f1_changed",
           "xor_eps02", "xor_v2", "full_occupancy", "full_mirror")


def signs(differences, lower=False):
    differences = list(differences)
    if not differences or not all(math.isfinite(v) for v in differences):
        raise ValueError("finite, nonempty paired differences required")
    favorable = [-v if lower else v for v in differences]
    return dict(wins=sum(v > 0 for v in favorable), ties=sum(v == 0 for v in favorable),
                losses=sum(v < 0 for v in favorable), differences=differences)


def derive(result):
    seeds = result["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique seeds required")
    counts, encoder_counts, r2 = {}, {}, {}
    for method in (*report.METHODS, *report.ENCODERS):
        counts[method] = {}
        for metric in METRICS:
            differences = [result["per_seed"][f"{seed}:1:{method}"][metric] -
                           result["per_seed"][f"{seed}:3:{method}"][metric] for seed in seeds]
            row = signs(differences, lower=metric in LOWER)
            row["direction"] = "nondirectional" if metric == "full_mirror" else "lower" if metric in LOWER else "higher"
            if metric == "full_mirror":
                row = dict(lower=row["losses"], ties=row["ties"], higher=row["wins"],
                           differences=differences, direction="nondirectional")
            counts[method][metric] = row
    for test in report.TESTS:
        for representation in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
            for kind in ("affine", "mlp"):
                for source in ("exact", "decoded"):
                    arm = f"{representation}_{kind}_{source}"
                    paired = []
                    for site in (1, 3):
                        values = [result["encoder_r2"][f"{seed}:{site}:{test}:{arm}"] for seed in seeds]
                        signs(values)
                        r2[f"{site}:{test}:{arm}"] = dict(mean=statistics.mean(values),
                                  sd=statistics.stdev(values) if len(values) > 1 else None)
                        paired.append(values)
                    encoder_counts[f"{test}:{arm}"] = signs(a-b for a, b in zip(*paired))
    return dict(sign_counts=counts, encoder_r2_aggregate=r2, encoder_r2_sign_counts=encoder_counts)


def tables(result):
    lines = report.comparison(result)
    lines += ["", "## Strict B2 wins out of 3 agents", "",
              "Strict comparisons, no tolerance; ties are not wins. Exact W/T/L and raw differences are in the JSON.",
              "Full-State mirror has no better/worse direction: its last column gives B2 lower / tied / higher counts.", "",
              "| Method | Ladder occ ↓ | Mirror ↓ | Moving ↑ | Gap ↑ | F1 changed ↓ | eps02 ↑ | v2 ↑ | Full occ ↓ | Full mirror < / = / > |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, metrics in result["sign_counts"].items():
        cells = [str(metrics[key]["wins"]) for key in METRICS[:-1]]
        cells.append("/".join(str(metrics["full_mirror"][key]) for key in ("lower", "ties", "higher")))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    for heading, field in (("Encoder held-out R²: mean ± sample SD (B2 ; B4)", "encoder_r2_aggregate"),
                           ("Encoder R²: strict B2 wins out of 3", "encoder_r2_sign_counts")):
        lines += ["", f"## {heading}", "", "| Test / coordinates | Affine exact | Affine decoded | MLP exact | MLP decoded |", "|---|---:|---:|---:|---:|"]
        for test in report.TESTS:
            for representation in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
                cells = []
                for kind in ("affine", "mlp"):
                    for source in ("exact", "decoded"):
                        name = f"{test}:{representation}_{kind}_{source}"
                        if field == "encoder_r2_sign_counts":
                            cells.append(str(result[field][name]["wins"]))
                        else:
                            cells.append(" ; ".join(f"{result[field][f'{site}:{name}']['mean']:.4f}±{result[field][f'{site}:{name}']['sd']:.4f}" for site in (1, 3)))
                lines.append(f"| {test} / {representation} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    result = report.aggregate(report.REPO, seeds=(42, 43, 44))
    result.update(derive(result))
    folder = Path(__file__).parent / "results"
    (folder / "campaign_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "campaign_tables.md").write_text(tables(result))
    print(f"Validated {len(result['per_seed'])} method/site/seed rows; {len(result['files'])} compact input files; seeds 42–44")
    print("Aggregate: mean ± sample SD; strict per-seed wins/ties/losses and encoder R² saved")


if __name__ == "__main__":
    main()
