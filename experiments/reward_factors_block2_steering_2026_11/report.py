"""Validate seed-42 pilot reports and compare with saved block-4 controls."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .run import METHODS, TESTS

REPO = Path(__file__).resolve().parents[2]
ENCODERS = tuple(f"{r}_{k}_{s}" for r in ("marginal", "joint") for k in ("affine", "mlp") for s in ("exact", "decoded"))


def source_hashes(method):
    if method in ENCODERS:
        files = ("reward_factors_block2_steering_2026_11/encoder_run.py", "belief_quotient_encoder_nonlinearity_2026_11/encoders.py")
    else:
        files = ("reward_factors_block2_steering_2026_11/run.py", "belief_quotient_block2_steering_2026_11/continuation.py",
                 "reward_factors_block2_steering_2026_11/das.py")
    return {Path(name).name: hashlib.sha256((REPO / "experiments" / name).read_bytes()).hexdigest() for name in files}


def path(root, test, seed, method, site):
    prefix = "belief_quotient" if test.startswith("ladder") else "reward_factors"
    return root / f"experiments/{prefix}_block2_steering_2026_11/results/pilot_site{site}_{test}_seed{seed}_{method}.json"


def load(root, test, seed, method, site):
    file = path(root, test, seed, method, site)
    archived = False
    if not file.exists() and site == 3:
        encoder = method in ENCODERS
        prefix = "belief_quotient" if test.startswith("ladder") else "reward_factors"
        study = "encoder_nonlinearity" if encoder else "causal_decoding"
        file = root / f"experiments/{prefix}_{study}_2026_11/results/campaign_{test}_seed{seed}_{method}.json"
        archived = True
        if test.startswith("ladder") and method == "map":
            raise FileNotFoundError("fresh B4 ladder grid reference required; historical 'map' was intact")
    row = json.loads(file.read_text())
    expected = dict(test=test, seed=seed, smoke=False, tuning_episodes=16, confirmation_episodes=32, horizon=1024)
    if not archived:
        expected.update(site=site, positions="latest_every_step", action_change_reference="intact_same_executed_history")
    for key, value in expected.items():
        if row.get(key, False if key == "smoke" else None) != value:
            raise ValueError(f"{file}: expected {key}={value}, got {row.get(key)}")
    if not archived and row.get("code_hashes") != source_hashes(method):
        raise ValueError(f"{file}: implementation hash mismatch")
    return row, file, archived


def measures(rows, method):
    ladder, full = (rows[test, method]["metrics"] for test in ("ladder", "ladder_full"))
    moving = rows["moving", method]["metrics"]["0"]
    intact, reference = (rows["moving", name]["metrics"]["0"]["reward"] for name in ("intact", "map"))
    if reference <= intact:
        raise ValueError("nonpositive paired map/intact moving gap")
    return dict(ladder_occupancy=ladder["2"]["reward"], ladder_mirror=ladder["1"]["actions_changed"],
                moving_reward=moving["reward"], moving_gap=(moving["reward"]-intact)/(reference-intact),
                moving_f1_changed=moving["actions_changed"],
                xor_eps02=rows["xor_eps02", method]["metrics"]["0"]["reward"],
                xor_v2=rows["xor_v2", method]["metrics"]["0"]["reward"],
                full_occupancy=full["2"]["reward"], full_mirror=full["1"]["actions_changed"])


def aggregate(root, seeds=(42,)):
    manifest, data, summary, checkpoints = {}, {}, {}, {}
    encoder_r2, donor_tv, controls = {}, {}, {}
    sources = set()
    methods = (*METHODS, *ENCODERS)
    for seed in seeds:
        for site in (1, 3):
            rows = {}
            for test in TESTS:
                for method in methods:
                    actual = method.replace("joint_", "marginal_", 1) if test.startswith("ladder") and method.startswith("joint_") else method
                    row, file, archived = load(root, test, seed, actual, site)
                    checkpoint = row["checkpoint"]["sha256"]
                    reference = checkpoints.setdefault(f"{seed}:{test}", checkpoint)
                    if checkpoint != reference:
                        raise ValueError(f"{file}: checkpoint mismatch across methods/sites")
                    rows[test, method] = row
                    manifest[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
                    sources.add("saved_B4" if archived else f"fresh_site{site}")
                    identifier = f"{seed}:{site}:{test}:{actual}"
                    if actual in ENCODERS:
                        encoder_r2[identifier] = row["diagnostics"]["test_r2"]
                    if actual.endswith("_full") and site == 1:
                        donor_tv[identifier] = {
                            mode: {key: metric[key] for key in ("strength", "donor_coverage", "donor_tv_unit", "donor_tv_selected")}
                            for mode, metric in row["metrics"].items()}
            for test in TESTS:
                for representation in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
                    original = rows[test, "eq4" if representation == "marginal" else "eq4_joint"]["metrics"]
                    encoder = rows[test, f"{representation}_affine_exact"]["metrics"]
                    differences = [dict(mode=mode, field=key, eq4=metric[key], affine_exact=encoder[mode][key])
                                   for mode, metric in original.items()
                                   for key in ("strength", "reward", "factor1_reward", "actions_changed")
                                   if key in metric and key in encoder[mode] and metric[key] != encoder[mode][key]]
                    controls[f"{seed}:{site}:{test}:{representation}"] = dict(exact_match=not differences, differences=differences)
            for method in methods:
                result = measures(rows, method)
                data[f"{seed}:{site}:{method}"] = result
                for key, value in result.items():
                    summary.setdefault(f"{site}:{method}", {}).setdefault(key, []).append(value)
    for metrics in summary.values():
        for key, values in metrics.items():
            metrics[key] = dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else None)
    return dict(seeds=list(seeds), sources=sorted(sources), per_seed=data, aggregate=summary, files=manifest,
                checkpoints=checkpoints, encoder_r2=encoder_r2, donor_tv=donor_tv, baseline_consistency=controls)


def comparison(result):
    lines = ["# Block-2 pilot versus block 4", "", "16 selection / 32 confirmation episodes × 1,024 steps; seed 42 only.",
             "Every cell is **B2 ; B4**. Ladder/Full-State cells are occupancy / mirror actions changed; other main-table cells are reward/step.",
             "No across-agent uncertainty estimate from one seed. Ladder joint encoder entries repeat its single representation.", ""]
    def f(site, method, key):
        row = result["aggregate"][f"{site}:{method}"][key]
        return f"{row['mean']:.4f}" + (f"±{row['sd']:.4f}" if row["sd"] is not None else "")
    def pair(method, key):
        return " ; ".join(f(site, method, key) for site in (1, 3))
    def ladder(method, prefix):
        return " ; ".join(f"{f(site, method, prefix+'_occupancy')}/{f(site, method, prefix+'_mirror')}" for site in (1, 3))
    for title, methods in (("Closed-loop methods", METHODS), ("Encoder arms", ENCODERS)):
        lines += [f"## {title}", "", "| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State |", "|---|---:|---:|---:|---:|---:|"]
        for method in methods:
            lines.append(f"| {method} | {ladder(method, 'ladder')} | {pair(method, 'moving_reward')} | "
                         f"{pair(method, 'xor_eps02')} | {pair(method, 'xor_v2')} | {ladder(method, 'full')} |")
        lines.append("")
    lines += ["## Moving specificity", "", "| Method | Gap closed (B2 ; B4) | Factor-1 actions changed (B2 ; B4) |", "|---|---:|---:|"]
    for method in (*METHODS, *ENCODERS):
        lines.append(f"| {method} | {pair(method, 'moving_gap')} | {pair(method, 'moving_f1_changed')} |")
    return lines


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", nargs="+", type=int, default=(42,))
    args = parser.parse_args()
    result = aggregate(REPO, args.seed)
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    (output / "pilot_summary.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    lines = comparison(result)
    lines += ["", "## Affine/exact versus Eq.(4) control audit", ""]
    for site in (1, 3):
        controls = [row for name, row in result["baseline_consistency"].items() if name.split(":")[1] == str(site)]
        lines.append(f"B{site+1}: {sum(row['exact_match'] for row in controls)}/{len(controls)} exact metric/strength matches.")
    lines += ["", "Differences are retained, not waived. Per-field discrepancies are in `pilot_summary.json:baseline_consistency`; their cause is not established.",
              "", "## B2 full-donor TV", "", "Conditional on supported matches; undefined for zero coverage. Unit and selected-lambda replacements are separate.",
              "Archived B4 reports did not save empirical TV; supported B4 unit replacement reproduces the donor distribution by construction.", "",
              "| Seed / site / test / donor | Mode | Lambda | Coverage | TV unit | TV selected |", "|---|---|---:|---:|---:|---:|"]
    display = lambda value: "—" if value is None else f"{value:.6f}"
    for name, rows in result["donor_tv"].items():
        for mode, row in rows.items():
            cells = [str(mode), str(row["strength"]), display(row["donor_coverage"]), display(row["donor_tv_unit"]), display(row["donor_tv_selected"])]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Held-out encoder R²", "", "R² predicts activations using each arm's belief-input source. Cells are **B2 ; B4**.", "",
              "| Seed / test / coordinates | Affine exact | Affine decoded | MLP exact | MLP decoded |", "|---|---:|---:|---:|---:|"]
    for seed in result["seeds"]:
        for test in TESTS:
            for representation in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
                cells = [" ; ".join(display(result["encoder_r2"][f"{seed}:{site}:{test}:{representation}_{kind}_{source}"])
                                   for site in (1, 3)) for kind in ("affine", "mlp") for source in ("exact", "decoded")]
                lines.append(f"| {seed} / {test} / {representation} | " + " | ".join(cells) + " |")
    (output / "pilot_tables.md").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(result['per_seed'])} method/site/seed rows; {len(result['files'])} compact input files")
    print(f"Baseline audit: {sum(row['exact_match'] for row in result['baseline_consistency'].values())}/{len(result['baseline_consistency'])} affine/exact controls match Eq.(4) exactly")


if __name__ == "__main__":
    main()
