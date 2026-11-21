"""Validate the complete encoder ablation and emit paired compact tables."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from .run import ARMS, REPO, TESTS

SEEDS = (42, 43, 44)


def read(path):
    return json.loads(path.read_text())


def paths(root, test, seed, representation, kind, source, pilot=False):
    prefix = "belief_quotient" if test.startswith("ladder") else "reward_factors"
    return root / "experiments" / f"{prefix}_encoder_nonlinearity_2026_11/results" / f"{'pilot' if pilot else 'campaign'}_{test}_seed{seed}_{representation}_{kind}_{source}.json"


def collect(root=REPO, pilot=False):
    rows, manifest = {}, {}
    seeds = (42,) if pilot else SEEDS
    fingerprints = set()
    for test, seed in itertools.product(TESTS, seeds):
        representations = ("marginal",) if test.startswith("ladder") else ("marginal", "joint")
        for representation, (kind, source) in itertools.product(representations, ARMS):
            path = paths(root, test, seed, representation, kind, source, pilot)
            row = read(path)
            required = dict(test=test, seed=seed, representation=representation, encoder=kind, source=source,
                            pilot=pilot, historical_encoder_source="exact", write_current_source="probe_decoded_unclipped",
                            site=3, horizon=1024)
            if not pilot:
                required.update(tuning_episodes=16, confirmation_episodes=32, held_episodes=12,
                                train_rows=12000 if test.startswith("ladder") else 60000)
            for key, value in required.items():
                if row.get(key) != value:
                    raise ValueError(f"{path}: expected {key}={value}")
            fingerprints.add(tuple(sorted(row["code_hashes"].items())))
            rows[test, seed, representation, kind, source] = row
            manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(fingerprints) != 1:
        raise ValueError("mixed experiment code hashes")
    return rows, manifest


def mean_sd(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else 0.)


def display(values):
    result = mean_sd(values)
    return f"{result['mean']:.4f}±{result['sd']:.4f}"


def aggregate(root=REPO, pilot=False):
    rows, manifest = collect(root, pilot)
    seeds = (42,) if pilot else SEEDS
    reference_manifest, platform_differences = {}, []
    references = {}
    if not pilot:
        for test, seed in itertools.product(TESTS, seeds):
            prefix = "belief_quotient" if test.startswith("ladder") else "reward_factors"
            for method in ("eq4", "eq4_joint", "intact", "map"):
                path = root / f"experiments/{prefix}_causal_decoding_2026_11/results/campaign_{test}_seed{seed}_{method}.json"
                reference = read(path)
                for key, expected in dict(pilot=False, campaign=True, smoke=False, tuning_episodes=16,
                                          confirmation_episodes=32, horizon=1024).items():
                    if reference.get(key) != expected:
                        raise ValueError(f"invalid fresh control {path}: {key}")
                references[test, seed, method] = reference
                reference_manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                cpu_path = root / f"experiments/reward_factors_encoder_nonlinearity_2026_11/results/cpu_recheck_{test}_seed{seed}_{method}.json"
                if test == "moving" and method in ("intact", "map"):
                    cpu_reference = read(cpu_path)
                    for key in ("test", "seed", "method", "checkpoint", "campaign", "smoke", "pilot", "confirmation_episodes", "tuning_episodes", "horizon"):
                        if cpu_reference[key] != reference[key]:
                            raise ValueError(f"CPU reference provenance differs: {cpu_path}: {key}")
                    references[test, seed, method] = cpu_reference
                    reference_manifest[str(cpu_path.relative_to(root))] = hashlib.sha256(cpu_path.read_bytes()).hexdigest()
            for representation in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
                arm = rows[test, seed, representation, "affine", "exact"]
                reference = references[test, seed, "eq4_joint" if representation == "joint" else "eq4"]
                if arm["checkpoint"]["sha256"] != reference["checkpoint"]["sha256"]:
                    raise ValueError(f"checkpoint differs: {test} {seed}")
                mismatches = []
                for mode, metrics in arm["matched_lambda_metrics"].items():
                    for key in ("strength", "reward", "factor1_reward", "actions_changed"):
                        if metrics[key] != reference["metrics"][mode][key]:
                            mismatches.append(dict(mode=mode, metric=key, local=metrics[key], published=reference["metrics"][mode][key]))
                if mismatches:
                    method = "eq4_joint" if representation == "joint" else "eq4"
                    cpu_path = root / f"experiments/reward_factors_encoder_nonlinearity_2026_11/results/cpu_recheck_{test}_seed{seed}_{method}.json"
                    cpu_reference = read(cpu_path)
                    for key in ("test", "seed", "method", "checkpoint", "campaign", "smoke", "pilot", "confirmation_episodes", "tuning_episodes", "horizon"):
                        if cpu_reference[key] != reference[key]:
                            raise ValueError(f"CPU recheck provenance differs: {cpu_path}: {key}")
                    for mode, metrics in arm["matched_lambda_metrics"].items():
                        for key in ("strength", "reward", "factor1_reward", "actions_changed"):
                            if metrics[key] != cpu_reference["metrics"][mode][key]:
                                raise ValueError(f"historical CPU recheck mismatch: {test} {seed} {representation} {mode} {key}")
                    reference_manifest[str(cpu_path.relative_to(root))] = hashlib.sha256(cpu_path.read_bytes()).hexdigest()
                    platform_differences.append(dict(test=test, seed=seed, representation=representation, differences=mismatches,
                                                     unchanged_runner_cpu_recheck=str(cpu_path.relative_to(root))))

    metric_keys = ("ladder_occupancy", "ladder_mirror", "moving_reward", "moving_gap_closed", "moving_f1_changed",
                   "xor_eps02_reward", "xor_v2_reward", "full_occupancy", "full_mirror")

    def measurements(representation, kind, source, metric_set):
        values = {key: [] for key in metric_keys}
        for seed in seeds:
            for test in TESTS:
                rep = "marginal" if test.startswith("ladder") else representation
                row = rows[test, seed, rep, kind, source][metric_set]
                if test.startswith("ladder"):
                    prefix = "full" if test == "ladder_full" else "ladder"
                    values[prefix + "_occupancy"].append(row["2"]["reward"])
                    values[prefix + "_mirror"].append(row["1"]["actions_changed"])
                elif test == "moving":
                    values["moving_reward"].append(row["0"]["reward"])
                    values["moving_f1_changed"].append(row["0"]["actions_changed"])
                    if not pilot:
                        intact = references[test, seed, "intact"]["metrics"]["0"]["reward"]
                        mapped = references[test, seed, "map"]["metrics"]["0"]["reward"]
                        values["moving_gap_closed"].append((row["0"]["reward"] - intact) / (mapped - intact))
                else:
                    values[test + "_reward"].append(row["0"]["reward"])
        return values

    summary, paired, encoder_scores = {}, {}, {}
    lines = ["# Eq. (4) encoder-only experiment", "", "Mean ± sample SD across agent seeds 42–44 (pilot: seed 42 only).",
             "Ladder/Full-State cells: occupancy / mirror actions changed. Gap normalization uses paired fresh intact/map references.",
             "Fixed-λ affine/exact is the local historical control; any discrepancies from previously published remote controls require an exact unchanged-runner CPU recheck (recorded in the manifest).",
             "Full-State mirror changes are a positive-control response, not a spillover-minimization objective.",
             "", "## Held-out encoder R²", "", "| Test / coordinates | Affine exact | Affine decoded | MLP exact | MLP decoded |",
             "|---|---:|---:|---:|---:|"]
    for test in TESTS:
        for rep in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
            cells = []
            for kind, source in ARMS:
                scores = [rows[test, seed, rep, kind, source]["diagnostics"]["test_r2"] for seed in seeds]
                encoder_scores[f"{test}/{rep}/{kind}/{source}"] = dict(mean_sd(scores), per_seed=scores)
                cells.append(display(scores))
            lines.append(f"| {test} / {rep} | " + " | ".join(cells) + " |")
    for metric_set, title in (("matched_lambda_metrics", "Archived λ held fixed: strict encoder-only comparison"),
                              ("metrics", "All four arms reselected with the historical grid/rule on 16 episodes")):
        lines.extend(["", f"## {title}", "", "| Encoder / source / coordinates | Ladder ↓ / ↓ | Moving reward ↑ | Moving gap ↑ | Moving F1 changed ↓ | XOR eps02 ↑ | XOR v2 ↑ | Full-State |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for rep, (kind, source) in itertools.product(("marginal", "joint"), ARMS):
            name = f"{kind}/{source}/{rep}"
            values = measurements(rep, kind, source, metric_set)
            summary[f"{metric_set}/{name}"] = {key: dict(mean_sd(value), per_seed=value) for key, value in values.items() if value}
            f = lambda key: display(values[key]) if values[key] else "—"
            cells = [f("ladder_occupancy") + " / " + f("ladder_mirror"), f("moving_reward"), f("moving_gap_closed"),
                     f("moving_f1_changed"), f("xor_eps02_reward"), f("xor_v2_reward"), f("full_occupancy") + " / " + f("full_mirror")]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.extend(["", "### Paired differences versus historical affine/exact at its archived λ", "",
                      "Signs are raw new−historical differences (negative favors occupancy/spillover; positive favors reward/gap). Full-State mirror is nondirectional.",
                      "", "| Arm | Metric | Δ seed42 | Δ seed43 | Δ seed44 | Mean ± SD |", "|---|---|---:|---:|---:|---:|"])
        for rep, (kind, source) in itertools.product(("marginal", "joint"), ARMS):
            if kind == "affine" and source == "exact" and metric_set == "matched_lambda_metrics":
                continue
            values = measurements(rep, kind, source, metric_set)
            baseline = measurements(rep, "affine", "exact", "matched_lambda_metrics")
            name = f"{kind}/{source}/{rep}"
            for key, value in values.items():
                if not value or (rep == "joint" and (key.startswith("ladder") or key.startswith("full"))):
                    continue
                delta = [a - b for a, b in zip(value, baseline[key])]
                paired[f"{metric_set}/{name}/{key}"] = dict(mean_sd(delta), per_seed=delta,
                                                           signs=["+" if x > 0 else "−" if x < 0 else "0" for x in delta])
                cells = [f"{x:+.6f}" for x in delta] + ["—"] * (3-len(delta))
                lines.append(f"| {name} | {key} | " + " | ".join(cells) + f" | {display(delta)} |")
    destination = root / "experiments/reward_factors_encoder_nonlinearity_2026_11/results"
    destination.mkdir(parents=True, exist_ok=True)
    prefix = "pilot" if pilot else "campaign"
    (destination / f"{prefix}_summary.json").write_text(json.dumps(dict(encoder_r2=encoder_scores, metrics=summary, paired=paired), indent=2) + "\n")
    (destination / f"{prefix}_manifest.json").write_text(json.dumps(dict(reports=manifest, fresh_controls=reference_manifest,
                                                                        platform_differences=platform_differences), indent=2) + "\n")
    (destination / f"{prefix}_tables.md").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(rows)} encoder reports; {len(reference_manifest)} control files; {len(platform_differences)} published-control discrepancies reproduced exactly by the unchanged historical runner on CPU" if not pilot else f"Validated {len(rows)} pilot encoder reports")
    print(destination / f"{prefix}_tables.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", action="store_true")
    aggregate(pilot=parser.parse_args().pilot)
