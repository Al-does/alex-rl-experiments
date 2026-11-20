"""Validate and summarize all 255 confirmation reports, never smoke outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from .run import METHODS, TESTS

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def mean_sd(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values))


def formatted(values):
    result = mean_sd(values)
    return f"{result['mean']:.4f} ± {result['sd']:.4f}"


def collect(root):
    data, manifest, checkpoints = {}, {}, {}
    for test in TESTS:
        study = "belief_quotient" if test.startswith("ladder") else "reward_factors"
        for seed in SEEDS:
            for method in METHODS:
                path = root / f"experiments/{study}_causal_decoding_2026_11/results/campaign_{test}_seed{seed}_{method}.json"
                row = json.loads(path.read_text())
                expected = dict(test=test, seed=seed, method=method, campaign=True, smoke=False,
                                pilot=False, fit_episodes=24, tuning_episodes=16, confirmation_episodes=32, horizon=1024)
                for key, value in expected.items():
                    if row.get(key) != value:
                        raise ValueError(f"{path}: expected {key}={value}, got {row.get(key)}")
                sha = row["checkpoint"]["sha256"]
                if checkpoints.setdefault((test, seed), sha) != sha:
                    raise ValueError(f"checkpoint mismatch: {path}")
                data[test, seed, method] = row
                manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data, manifest


def historical(root, ref):
    if subprocess.check_output(["git", "rev-parse", "--is-shallow-repository"], cwd=root, text=True).strip() == "true":
        raise ValueError("unshallow the local checkout before reading historical baseline objects")
    rows, provenance = {}, {}
    def read(study, name):
        path = f"experiments/{study}_twin_steering_2026_11/results/{name}.json"
        raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=root)
        provenance[path] = dict(ref=ref, sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(raw)
    ladder = [read("belief_quotient", f"ladder_seed{seed}") for seed in SEEDS]
    for name, old, strength in (("historical_eq4_selected", True, "selected"),
                                ("historical_twin_selected", False, "selected"),
                                ("historical_eq4_full", True, "full_strength"),
                                ("historical_twin_full", False, "full_strength")):
        comparisons = [(row["old"] if old else row)["comparisons"] for row in ladder]
        rows[name] = dict(ladder_rho=mean_sd([r["rho_flip"][strength]["occupancy"] for r in comparisons]),
                          ladder_mirror=mean_sd([r["mirror"]["full_strength"]["action_change"] for r in comparisons]))
    for cohort in ("eps02", "v2"):
        reports = [read("reward_factors", f"{cohort}_xor_seed{seed}") for seed in SEEDS]
        for method in ("marginal", "joint"):
            for old in (True, False):
                label = f"historical_{'eq4' if old else 'twin'}_{method}"
                values = []
                for row in reports:
                    comparison = (row["old"] if old else row)["comparisons"][method]
                    values.append((comparison if old else comparison["selected"])["total_reward"])
                rows.setdefault(label, {})[f"xor_{cohort}"] = mean_sd(values)
    return dict(rows=rows, provenance=provenance,
                caveat="Historical XOR uses 64 selection / 256 confirmation episodes, not this campaign's 16/32. No moving or Full-State twin run exists in these reports.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--historical-ref", default="origin/devin/1795127017-twin-steering")
    args = parser.parse_args()
    data, manifest = collect(ROOT)
    summary, lines = {}, ["# Three-seed causal-decoding campaign", "",
        "Mean ± sample SD across frozen agent seeds 42–44 (n=3), not standard error or population confidence intervals.",
        "16 selection episodes and 32 confirmation episodes per method, each 1,024 steps.",
        "Moving gaps are normalized per seed using paired intact and exact-filter-map references before aggregation.", "",
        "| Method | Ladder rho ↓ | Ladder mirror ↓ | Moving reward ↑ | Moving gap ↑ | Moving factor-1 reward | Moving factor-1 actions changed ↓ | XOR eps02 ↑ | XOR v2 ↑ | Full-State rho / mirror |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for method in METHODS:
        values = {key: [] for key in ("ladder_rho", "ladder_mirror", "moving_reward", "moving_gap", "moving_factor1_reward",
                                      "moving_factor1_changed", "xor_eps02", "xor_v2", "full_rho", "full_mirror")}
        for seed in SEEDS:
            ladder = data["ladder", seed, method]["metrics"]
            full = data["ladder_full", seed, method]["metrics"]
            moving = data["moving", seed, method]["metrics"]["0"]
            intact = data["moving", seed, "intact"]["metrics"]["0"]["reward"]
            reference = data["moving", seed, "map"]["metrics"]["0"]["reward"]
            if reference <= intact:
                raise ValueError(f"nonpositive moving reference gap for seed {seed}")
            for key, value in dict(ladder_rho=ladder["2"]["reward"], ladder_mirror=ladder["1"]["actions_changed"],
                                   full_rho=full["2"]["reward"], full_mirror=full["1"]["actions_changed"],
                                   moving_reward=moving["reward"], moving_gap=(moving["reward"]-intact)/(reference-intact),
                                   moving_factor1_reward=moving["factor1_reward"], moving_factor1_changed=moving["actions_changed"],
                                   xor_eps02=data["xor_eps02", seed, method]["metrics"]["0"]["reward"],
                                   xor_v2=data["xor_v2", seed, method]["metrics"]["0"]["reward"]).items():
                values[key].append(value)
        summary[method] = {key: mean_sd(v) for key, v in values.items()}
        cells = [formatted(values[key]) for key in list(values)[:-2]]
        cells.append(f"{formatted(values['full_rho'])} / {formatted(values['full_mirror'])}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    history = historical(ROOT, args.historical_ref)
    lines += ["", "## Historical twin / Eq. (4) references", "", history["caveat"], "",
              "| Historical method | Ladder rho | Ladder mirror | XOR eps02 | XOR v2 |", "|---|---:|---:|---:|---:|"]
    for name, row in history["rows"].items():
        cells = [f"{row[key]['mean']:.4f} ± {row[key]['sd']:.4f}" if key in row else "—"
                 for key in ("ladder_rho", "ladder_mirror", "xor_eps02", "xor_v2")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Behavioural methods are filter-assisted upper-bound-style references, not certified bounds.",
              "XOR null writes equal the Eq. (4) direction; any differences reflect lambda selection only.",
              "The ladder map row duplicates intact; it is not a Bayes-reference controller.", ""]
    folder = ROOT / "experiments/reward_factors_causal_decoding_2026_11"
    (folder / "campaign_tables.md").write_text("\n".join(lines))
    for name, result in (("campaign_manifest", manifest), ("campaign_summary", summary), ("historical_baselines", history)):
        (folder / "results" / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("\n".join(lines))
    print(f"Validated {len(manifest)} complete reports with consistent checkpoints and 16/32 episode budgets")


if __name__ == "__main__":
    main()
