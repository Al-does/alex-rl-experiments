"""Validate seed-42 pilot reports and compare with saved block-4 controls."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .run import METHODS, TESTS

REPO = Path(__file__).resolve().parents[2]
ENCODERS = tuple(f"{r}_{k}_{s}" for r in ("marginal", "joint") for k in ("affine", "mlp") for s in ("exact", "decoded"))


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
    manifest, data, summary = {}, {}, {}
    sources = set()
    methods = (*METHODS, *ENCODERS)
    for seed in seeds:
        for site in (1, 3):
            rows = {}
            for test in TESTS:
                for method in methods:
                    actual = method.replace("joint_", "marginal_", 1) if test.startswith("ladder") and method.startswith("joint_") else method
                    row, file, archived = load(root, test, seed, actual, site)
                    rows[test, method] = row
                    manifest[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
                    sources.add("saved_B4" if archived else f"fresh_site{site}")
            for method in methods:
                result = measures(rows, method)
                data[f"{seed}:{site}:{method}"] = result
                for key, value in result.items():
                    summary.setdefault(f"{site}:{method}", {}).setdefault(key, []).append(value)
    for metrics in summary.values():
        for key, values in metrics.items():
            metrics[key] = dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else None)
    return dict(seeds=list(seeds), sources=sorted(sources), per_seed=data, aggregate=summary, files=manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", nargs="+", type=int, default=(42,))
    args = parser.parse_args()
    result = aggregate(REPO, args.seed)
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    (output / "pilot_summary.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    lines = ["# Block-2 pilot versus block 4", "", "16 selection / 32 confirmation episodes × 1,024 steps; seed 42 only.",
             "B4 is saved campaign data unless a fresh site-3 report exists. No across-agent uncertainty estimate from one seed.", "",
             "| Method | Site | Ladder occupancy / mirror | Moving reward / gap / F1 changed | XOR eps02 | XOR v2 | Full-State occupancy / mirror |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, row in result["aggregate"].items():
        site, method = name.split(":", 1)
        f = lambda key: f"{row[key]['mean']:.4f}" + (f"±{row[key]['sd']:.4f}" if row[key]["sd"] is not None else "")
        lines.append(f"| {method} | B{int(site)+1} | {f('ladder_occupancy')} / {f('ladder_mirror')} | "
                     f"{f('moving_reward')} / {f('moving_gap')} / {f('moving_f1_changed')} | {f('xor_eps02')} | {f('xor_v2')} | "
                     f"{f('full_occupancy')} / {f('full_mirror')} |")
    (output / "pilot_tables.md").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(result['per_seed'])} method/site/seed rows; {len(result['files'])} compact input files")


if __name__ == "__main__":
    main()
