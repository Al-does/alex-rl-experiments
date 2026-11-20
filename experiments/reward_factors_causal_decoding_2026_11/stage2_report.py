"""Summarize the resumable stage-2 CPU pilot without loading JAX."""

import hashlib
import json
from pathlib import Path

from .run import METHODS, TESTS

METHODS = tuple(method for method in METHODS if method not in ("action", "behavioural", "behavioural_action"))

ROOT = Path(__file__).resolve().parent
LADDER = ROOT.parent / "belief_quotient_causal_decoding_2026_11"


def main():
    files = sorted(p for root in (ROOT, LADDER) for p in (root / "results").glob("stage2_*_seed42_*.json"))
    data = {(r["test"], r["method"]): r for p in files for r in (json.loads(p.read_text()),)}
    for test in TESTS:
        for method in METHODS:
            if (test, method) not in data:
                raise ValueError(f"missing {test} {method}")
    tests = ("ladder", "moving", "xor_eps02", "xor_v2", "ladder_full")
    lines = ["# Stage 2: seed-42 local CPU pilot", "",
             "Ladder and moving: eight episodes each for fit, held-out diagnosis, tuning and confirmation.",
             "Other tests: two episodes per split. All episodes: 1,024 steps. Gradient writes: 60 steps.",
             "Ladder cells: rho-flip occupancy / delta-mirror actions changed. Others: reward/step.", "",
             "| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        cells = []
        for test in tests:
            row = data[test, method]["metrics"]
            cells.append(f"{row['2']['reward']:.4f} / {row['1']['actions_changed']:.4f}" if test.startswith("ladder")
                         else f"{row['0']['reward']:.4f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## Moving transfer", "", "Gap uses paired intact and exact-filter map references.", "",
              "| Method | Reward | Gap closed | Factor-1 reward | Factor-1 actions changed | Donor coverage |",
              "|---|---:|---:|---:|---:|---:|"]
    intact = data["moving", "intact"]["metrics"]["0"]["reward"]
    reference = data["moving", "map"]["metrics"]["0"]["reward"]
    for method in METHODS:
        row = data["moving", method]["metrics"]["0"]
        lines.append(f"| {method} | {row['reward']:.4f} | {(row['reward']-intact)/(reference-intact):.3f} | "
                     f"{row['factor1_reward']:.4f} | {row['actions_changed']:.4f} | {row['donor_coverage']:.3f} |")
    lines += ["", "## Policy-map calibration", "", "KL direction is actual agent || map(Bayes), temperature 1.5.",
              "Mean-policy KL uses a constant fitted on training policies; inverse KL uses map(decoded belief).", "",
              "| Test | Map train KL | Map test KL | Map test TV | Mean-policy test KL | Inverse policy KL | Inverse–Bayes MAE/factor | rho MAE/factor | signed delta MAE/factor |",
              "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for test in tests:
        row = data[test, "policy"]["diagnostics"]
        scalar = [f"{row[key]:.5f}" for key in ("policy_map_train_kl", "policy_map_test_kl", "policy_map_test_tv",
                                               "policy_mean_test_kl", "inverse_policy_test_kl")]
        gaps = [", ".join(f"{v:.4f}" for v in row[key]) for key in
                ("policy_implied_mae_per_factor", "policy_implied_rho_mae", "policy_implied_delta_mae")]
        lines.append(f"| {test} | " + " | ".join(scalar + gaps) + " |")
    lines += ["", "## Passive decoders and DAS", "",
              "| Test | Affine R² | Band R² | MLP R² | DAS rank | Block-4 held-out fidelity |",
              "|---|---:|---:|---:|---:|---:|"]
    for test in tests:
        das = data[test, "das"]["diagnostics"]["das"]
        rank = das["selected_rank_block4"]
        fidelity = das["3"][str(rank)]["held_out"]["fidelity"]
        lines.append(f"| {test} | {data[test, 'eq4']['diagnostics']['affine_r2']:.3f} | "
                     f"{data[test, 'band']['diagnostics']['band_r2']:.3f} | "
                     f"{data[test, 'mlp']['diagnostics']['decoder_r2']:.3f} | {rank} | {fidelity:.3f} |")
    lines += ["", "## Wall seconds per method (two allocated CPU cores per worker)", "",
              "Includes checkpoint loading, fitting, diagnostics, lambda selection, and confirmation.",
              "Jobs share eight CPU cores; these are not isolated performance benchmarks.", "",
              "| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        lines.append(f"| {method} | " + " | ".join(f"{data[test, method]['wall_seconds']:.1f}" for test in tests) + " |")
    allocation = sum(row["wall_seconds"] * len(row["cpus"]) for row in data.values()) / 3600
    lines += ["", f"Total allocated core-hours: {allocation:.3f} (excludes preliminary diagnostic runs).", ""]
    text = "\n".join(lines)
    (ROOT / "stage2_tables.md").write_text(text)
    manifest = {str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT / "results" / "stage2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
