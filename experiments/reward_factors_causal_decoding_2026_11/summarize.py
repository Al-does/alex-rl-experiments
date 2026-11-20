"""Render compact pilot tables from immutable per-job reports."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LADDER = ROOT.parent / "belief_quotient_causal_decoding_2026_11"


def main():
    files = sorted([p for root in (ROOT, LADDER) for p in (root / "results").glob("pilot_*.json")])
    data = {(r["test"], r["method"]): r for p in files for r in (json.loads(p.read_text()),)}
    tests = ("ladder", "moving", "xor_eps02", "xor_v2", "ladder_full")
    methods = ("intact", "eq4", "eq4_joint", "policy", "das", "band", "mlp", "null", "norm_null", "map")
    lines = ["# Seed-42 CPU pilot", "", "Two fit, tuning and confirmation episodes; 1,024 decisions each.",
             "No uncertainty or population inference is justified. Eq. (4) retains archived strengths,",
             "but its pilot decoder fits use fewer rows than the archive.", "",
             "## Closed loop", "",
             "Ladder cells: rho-flip occupancy / delta-mirror actions changed. Other cells: reward per step.", "",
             "| Method | Ladder | Moving | XOR eps02 | XOR v2 | Full-State control |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        cells = []
        for test in tests:
            row = data.get((test, method))
            if row is None:
                cells.append("—")
            elif test.startswith("ladder"):
                cells.append(f"{row['metrics']['2']['reward']:.4f} / {row['metrics']['1']['actions_changed']:.4f}")
            else:
                cells.append(f"{row['metrics']['0']['reward']:.4f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## Moving transfer", "", "Gap closed uses the *paired pilot* exact-filter map and intact controls.",
              "| Method | Reward | Gap closed | Factor-1 reward | Factor-1 actions changed |",
              "|---|---:|---:|---:|---:|"]
    intact = data['moving', 'intact']['metrics']['0']['reward']
    reference = data['moving', 'map']['metrics']['0']['reward']
    for method in methods:
        row = data['moving', method]['metrics']['0']
        gap = (row['reward'] - intact) / (reference - intact)
        lines.append(f"| {method} | {row['reward']:.4f} | {gap:.3f} | {row['factor1_reward']:.4f} | {row['actions_changed']:.4f} |")
    lines += ["", "## Held-out decoders", "", "R² is pooled across belief probabilities; policy inverse entries are per-factor probability MAE.",
              "| Test | Affine R² | Band R² | MLP R² | Policy-implied MAE | DAS rank | DAS block-4 fidelity / reference | Block-2 fidelity / reference (same rank) |",
              "|---|---:|---:|---:|---|---:|---:|---:|"]
    for test in tests:
        row = data[test, 'das']['diagnostics']['das']
        rank = str(row['selected_rank_block4'])
        fidelity = row['3'][rank]['held_out']
        early = row['1'][rank]['held_out']
        mae = ", ".join(f"{v:.3f}" for v in data[test, 'policy']['diagnostics']['policy_implied_mae_per_factor'])
        lines.append(f"| {test} | {data[test, 'eq4']['diagnostics']['affine_r2']:.3f} | {data[test, 'band']['diagnostics']['band_r2']:.3f} | {data[test, 'mlp']['diagnostics']['decoder_r2']:.3f} | {mae} | {rank} | {fidelity['fidelity']:.3f} / {fidelity['reference_patch_fidelity']:.3f} | {early['fidelity']:.3f} / {early['reference_patch_fidelity']:.3f} |")
    allocation = sum(r['wall_seconds'] * len(r['cpus']) for r in data.values()) / 3600
    lines += ["", f"Recorded final-job CPU allocation × wall time: {allocation:.2f} core-hours (not process CPU time; excludes discarded pilots).", ""]
    result = "\n".join(lines)
    (ROOT / "tables.md").write_text(result)
    manifest = {str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT / "results" / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
