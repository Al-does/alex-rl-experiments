"""Validate the six frozen-policy reports and tabulate the requested coordinates."""
import json

from .analyze import HERE, STUDY, sha

LAGS = (0, 1, 2, 8, 32, 128)
PAPER_FILTER = {
    "Full-State Quotient": {"mirror": [.589, .570], "rho_flip": [.643, .586]},
    "Reward-State Quotient": {"mirror": [.603, .577], "rho_flip": [.641, .582]},
}
PAPER_TRANSFORMER = {
    "Full-State Quotient": {"mirror": [.463, .012], "rho_flip": [.555, .007]},
    "Reward-State Quotient": {"mirror": [.350, .006], "rho_flip": [.581, .011]},
}
CHECKPOINTS = {
    (2, "transformer"): "6036a76c1f0c40b2a13481bf52787803e9cef390fd5fae07d7248e0b1966ce6a",
    (3, "transformer"): "d43a629419a7795ed18202e1537883906dba69e889575bb7cc090b0a21366589",
    (2, "gru128"): "366c60e7ed8b46e3c4dc054b2c4316b188dd8cb6b73c52590ba195ea1a317fe9",
    (3, "gru128"): "ee5ab547b590c8d0f503610ed7549e46a379ed18131911e64f9de66c7b407d47",
    (2, "gru92"): "7430df711a461754891f54f471f93f6172d4ce17cb8c0e49f6d53bdd48b65750",
    (3, "gru92"): "5047f39c5a19c490f6897dbde54067175393d93768743b5e9b602d306e569c04",
}


def main():
    reports, rows, sanity_rows = [], [], []
    for (variant, architecture), expected in CHECKPOINTS.items():
        path = HERE / "results" / f"variant_{variant}_{architecture}_seed42.json"
        report = json.loads(path.read_text())
        assert report["checkpoint"]["sha256"] == expected
        assert report["checkpoint"]["step"] == 2012217344
        for name, digest in report["source_sha256"].items():
            assert sha(HERE / name) == digest, f"source differs: {name}"
        condition = "Reward-State Quotient" if variant == 2 else "Full-State Quotient"
        for name, item in report["pulse"]["conditions"].items():
            mode, site = name.rsplit("_", 1)
            coordinate = 1 if mode == "mirror" else 0
            curve = [v[coordinate] for v in item["measured"]["absolute"]]
            assert len(curve) == 129 and all(v >= 0 for v in curve)
            rows.append({"architecture": architecture, "condition": condition, "pulse": mode,
                         "site": site, "at_pulse": curve[0], "plus_one": curve[1],
                         "retained_fraction": curve[1] / curve[0] if curve[0] else None,
                         "lags": {str(k): curve[k] for k in LAGS[2:]}})
        if architecture == "transformer":
            local = STUDY / "artifacts" / "arch_pulse" / f"transformer_v{variant}_original.json"
            assert sha(local) == report["transformer_sanity"]["full_report_sha256"]
            original = json.loads(local.read_text())
            for name, item in report["pulse"]["conditions"].items():
                coordinate = 1 if name.startswith("mirror") else 0
                new = [v[coordinate] for v in item["measured"]["absolute"]]
                old = [v[coordinate] for v in original["pulse"]["conditions"][name]["measured"]["absolute"]]
                sanity_rows.append({"condition": condition, "pulse_site": name,
                                    "original_at_pulse": old[0], "original_plus_one": old[1],
                                    "cpu_at_pulse": new[0], "cpu_plus_one": new[1],
                                    "max_requested_coordinate_difference": max(abs(a-b) for a, b in zip(new, old))})
        reports.append({"path": str(path.relative_to(HERE)), "sha256": sha(path),
                        "architecture": architecture, "condition": condition,
                        "checkpoint": report["checkpoint"], "probes": report["probes"],
                        "reference_driver_revision": report["reference_driver_revision"],
                        "results_revision": report["results_revision"], "runtime": report["runtime"]})
    summary = {"rows": rows, "reports": reports, "transformer_sanity": sanity_rows,
               "paper_exact_filter_reference": PAPER_FILTER, "paper_transformer_15_seed_block3": PAPER_TRANSFORMER,
               "warning": "CPU refit is not numerically identical to GPU stable reports; see transformer_sanity. Single training seed, fixed-input intervention, not a behavioral test."}
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Seed-42 architecture pulse comparison", "", "CPU refits; each cell is at-pulse → +1 (retained fraction). GRU sites are carried hidden states; Transformer sites are post-block, decoded after block 4.", "",
             "| Agent | Condition | Pulse | Site 1 | Site 2 | Site 3 | Site 4 |", "|---|---|---|---:|---:|---:|---:|"]
    for architecture in ("transformer", "gru128", "gru92"):
        for condition in ("Full-State Quotient", "Reward-State Quotient"):
            for mode in ("mirror", "rho_flip"):
                selected = [r for r in rows if (r["architecture"], r["condition"], r["pulse"]) == (architecture, condition, mode)]
                cells = [f"{r['at_pulse']:.6f} → {r['plus_one']:.6f} ({r['retained_fraction']:.3f})" for r in selected]
                cells += ["—"] * (4-len(cells))
                label = "δ mirror" if mode == "mirror" else "ρ flip"
                lines.append(f"| {architecture} | {condition} | {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Later lags", "", "| Agent | Condition | Pulse/site | +2 | +8 | +32 | +128 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['architecture']} | {row['condition']} | {row['pulse']} {row['site']} | " + " | ".join(f"{v:.6f}" for v in row["lags"].values()) + " |")
    lines += ["", "## Held-out decoding R²", "", "Site order is bottom to top; Transformer additionally includes final LayerNorm (site 5).", "", "| Agent | Condition | ρ by site | δ by site |", "|---|---|---|---|"]
    for report in reports:
        lines.append(f"| {report['architecture']} | {report['condition']} | " + " | ".join(", ".join(f"{p[k]:.6f}" for p in report["probes"]) for k in ("rho", "delta")) + " |")
    lines += ["", "## Transformer sanity: CPU refit versus original GPU stable reports", "", "| Condition | Pulse/site | Original at/+1 | CPU at/+1 | Max curve difference (requested coordinate) |", "|---|---|---:|---:|---:|"]
    for row in sanity_rows:
        lines.append(f"| {row['condition']} | {row['pulse_site']} | {row['original_at_pulse']:.9f}/{row['original_plus_one']:.9f} | {row['cpu_at_pulse']:.9f}/{row['cpu_plus_one']:.9f} | {row['max_requested_coordinate_difference']:.9f} |")
    lines += ["", "Exact numerical reproduction is **not** established. The pinned pulse/core source hashes match the original report's hashes. Exact-filter curves agree within 1.8e-6 across all lags, but fitted probe scores differ slightly; the contribution of CPU/GPU activations versus fitting is not isolated. Do not call this an exact sanity pass.", "", "## Paper references (not these agents' own-history filter estimates)", "", "| Condition | Pulse | Exact filter at/+1 | 15-seed Transformer block 3 at/+1 |", "|---|---|---:|---:|"]
    for condition, modes in PAPER_FILTER.items():
        for mode, values in modes.items():
            baseline = PAPER_TRANSFORMER[condition][mode]
            lines.append(f"| {condition} | {mode} | {values[0]:.3f}/{values[1]:.3f} | {baseline[0]:.3f}/{baseline[1]:.3f} |")
    (HERE / "results" / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(reports)} reports, {len(rows)} pulse/site curves, 129 lags each; checkpoint and analysis-source hashes match.")
    print("Transformer requested-coordinate maximum absolute curve difference:", max(r["max_requested_coordinate_difference"] for r in sanity_rows))


if __name__ == "__main__":
    main()
