"""Merge compact selected-fit reports without opening model artifacts."""
import argparse
import csv
import json
from pathlib import Path

from .data import LADDER, ROOT, RR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, default=[ROOT/s/"analysis_gbs"/"results" for s in (LADDER, RR)])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, seen = [], set()
    for root in args.inputs:
        for manifest in sorted(root.rglob("run_manifest.json")):
            m = json.loads(manifest.read_text())
            if m["status"] != "completed":
                continue
            summary = json.loads(manifest.with_name("summary.json").read_text())
            for path in sorted(manifest.parent.glob("seed*_k*.json")):
                fit = json.loads(path.read_text())
                key = (fit["condition"], fit["seed"], fit["control"], fit["class"], fit["k"])
                if key in seen:
                    raise ValueError(f"duplicate fit {key}; choose disjoint inputs")
                seen.add(key)
                elbow, = [e for e in summary["elbows"] if e["seed"] == fit["seed"] and e["class"] == fit["class"]]
                row = {k: fit[k] for k in ("condition", "seed", "control", "class", "k", "test_kl")}
                row.update(val_kl=fit["selected"]["val_kl"], elbow_k=elbow["elbow_k"],
                           commutation_mse=fit["commutation"]["mse"],
                           commutation_baseline=fit["commutation"]["constant_decoder_mse"])
                for target, metrics in fit["geometry"].items():
                    for metric, value in metrics.items():
                        row[f"{target}_{metric}"] = value
                row["hmm_matrix_error"] = (fit["hmm_matrices"] or {}).get("combined_mse")
                rows.append(row)
    if not rows:
        raise ValueError("no completed fits found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} fits to {args.output}")


if __name__ == "__main__":
    main()
