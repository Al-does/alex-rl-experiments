"""Merge compact selected-fit reports without opening model artifacts."""
import argparse
import csv
import json
from pathlib import Path

from .data import LADDER, ROOT, RR


def read_fits(inputs, *, include_partial=False, run_prefix=None):
    """Read atomic fit snapshots, including fitted-but-not-yet-probed rows on request."""
    seen = set()
    for root in inputs:
        for manifest in sorted(Path(root).rglob("run_manifest.json")):
            if run_prefix and not manifest.parent.name.startswith(run_prefix+"-"):
                continue
            m = json.loads(manifest.read_text())
            if m["status"] != "completed" and not include_partial:
                continue
            for path in sorted(manifest.parent.glob("seed*_k*.json")):
                fit = json.loads(path.read_text())
                if not {"condition", "seed", "control", "class", "k", "test_kl", "selected"} <= fit.keys():
                    continue
                key = tuple(fit[k] for k in ("condition", "seed", "control", "class", "k"))
                if key in seen:
                    raise ValueError(f"duplicate fit {key}; choose disjoint inputs")
                seen.add(key)
                yield manifest, m, path, fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, default=[ROOT/s/"analysis_gbs"/"results" for s in (LADDER, RR)])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for manifest, _, path, fit in read_fits(args.inputs):
        summary = json.loads(manifest.with_name("summary.json").read_text())
        elbow, = [e for e in summary["elbows"] if e["seed"] == fit["seed"] and e["class"] == fit["class"]]
        row = {k: fit[k] for k in ("condition", "seed", "control", "class", "k", "test_kl")}
        row.update(val_kl=fit["selected"]["val_kl"], elbow_k=elbow["elbow_k"],
                   commutation_mse=(fit.get("commutation") or {}).get("mse"),
                   commutation_baseline=(fit.get("commutation") or {}).get("constant_decoder_mse"))
        for target, metrics in (fit.get("geometry") or {}).items():
            for metric, value in metrics.items():
                row[f"{target}_{metric}"] = value
        row["hmm_matrix_error"] = (fit.get("hmm_matrices") or {}).get("combined_mse")
        row.update({key: value for key, value in fit.items() if key.endswith("_error")})
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
