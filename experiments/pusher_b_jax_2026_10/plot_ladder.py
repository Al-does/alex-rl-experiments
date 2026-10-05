"""One probe/eval-loss plot per held-out rung, overlaying CE and CE+Kelly.

    python -m experiments.pusher_b_jax_2026_10.plot_ladder \
        RESULTS_DIR [RESULTS_DIR ...] --out-dir results/heldout_ladder

Each RESULTS_DIR is a run's results directory (holds summary.json and
curve.json). Writes ``ladder_h{50,65,80,95}.png``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.pusher_b_jax_2026_10.heldout import plot_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rungs: dict[int, dict[str, list]] = defaultdict(dict)
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        curve = json.loads((run / "curve.json").read_text())
        held = round(100 * summary["heldout_fraction"])
        arm = "CE+Kelly" if summary["condition"].endswith("kelly") else "CE"
        rungs[held][f"{arm} s{summary['seed']}"] = curve["checkpoints"]
    for held, series in sorted(rungs.items()):
        path = plot_curves(
            dict(sorted(series.items())),
            args.out_dir / f"ladder_h{held}.png",
            title=f"{held}% of sequences held out",
        )
        print(path)


if __name__ == "__main__":
    main()
