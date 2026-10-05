"""Overlay CE and CE+Kelly per rung (and the eps=0 control).

    python -m experiments.factored_sum_channel_n3_2026_10.plot_ladder \
        RESULTS_DIR [RESULTS_DIR ...] --out-dir results/ladder

Each RESULTS_DIR holds a run's summary.json and curve.json. Writes
``ladder_<rung>.png`` (probe + excess CE) and ``diagnostics_<rung>.png``.
Works for both the N=3 and N=4 studies.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.factored_sum_channel_n3_2026_10.heldout import (
    plot_curves,
    plot_diagnostics,
)
from experiments.factored_sum_channel_n3_2026_10.process import EPSILON


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rungs: dict[str, dict[str, list]] = defaultdict(dict)
    references: dict[str, dict] = {}
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        curve = json.loads((run / "curve.json").read_text())
        rung = f"n{summary['n_factors']}_h{round(100 * summary['heldout_fraction'])}"
        if summary["epsilon"] != EPSILON:
            rung += f"_eps{summary['epsilon']:g}"
        arm = "CE+Kelly" if summary["condition"].endswith("kelly") else "CE"
        rungs[rung][f"{arm} s{summary['seed']}"] = curve["checkpoints"]
        references.setdefault(rung, curve["references"])
    for rung, series in sorted(rungs.items()):
        series = dict(sorted(series.items()))
        for plot, stem in ((plot_curves, "ladder"), (plot_diagnostics, "diagnostics")):
            print(
                plot(
                    series,
                    references[rung],
                    args.out_dir / f"{stem}_{rung}.png",
                    title=rung,
                )
            )


if __name__ == "__main__":
    main()
