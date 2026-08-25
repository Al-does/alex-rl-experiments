#!/usr/bin/env python3
"""One-off: project legacy results/progress.jsonl into training_curves.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Reuse harness projection until experiment storage module lands on main.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workspace"))
from harness.training_curves import compact_training_curve_row  # noqa: E402


def convert(progress_path: Path) -> int:
    lines = [
        line
        for line in progress_path.read_text().splitlines()
        if line.strip()
    ]
    curves_path = progress_path.with_name("training_curves.jsonl")
    compact_lines: list[str] = []
    for line in lines:
        row = compact_training_curve_row(json.loads(line))
        if row:
            compact_lines.append(json.dumps(row, sort_keys=True))
    curves_path.write_text("\n".join(compact_lines) + ("\n" if compact_lines else ""))
    return len(compact_lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: convert_progress_to_training_curves.py <progress.jsonl> ...")
        return 2
    for raw in argv[1:]:
        path = Path(raw)
        count = convert(path)
        print(f"{path}: wrote {count} compact rows -> {path.with_name('training_curves.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
