"""Cycle 5 wrapper for all-checkpoint trajectory campaign aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.trajectory_campaign import (
    aggregate,
    main as _main,
    write_campaign,
)

__all__ = ["aggregate", "write_campaign"]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--root" not in arguments:
        arguments[0:0] = ["--root", str(Path(__file__).resolve().parent)]
    return _main(arguments, cycle=5)


if __name__ == "__main__":
    raise SystemExit(main())
