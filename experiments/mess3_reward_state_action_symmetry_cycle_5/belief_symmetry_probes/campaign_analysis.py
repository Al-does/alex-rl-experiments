"""Cycle 5 wrapper for shared belief-probe campaign aggregation."""

from pathlib import Path
import sys

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.campaign_analysis import (
    aggregate,
    main as _main,
)

__all__ = ["aggregate"]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--root" not in arguments:
        arguments[0:0] = ["--root", str(Path(__file__).resolve().parent)]
    return _main(arguments, cycle=5)


if __name__ == "__main__":
    raise SystemExit(main())
