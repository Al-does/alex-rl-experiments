"""Cycle 5 wrapper for token-swap versus probe comparison."""

from collections.abc import Sequence

from experiments.mess3_reward_state_action_symmetry_cycle_4.token_swap_diagnostic.compare_campaign import (
    main as _main,
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv, cycle=5)


if __name__ == "__main__":
    raise SystemExit(main())
