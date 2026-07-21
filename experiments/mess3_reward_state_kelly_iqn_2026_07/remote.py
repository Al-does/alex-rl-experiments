"""Assign one of eight full conditions to each freshly provisioned Vast box."""

from __future__ import annotations

import argparse
import os
import re
import subprocess


CONDITION_MODULES = (
    "experiments.mess3_reward_state_kelly_iqn_2026_07.ppo_gamma_0.experiment",
    "experiments.mess3_reward_state_kelly_iqn_2026_07.ppo_gamma_099.experiment",
    "experiments.mess3_reward_state_kelly_iqn_2026_07.iqn_gamma_0.experiment",
    "experiments.mess3_reward_state_kelly_iqn_2026_07.iqn_gamma_099.experiment",
    "experiments.mess3_reward_state_kelly_iqn_2026_07.kelly_gamma_0.experiment",
    "experiments.mess3_reward_state_kelly_iqn_2026_07.kelly_gamma_099.experiment",
    "experiments.mess3_reward_state_kelly_iqn_2026_07.kelly_iqn_gamma_0.experiment",
    "experiments.mess3_reward_state_kelly_iqn_2026_07.kelly_iqn_gamma_099.experiment",
)


def assigned_index(label: str) -> int:
    """Convert the provisioner's one-based shot suffix to a condition index."""

    match = re.search(r"-(\d+)-[0-9a-f]{6}$", label)
    if match is None:
        raise ValueError(f"cannot extract Vast shot number from {label!r}")
    index = int(match.group(1)) - 1
    if not 0 <= index < len(CONDITION_MODULES):
        raise ValueError(f"Vast shot {index + 1} is outside the eight-arm battery")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.index is not None and args.offset != 0:
        raise ValueError("--index and --offset cannot be combined")
    index = (
        args.index
        if args.index is not None
        else (
            assigned_index(os.environ.get("VAST_INSTANCE_LABEL", ""))
            + args.offset
        )
    )
    if not 0 <= index < len(CONDITION_MODULES):
        raise ValueError(f"condition index must be in [0, {len(CONDITION_MODULES)})")
    subprocess.run(
        [
            "rl-harness",
            CONDITION_MODULES[index],
            "--seed",
            str(args.seed),
            "--hardware-profile",
            "cuda4090",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
