"""List or launch the 27-run pure-JAX guess-or-pass transfer campaign."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from experiments.pusher_b10_guess_or_pass_2026_10.campaign import (
    LEAF_COSTS,
    SOURCES,
    RunSpec,
)
from experiments.pusher_b10_guess_or_pass_2026_10.campaign import (
    campaign_runs as _legacy_campaign_runs,
)

ROOT = Path(__file__).resolve().parents[2]
TRANSFER_MODULE = "experiments.pusher_b10_guess_or_pass_jax_2026_10.transfer"
LEAVES = tuple(leaf for leaf, _ in LEAF_COSTS)
SOURCE_ORDER = SOURCES
ORDER = ("scratch", "full", "frozen")


def campaign_runs(frozen_cost: float = 2.0) -> tuple[RunSpec, ...]:
    legacy_runs = _legacy_campaign_runs(frozen_cost)
    return tuple(
        replace(
            run,
            run_name=run.run_name.replace(
                "guess-or-pass-ppo-ce-", "guess-or-pass-jax-"
            ),
        )
        for run in legacy_runs
    )


def _command(
    spec: RunSpec,
    *,
    hardware: str,
    upload_artifacts: bool | None,
    smoke: bool,
) -> list[str]:
    module = (
        "experiments.pusher_b10_guess_or_pass_jax_2026_10."
        f"{spec.leaf}.experiment"
    )
    if spec.source is None:
        command = [sys.executable, "-m", "harness.cli", module]
    else:
        command = [
            sys.executable,
            "-m",
            TRANSFER_MODULE,
            module,
            "--source",
            spec.source,
            "--trunk",
            str(spec.trunk),
        ]
    command.extend(
        [
            "--seed",
            str(spec.seed),
            "--run-id",
            spec.run_name,
            "--hardware-profile",
            hardware,
        ]
    )
    if upload_artifacts is not None:
        command.append(
            "--upload-artifacts" if upload_artifacts else "--no-upload-artifacts"
        )
    if smoke:
        command.append("--smoke")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--run-index", type=int, nargs="+", metavar="N")
    action.add_argument("--all", action="store_true")
    parser.add_argument(
        "--frozen-cost",
        type=float,
        default=2.0,
        choices=(2.0, 1.5, 1.0),
    )
    parser.add_argument("--hardware-profile", "--hardware", default="auto")
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runs = campaign_runs(args.frozen_cost)
    commands = [
        _command(
            run,
            hardware=args.hardware_profile,
            upload_artifacts=args.upload_artifacts,
            smoke=args.smoke,
        )
        for run in runs
    ]
    if args.list:
        for command in commands:
            print(shlex.join(command))
        return 0
    indices = (
        list(args.run_index)
        if args.run_index is not None
        else [*range(21), *range(21, 27)]
    )
    invalid_indices = [index for index in indices if index < 0 or index >= len(runs)]
    if invalid_indices:
        parser.error(
            f"--run-index values must be between 0 and {len(runs) - 1}; "
            f"invalid: {', '.join(map(str, invalid_indices))}"
        )
    results = []
    for index in indices:
        result = subprocess.run(commands[index], cwd=ROOT, check=False)
        results.append((index, result.returncode))
    print(
        "Campaign exit codes: "
        + ", ".join(f"{index}={code}" for index, code in results)
    )
    return 0 if all(code == 0 for _, code in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
