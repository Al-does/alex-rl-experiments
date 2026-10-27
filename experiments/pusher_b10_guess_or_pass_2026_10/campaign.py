"""List or launch one run from the PPO versus PPO+CE transfer campaign."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANSFER_MODULE = "experiments.pusher_b10_guess_or_pass_2026_10.transfer"
LEAF_COSTS = (
    ("ppo_c2", 2.0),
    ("ppo_c1p5", 1.5),
    ("ppo_c1", 1.0),
)
SOURCES = (
    "ppo_iter4",
    "ppo_iter143",
    "ppo_ce_iter4",
    "ppo_ce_iter211",
    "ppo_iter8",
    "ppo_ce_iter8",
)


@dataclass(frozen=True)
class RunSpec:
    leaf: str
    source: str | None
    trunk: str | None
    cost: float
    seed: int
    run_name: str


def _run_spec(
    leaf: str,
    cost: float,
    source: str | None = None,
    trunk: str | None = None,
) -> RunSpec:
    source_name = source or "scratch"
    trunk_name = trunk or "scratch"
    return RunSpec(
        leaf=leaf,
        source=source,
        trunk=trunk,
        cost=cost,
        seed=42,
        run_name=(
            f"guess-or-pass-ppo-ce-seed42-{leaf}-{source_name}-{trunk_name}"
        ),
    )


def campaign_runs(frozen_cost: float = 2.0) -> tuple[RunSpec, ...]:
    frozen_leaf_cost = next(
        (
            (leaf, cost)
            for leaf, cost in LEAF_COSTS
            if cost == frozen_cost
        ),
        None,
    )
    if frozen_leaf_cost is None:
        raise ValueError(f"unsupported frozen cost: {frozen_cost}")
    frozen_leaf, frozen_cost = frozen_leaf_cost
    return (
        *(_run_spec(leaf, cost) for leaf, cost in LEAF_COSTS),
        *(
            _run_spec(leaf, cost, source, "full")
            for source in SOURCES
            for leaf, cost in LEAF_COSTS
        ),
        *(
            _run_spec(frozen_leaf, frozen_cost, source, "frozen")
            for source in SOURCES
        ),
    )


def _command(
    spec: RunSpec,
    *,
    hardware: str,
    upload_artifacts: bool | None,
    smoke: bool,
) -> list[str]:
    module = (
        "experiments.pusher_b10_guess_or_pass_2026_10."
        f"{spec.leaf}.experiment"
    )
    if spec.source is None:
        command = [sys.executable, "-m", "harness.cli", module]
    else:
        command = [
            sys.executable,
            "-m",
            TRANSFER_MODULE,
            spec.leaf,
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
            spec,
            hardware=args.hardware_profile,
            upload_artifacts=args.upload_artifacts,
            smoke=args.smoke,
        )
        for spec in runs
    ]
    if args.list:
        for command in commands:
            print(shlex.join(["python", *command[1:]]))
        return 0
    invalid_indices = [
        index
        for index in args.run_index
        if index < 0 or index >= len(commands)
    ]
    if invalid_indices:
        parser.error(
            f"--run-index values must be between 0 and {len(commands) - 1}; "
            f"invalid: {', '.join(map(str, invalid_indices))}"
        )
    exit_codes = []
    for index in args.run_index:
        result = subprocess.run(commands[index], cwd=ROOT, check=False)
        exit_codes.append((index, result.returncode))
    print(
        "Campaign exit codes: "
        + ", ".join(f"{index}={code}" for index, code in exit_codes)
    )
    return 0 if all(code == 0 for _, code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
