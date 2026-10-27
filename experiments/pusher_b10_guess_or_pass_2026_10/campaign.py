"""List or launch one run from the PPO versus PPO+CE transfer campaign."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANSFER_MODULE = "experiments.pusher_b10_guess_or_pass_2026_10.transfer"
LEAF_COSTS = (("ppo_c2", 2.0), ("ppo_c1", 1.0))
SOURCES = ("ppo_iter4", "ppo_iter143", "ppo_ce_iter4", "ppo_ce_iter211")


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


CAMPAIGN_RUNS = (
    *(_run_spec(leaf, cost) for leaf, cost in LEAF_COSTS),
    *(
        _run_spec(leaf, cost, source, trunk)
        for source in SOURCES
        for trunk in ("full", "frozen")
        for leaf, cost in LEAF_COSTS
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
        command = ["uv", "run", "rl-harness", module]
    else:
        command = [
            "uv",
            "run",
            "python",
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
    action.add_argument("--run-index", type=int, metavar="N")
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
    commands = [
        _command(
            spec,
            hardware=args.hardware_profile,
            upload_artifacts=args.upload_artifacts,
            smoke=args.smoke,
        )
        for spec in CAMPAIGN_RUNS
    ]
    if args.list:
        for command in commands:
            print(shlex.join(command))
        return 0
    if args.run_index < 0 or args.run_index >= len(commands):
        parser.error(f"--run-index must be between 0 and {len(commands) - 1}")
    result = subprocess.run(commands[args.run_index], cwd=ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
