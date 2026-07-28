"""Launch one action-symmetry experiment on RunPod Flash.

This study requires an explicit ``cuda4090`` resource profile during remote
preflight: the default ``cuda4090_gpuinfer`` profile reserves 1.8 GPUs. The
profile describes a one-GPU layout, not an exact GPU model.

Examples:

    uv run python experiments/mess3_reward_state_action_symmetry_cycle_1/server_job.py \
      --condition variant_1 --dry-run

    uv run python experiments/mess3_reward_state_action_symmetry_cycle_1/server_job.py \
      --condition battery --yes
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence

STUDY = "mess3_reward_state_action_symmetry_cycle_1"
CONDITIONS = (
    "variant_1",
    "variant_2",
    "variant_3",
    "variant_2_entropy_anneal",
    "battery",
)
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARNESS = Path("/rl-harness")
DEFAULT_APP = "rlh-flash-mess3-action-symmetry"


def _git_sha(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()


def default_max_age(condition: str, *, smoke: bool) -> float:
    """Return a conservative execution ceiling in hours."""
    if smoke:
        # A local CPU variant smoke completes in under one minute. Allow extra
        # time for remote bootstrap, probes, and B2 upload without permitting a
        # stalled smoke to occupy a GPU for hours.
        return 20 / 60 if condition == "battery" else 10 / 60
    return 36.0 if condition == "battery" else 12.0


def experiment_module(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    return f"experiments.{STUDY}.{condition}.experiment"


def build_run_command(
    *,
    condition: str,
    run_name: str,
    seed: int,
    smoke: bool,
) -> str:
    argv = [
        "rl-harness",
        experiment_module(condition),
        "--seed",
        str(seed),
        # Both Flash preflight and runtime see the one-GPU-safe profile.
        "--hardware",
        "cuda4090",
        "--upload-artifacts",
        "--run-id",
        run_name,
    ]
    if smoke:
        argv.append("--smoke")
    return shlex.join(argv)


def flash_command(harness: Path, *args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--directory",
        str(harness),
        "--group",
        "flash",
        "python",
        "-m",
        "devops.flash.provision",
        *args,
    ]


def build_up_command(
    *,
    harness: Path,
    endpoint_id: str,
    condition: str,
    run_name: str,
    seed: int,
    smoke: bool,
    experiment_ref: str,
    library_ref: str,
    max_age: float,
    queue_timeout: float,
    max_price: float,
    max_estimated_cost: float,
    progress_interval: float,
    no_progress_timeout: float,
    dry_run: bool,
) -> list[str]:
    argv = flash_command(
        harness,
        "up",
        "--endpoint-id",
        endpoint_id,
        "--experiment-ref",
        experiment_ref,
        "--library-ref",
        library_ref,
        "--run-name",
        run_name,
        "--run",
        build_run_command(
            condition=condition,
            run_name=run_name,
            seed=seed,
            smoke=smoke,
        ),
        "--max-age",
        str(max_age),
        "--queue-timeout",
        str(queue_timeout),
        "--progress-interval",
        str(progress_interval),
        "--no-progress-timeout",
        str(no_progress_timeout),
        "--max-price",
        str(max_price),
        "--max-estimated-cost",
        str(max_estimated_cost),
        "--forward-b2",
        "--self-destruct",
    )
    argv.append("--dry-run" if dry_run else "--yes")
    return argv


def _parse_endpoint_id(output: str) -> str | None:
    for line in output.splitlines():
        if "verified endpoint " in line:
            return line.split("verified endpoint ", 1)[1].split(":", 1)[0].strip()
    return None


def deploy(
    *,
    harness: Path,
    app: str,
    dry_run: bool,
) -> str | None:
    argv = flash_command(
        harness,
        "deploy",
        "--app",
        app,
        "--environment",
        "production",
        "--max-workers",
        "1",
        "--dry-run" if dry_run else "--yes",
    )
    print("+ " + shlex.join(argv), flush=True)
    completed = subprocess.run(argv, check=False, text=True, capture_output=True)
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode:
        raise RuntimeError(f"Flash deploy failed with code {completed.returncode}")
    if dry_run:
        return None
    endpoint_id = _parse_endpoint_id(completed.stdout)
    if not endpoint_id:
        raise RuntimeError("Flash deploy succeeded but returned no endpoint ID")
    return endpoint_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, default="variant_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--app", default=DEFAULT_APP)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--experiment-ref")
    parser.add_argument("--library-ref")
    parser.add_argument("--max-age", type=float)
    parser.add_argument("--queue-timeout", type=float, default=30.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--no-progress-timeout", type=float, default=15.0)
    parser.add_argument("--max-price", type=float, default=1.25)
    parser.add_argument("--max-estimated-cost", type=float, default=5.0)
    parser.add_argument("--full", dest="smoke", action="store_false")
    parser.set_defaults(smoke=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    harness = args.harness.expanduser().resolve()
    if not harness.is_dir():
        raise SystemExit(f"harness checkout missing: {harness}")

    run_name = args.run_name or (
        f"{STUDY}-{args.condition}-seed{args.seed}-"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    experiment_ref = args.experiment_ref or _git_sha(EXPERIMENT_ROOT)
    library_ref = args.library_ref or _git_sha(harness)
    max_age = args.max_age or default_max_age(args.condition, smoke=args.smoke)
    endpoint_id = args.endpoint_id
    if endpoint_id is None:
        endpoint_id = deploy(
            harness=harness,
            app=args.app,
            dry_run=args.dry_run,
        )

    up = build_up_command(
        harness=harness,
        endpoint_id=endpoint_id or "DEPLOYED_ENDPOINT_ID",
        condition=args.condition,
        run_name=run_name,
        seed=args.seed,
        smoke=args.smoke,
        experiment_ref=experiment_ref,
        library_ref=library_ref,
        max_age=max_age,
        queue_timeout=args.queue_timeout,
        max_price=args.max_price,
        max_estimated_cost=args.max_estimated_cost,
        progress_interval=args.progress_interval,
        no_progress_timeout=args.no_progress_timeout,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "condition": args.condition,
                "smoke": args.smoke,
                "max_age_hours": max_age,
                "hardware": "cuda4090",
                "experiment_ref": experiment_ref,
                "library_ref": library_ref,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("+ " + shlex.join(up), flush=True)
    if args.dry_run and endpoint_id is None:
        print(
            "Deploy dry-run completed; rerun with --yes to obtain an endpoint "
            "and execute the printed preflight.",
            flush=True,
        )
        return 0
    return subprocess.run(up, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
