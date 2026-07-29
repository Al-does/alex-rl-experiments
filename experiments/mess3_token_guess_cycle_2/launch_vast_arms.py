"""Rent three vast.ai boxes — one arm each — for the 15-seed 0.66M campaign.

Dry-run first, then live:

  uv run --directory /rl-harness --group devops \\
    python experiments/mess3_token_guess_cycle_2/launch_vast_arms.py --dry-run

  # same with --yes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ARMS = ("decoupled_kelly", "predictive_loss", "ppo")
DEFAULT_SEEDS = tuple(range(42, 57))
STUDY = "mess3_token_guess_cycle_2"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARNESS = Path("/rl-harness")
# 15 seeds × ~1h with headroom for slow hosts / probe time / sync.
DEFAULT_MAX_AGE_HOURS = 30.0


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()


def _vast_cmd(harness: Path, *args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--directory",
        str(harness),
        "--group",
        "devops",
        "python",
        "-m",
        "devops.vast.provision",
        *args,
    ]


def build_run_command(condition: str, seeds: Sequence[int]) -> str:
    seed_args = " ".join(str(seed) for seed in seeds)
    return (
        f"python experiments/{STUDY}/arm_queue.py "
        f"--condition {condition} "
        f"--seeds {seed_args} "
        f"--max-env-steps 700000 "
        f"--upload-artifacts "
        f"--push-each"
    )


def provision_arm(
    *,
    harness: Path,
    experiment_repo: Path,
    condition: str,
    seeds: Sequence[int],
    experiment_ref: str,
    library_ref: str,
    max_age: float,
    max_price: float,
    dry_run: bool,
    yes: bool,
    exclude_machines: Sequence[int] = (),
) -> int:
    run_name = f"{STUDY}-{condition}-15seed-0p66m"
    argv = _vast_cmd(
        harness,
        "up",
        "-n",
        "1",
        "--experiment-repo",
        str(experiment_repo),
        "--commit",
        experiment_ref,
        "--library-commit",
        library_ref,
        "--run",
        build_run_command(condition, seeds),
        "--run-name",
        run_name,
        "--max-age",
        str(max_age),
        "--max-price",
        str(max_price),
        "--forward-b2",
        "--self-destruct",
        "--teardown-on-error",
        "--regions",
        "US,CA",
        "--no-open",
    )
    if exclude_machines:
        argv.append("--exclude-machine")
        argv.extend(str(int(mid)) for mid in exclude_machines)
    if dry_run:
        argv.append("--dry-run")
    elif yes:
        argv.append("--yes")
    else:
        raise ValueError("pass --dry-run or --yes")
    print("+ " + " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False, text=True)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch three vast boxes for the 15-seed 0.66M campaign."
    )
    parser.add_argument("--conditions", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--experiment-repo", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--experiment-ref", default=None)
    parser.add_argument("--library-ref", default=None)
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument("--max-price", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--exclude-machine",
        type=int,
        nargs="+",
        default=[],
        help="vast machine_ids to skip (bad SSH / prior readiness failures)",
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=EXPERIMENT_ROOT
        / "experiments"
        / STUDY
        / "artifacts"
        / "vast_15seed_0p66m_state.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.yes:
        raise SystemExit("pass only one of --dry-run / --yes")
    if not args.dry_run and not args.yes:
        raise SystemExit("pass --dry-run or --yes")

    harness = args.harness.expanduser().resolve()
    experiment_repo = args.experiment_repo.expanduser().resolve()
    experiment_ref = args.experiment_ref or _git_sha(experiment_repo)
    library_ref = args.library_ref or _git_sha(harness)
    plan = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": list(args.conditions),
        "seeds": list(args.seeds),
        "experiment_ref": experiment_ref,
        "library_ref": library_ref,
        "max_age_hours": args.max_age,
        "run_commands": {
            condition: build_run_command(condition, args.seeds)
            for condition in args.conditions
        },
    }
    print(json.dumps(plan, indent=2), flush=True)

    results: dict[str, int] = {}
    for condition in args.conditions:
        code = provision_arm(
            harness=harness,
            experiment_repo=experiment_repo,
            condition=condition,
            seeds=args.seeds,
            experiment_ref=experiment_ref,
            library_ref=library_ref,
            max_age=args.max_age,
            max_price=args.max_price,
            dry_run=args.dry_run,
            yes=args.yes,
            exclude_machines=args.exclude_machine,
        )
        results[condition] = code
        if code != 0:
            print(f"[launch] {condition} provision exited {code}", flush=True)

    plan["provision_exit_codes"] = results
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.state_out}", flush=True)
    return 1 if any(code != 0 for code in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
