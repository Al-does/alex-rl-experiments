"""Rent three on-demand vast.ai boxes — one reward arm each — at 10M steps.

Dry-run first, then live:

  uv run python experiments/two_factor_reward_state_PPO_cycle_1/launch_vast_arms.py --dry-run
  uv run python experiments/two_factor_reward_state_PPO_cycle_1/launch_vast_arms.py --yes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

STUDY = "two_factor_reward_state_PPO_cycle_1"
CONDITIONS = ("reward_both", "reward_factor_1", "reward_factor_2")
DEFAULT_SEED = 42
DEFAULT_MAX_AGE_HOURS = 18.0
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_harness(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for candidate in (
        EXPERIMENT_ROOT.parent / "rl-harness",
        Path("/rl-harness"),
        Path("/agent/repos/rl-harness"),
    ):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("could not locate rl-harness checkout")


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


def build_run_command(condition: str, *, seed: int) -> str:
    module = (
        f"experiments.{STUDY}.{condition}.experiment "
        f"--seed {seed} --hardware auto --upload-artifacts"
    )
    return f"rl-harness {module}"


def provision_arm(
    *,
    harness: Path,
    experiment_repo: Path,
    condition: str,
    seed: int,
    experiment_ref: str,
    library_ref: str,
    max_age: float,
    max_price: float,
    dry_run: bool,
    yes: bool,
    exclude_machines: Sequence[int] = (),
) -> int:
    run_name = f"{STUDY}-{condition}-seed{seed}-10m"
    argv = _vast_cmd(
        harness,
        "up",
        "-n",
        "1",
        "--mode",
        "ondemand",
        "--experiment-repo",
        str(experiment_repo),
        "--commit",
        experiment_ref,
        "--library-commit",
        library_ref,
        "--run",
        build_run_command(condition, seed=seed),
        "--run-name",
        run_name,
        "--max-age",
        str(max_age),
        "--max-price",
        str(max_price),
        "--forward-b2",
        "--self-destruct",
        "--regions",
        "US,CA",
        "--no-open",
    )
    if exclude_machines:
        argv.append("--exclude-machine")
        argv.extend(str(int(machine_id)) for machine_id in exclude_machines)
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
        description="Launch three on-demand vast boxes for the 10M-step PPO study."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--harness", type=Path, default=None)
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
        / "vast_10m_state.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.yes:
        raise SystemExit("pass only one of --dry-run / --yes")
    if not args.dry_run and not args.yes:
        raise SystemExit("pass --dry-run or --yes")

    harness = _resolve_harness(args.harness)
    experiment_repo = args.experiment_repo.expanduser().resolve()
    experiment_ref = args.experiment_ref or _git_sha(experiment_repo)
    library_ref = args.library_ref or _git_sha(harness)
    plan = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": list(args.conditions),
        "seed": args.seed,
        "total_env_steps": 10_000_000,
        "experiment_ref": experiment_ref,
        "library_ref": library_ref,
        "max_age_hours": args.max_age,
        "exclude_machines": list(args.exclude_machine),
        "run_commands": {
            condition: build_run_command(condition, seed=args.seed)
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
            seed=args.seed,
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
