"""Run cycle-4 REINFORCE arms across seeds on one GPU box."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

from experiments.two_factor_reward_state_PPO_cycle_2.task import CONDITIONS
from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    write_budget_spec,
)
from harness.cli import execute_experiment, load_experiment, make_run_context

STUDY = "two_factor_reward_state_REINFORCE_cycle_4"


def _instance_id() -> str | None:
    path = Path("/root/vast_instance_id")
    if path.is_file():
        return path.read_text().strip() or None
    return os.environ.get("VAST_INSTANCE_ID")


def _push_results(run_name: str) -> bool:
    try:
        from devops.vast.self_destruct import push_results
    except ImportError as error:
        print(f"[seed_queue] push unavailable: {error}", flush=True)
        return False
    return bool(
        push_results(
            branch=os.environ.get("VAST_RESULTS_BRANCH", "results"),
            run_name=run_name,
            instance_id=_instance_id(),
        )
    )


def _run_one(
    *,
    condition: str,
    seed: int,
    target_agent_steps: int,
    hardware_profile: str,
    upload_artifacts: bool,
    push_each: bool,
) -> int:
    module = f"experiments.{STUDY}.{condition}.experiment"
    run_id = f"{STUDY}-{condition}-seed{seed}-{target_agent_steps // 1_000_000}m"
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=False,
        hardware_profile=hardware_profile,
    )
    write_budget_spec(context, target_agent_steps)
    print(
        f"[seed_queue] start condition={condition} seed={seed} "
        f"target_steps={target_agent_steps} run_id={run_id}",
        flush=True,
    )
    started = time.time()
    try:
        execute_experiment(
            experiment,
            context,
            command=[
                "python",
                "-m",
                f"experiments.{STUDY}.seed_queue",
                "--condition",
                condition,
                "--seeds",
                str(seed),
                "--target-agent-steps",
                str(target_agent_steps),
            ],
            runtime_overrides={
                "condition": condition,
                "seed": seed,
                "target_agent_steps": target_agent_steps,
                "hardware_profile": hardware_profile,
                "upload_artifacts": upload_artifacts,
            },
            upload_artifacts=upload_artifacts,
        )
    except Exception as error:  # noqa: BLE001 - keep the queue going
        print(f"[seed_queue] FAILED {run_id}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1

    print(
        f"[seed_queue] done {run_id} elapsed_s={time.time() - started:.1f}",
        flush=True,
    )
    if push_each and not _push_results(run_id):
        print(f"[seed_queue] push failed for {run_id}", flush=True)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential cycle-4 REINFORCE runner for one or more seeds."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=CONDITIONS,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--target-agent-steps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--hardware-profile",
        default="cuda4090",
    )
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--push-each",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = 0
    for seed in args.seeds:
        failures += _run_one(
            condition=args.condition,
            seed=seed,
            target_agent_steps=args.target_agent_steps,
            hardware_profile=args.hardware_profile,
            upload_artifacts=args.upload_artifacts,
            push_each=args.push_each,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
