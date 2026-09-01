"""Run all four cycle-2 arms for one seed on a single GPU box.

Runs, in order:

1. SAC ``reward_both``
2. SAC ``reward_factor_1``
3. PPO ``reward_both``
4. PPO ``reward_factor_1``

Example (on a provisioned box, inside the activated ``.venv``):

  python -m experiments.two_factor_reward_state_cycle_2.seed_queue --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

from harness.cli import execute_experiment, load_experiment, make_run_context

CONDITIONS = ("reward_both", "reward_factor_1")
ALGORITHMS = ("SAC", "PPO")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


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
    algorithm: str,
    condition: str,
    seed: int,
    hardware_profile: str,
    upload_artifacts: bool,
    push_each: bool,
) -> int:
    study = f"two_factor_reward_state_{algorithm}_cycle_2"
    module = f"experiments.{study}.{condition}.experiment"
    run_id = f"{study}-{condition}-seed{seed}"
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=False,
        hardware_profile=hardware_profile,
    )
    print(
        f"[seed_queue] start algorithm={algorithm} condition={condition} "
        f"seed={seed} run_id={run_id}",
        flush=True,
    )
    started = time.time()
    try:
        execute_experiment(
            experiment,
            context,
            command=[
                "seed_queue",
                "--algorithm",
                algorithm,
                "--condition",
                condition,
                "--seed",
                str(seed),
            ],
            runtime_overrides={
                "hardware_profile": hardware_profile,
                "seed": seed,
                "smoke": False,
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
        description=(
            "Sequential four-arm runner (2 SAC + 2 PPO) for one cycle-2 seed."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hardware-profile", default="cuda4090")
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--push-each",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="push compact results after each arm (default: on)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop the queue on the first arm failure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "seed": args.seed,
        "algorithms": list(ALGORITHMS),
        "conditions": list(CONDITIONS),
        "hardware_profile": args.hardware_profile,
        "upload_artifacts": args.upload_artifacts,
        "push_each": args.push_each,
    }
    print(plan, flush=True)
    codes: dict[str, int] = {}
    for algorithm in ALGORITHMS:
        for condition in CONDITIONS:
            key = f"{algorithm}:{condition}"
            codes[key] = _run_one(
                algorithm=algorithm,
                condition=condition,
                seed=args.seed,
                hardware_profile=args.hardware_profile,
                upload_artifacts=args.upload_artifacts,
                push_each=args.push_each,
            )
            if codes[key] != 0 and args.fail_fast:
                print({"arm_exit_codes": codes}, flush=True)
                return 1
    failed = [key for key, code in codes.items() if code != 0]
    print({"arm_exit_codes": codes, "failed": failed}, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
