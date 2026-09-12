"""Run all three action-symmetry variants of one component set on one GPU box.

Designed for vast.ai: sequential variants at the full budget, with a compact
results push after each so a mid-queue failure still lands completed runs on
the launch branch.

Example (on a provisioned box, already in the activated .venv):

  python -m experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.variant_queue \
    --set default --seed 42

  python -m experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.variant_queue \
    --set alpha095 --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence

from harness.cli import execute_experiment, load_experiment, make_run_context

STUDY = "nonergodic_mess3_reward_state_action_symmetry_cycle_5"
ARM_SETS = {
    "default": ("variant_1", "variant_2", "variant_3"),
    "alpha095": (
        "alpha095.variant_1",
        "alpha095.variant_2",
        "alpha095.variant_3",
    ),
}


def _instance_id() -> str | None:
    path = "/root/vast_instance_id"
    if os.path.isfile(path):
        text = open(path).read().strip()
        return text or None
    return os.environ.get("VAST_INSTANCE_ID")


def _push_results(run_name: str) -> bool:
    """Push compact experiments/ results; no-op success when nothing staged."""

    try:
        from devops.vast.self_destruct import push_results
    except ImportError as error:
        print(f"[variant_queue] push unavailable: {error}", flush=True)
        return False
    return bool(
        push_results(
            run_name=run_name,
            instance_id=_instance_id(),
        )
    )


def _run_one(
    *,
    arm: str,
    arm_set: str,
    seed: int,
    hardware_profile: str,
    upload_artifacts: bool,
    push_each: bool,
) -> int:
    module = f"experiments.{STUDY}.{arm}.experiment"
    slug = arm.replace(".", "-")
    run_id = f"nem3-rsa-c5-{slug}-s{seed}"
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=False,
        hardware_profile=hardware_profile,
    )
    print(
        f"[variant_queue] start arm={arm} seed={seed} run_id={run_id}",
        flush=True,
    )
    started = time.time()
    try:
        execute_experiment(
            experiment,
            context,
            command=[
                "variant_queue",
                "--set",
                arm_set,
                "--arm",
                arm,
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
    except Exception as error:  # keep queue going / surface failure
        print(f"[variant_queue] FAILED arm={arm}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1

    elapsed = time.time() - started
    print(f"[variant_queue] done arm={arm} elapsed_s={elapsed:.1f}", flush=True)
    if push_each:
        ok = _push_results(run_id)
        print(f"[variant_queue] push_each arm={arm} ok={ok}", flush=True)
        if not ok:
            return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="arm_set",
        required=True,
        choices=list(ARM_SETS),
        help="component-parameter set: default or alpha095",
    )
    parser.add_argument(
        "--arm",
        action="append",
        help=(
            "run only these arms from the set (repeatable; default: all "
            "three variants in order)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hardware-profile",
        "--hardware",
        default="auto",
        help="harness hardware profile name (default: auto)",
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
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.arm is not None:
        available = set(ARM_SETS[args.arm_set])
        unknown = [arm for arm in args.arm if arm not in available]
        if unknown:
            parser.error(f"unknown arms for set {args.arm_set}: {unknown}")
        arms = tuple(args.arm)
    else:
        arms = ARM_SETS[args.arm_set]
    codes: dict[str, int] = {}
    for arm in arms:
        codes[arm] = _run_one(
            arm=arm,
            arm_set=args.arm_set,
            seed=args.seed,
            hardware_profile=args.hardware_profile,
            upload_artifacts=args.upload_artifacts,
            push_each=args.push_each,
        )
        if codes[arm] and args.fail_fast:
            break
    print({"arm_exit_codes": codes}, flush=True)
    return 1 if any(codes.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
