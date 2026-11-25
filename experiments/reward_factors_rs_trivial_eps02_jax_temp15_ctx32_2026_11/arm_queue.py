"""Run this study's matched arms sequentially for one seed on one GPU box.

Compact results are pushed after every arm so a later failure cannot lose
finished arms. Example (inside the box's activated .venv):

  python -m experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.arm_queue --seed 42
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.shared import ARMS, STUDY
from harness.cli import execute_experiment, load_experiment, make_run_context

DEFAULT_ORDER = ("separate_critic", "shared_critic", "actor_only")


def _instance_id():
    path = Path("/root/vast_instance_id")
    return (path.read_text().strip() or None) if path.is_file() else os.environ.get("VAST_INSTANCE_ID")


def _push(run_name):
    try:
        from devops.vast.self_destruct import push_results
    except ImportError as error:
        print(f"[arm_queue] push unavailable: {error}", flush=True)
        return False
    return bool(push_results(branch=os.environ.get("VAST_RESULTS_BRANCH", "results"),
                             run_name=run_name, instance_id=_instance_id()))


def run_arm(arm, seed, *, hardware_profile, upload_artifacts, push_each):
    module = f"experiments.{STUDY}.{arm}_transformer.experiment"
    run_id = f"{arm}-seed{seed}"
    experiment = load_experiment(module)
    context = make_run_context(experiment, seed=seed, run_id=run_id, smoke=False,
                               hardware_profile=hardware_profile)
    print(f"[arm_queue] start arm={arm} seed={seed}", flush=True)
    started = time.time()
    try:
        execute_experiment(experiment, context,
                           command=["arm_queue", "--arm", arm, "--seed", str(seed)],
                           runtime_overrides={"hardware_profile": hardware_profile, "seed": seed,
                                              "smoke": False, "upload_artifacts": upload_artifacts},
                           upload_artifacts=upload_artifacts)
    except Exception as error:  # noqa: BLE001 - keep the queue going
        print(f"[arm_queue] FAILED arm={arm} seed={seed}: {error!r}", flush=True)
        if push_each:
            _push(f"{run_id}-failed")
        return 1
    print(f"[arm_queue] done arm={arm} seed={seed} elapsed_s={time.time() - started:.1f}", flush=True)
    if push_each and not _push(run_id):
        print(f"[arm_queue] push failed arm={arm} seed={seed}", flush=True)
        return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Sequential arms of {STUDY} for one seed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ORDER))
    parser.add_argument("--hardware-profile", default="cuda4090")
    parser.add_argument("--upload-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-each", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    codes = {arm: run_arm(arm, args.seed, hardware_profile=args.hardware_profile,
                          upload_artifacts=args.upload_artifacts, push_each=args.push_each)
             for arm in args.arms}
    print({"seed": args.seed, "arm_exit_codes": codes}, flush=True)
    return 1 if any(codes.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
