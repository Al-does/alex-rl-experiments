"""Run memoryless/GRU battery leaves across seeds on one GPU box.

Each leaf runs at its source campaign's launch budget: 10M agent steps for
the Belief-Quotient Ladder (cycle-7 ``seed_queue``) and 30M for Reward
Relevance (cycle-4 ``seed_queue``).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

from experiments.mess3_reward_state_action_symmetry_cycle_7 import shared as ladder_source
from experiments.two_factor_reward_state_REINFORCE_cycle_4 import shared as reward_source
from harness.cli import execute_experiment, load_experiment, make_run_context

ARCHITECTURES = ("memoryless", "gru", "gru92")
LADDER_LEAVES = ("variant_1", "variant_2", "variant_3")
REWARD_LEAVES = ("reward_both", "reward_factor_1")
LADDER_TARGET_AGENT_STEPS = 10_000_000
REWARD_TARGET_AGENT_STEPS = 30_000_000


def _leaves() -> dict[str, tuple[int, object]]:
    leaves = {}
    for architecture in ARCHITECTURES:
        for leaf in LADDER_LEAVES:
            leaves[f"belief_quotient_ladder_{architecture}_2026_11.{leaf}"] = (
                LADDER_TARGET_AGENT_STEPS, ladder_source.write_budget_spec)
        for leaf in REWARD_LEAVES:
            leaves[f"reward_relevance_{architecture}_2026_11.{leaf}"] = (
                REWARD_TARGET_AGENT_STEPS, reward_source.write_budget_spec)
    return leaves


LEAVES = _leaves()


def _instance_id() -> str | None:
    path = Path("/root/vast_instance_id")
    if path.is_file():
        return path.read_text().strip() or None
    return os.environ.get("VAST_INSTANCE_ID")


def _push_results(run_name: str) -> bool:
    try:
        from devops.vast.self_destruct import push_results
    except ImportError as error:
        print(f"[queue] push unavailable: {error}", flush=True)
        return False
    return bool(push_results(
        branch=os.environ.get("VAST_RESULTS_BRANCH", "results"),
        run_name=run_name,
        instance_id=_instance_id(),
    ))


def _run_one(*, leaf: str, seed: int, hardware_profile: str,
             upload_artifacts: bool, push_each: bool, smoke: bool) -> int:
    target_agent_steps, write_budget_spec = LEAVES[leaf]
    study, condition = leaf.split(".")
    run_id = f"{study}-{condition}-seed{seed}-{target_agent_steps // 1_000_000}m"
    experiment = load_experiment(f"experiments.{leaf}.experiment")
    context = make_run_context(experiment, seed=seed, run_id=run_id,
                               smoke=smoke, hardware_profile=hardware_profile)
    write_budget_spec(context, target_agent_steps)
    print(f"[queue] start leaf={leaf} seed={seed} "
          f"target_steps={target_agent_steps} run_id={run_id}", flush=True)
    started = time.time()
    try:
        execute_experiment(
            experiment, context,
            command=["python", "-m", "experiments.policy_baselines_2026_11_queue",
                     "--leaves", leaf, "--seeds", str(seed)],
            runtime_overrides={
                "leaf": leaf, "seed": seed,
                "target_agent_steps": target_agent_steps,
                "hardware_profile": hardware_profile,
                "upload_artifacts": upload_artifacts,
            },
            upload_artifacts=upload_artifacts,
        )
    except Exception as error:  # noqa: BLE001 - keep the queue going
        print(f"[queue] FAILED {run_id}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1
    print(f"[queue] done {run_id} elapsed_s={time.time() - started:.1f}", flush=True)
    if push_each and not _push_results(run_id):
        print(f"[queue] push failed for {run_id}", flush=True)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leaves", nargs="+", default=list(LEAVES), choices=list(LEAVES),
                        metavar="STUDY.LEAF", help="default: all 15 leaves")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--hardware-profile", default="cuda4090")
    parser.add_argument("--upload-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-each", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = 0
    for seed in args.seeds:
        for leaf in args.leaves:
            failures += _run_one(leaf=leaf, seed=seed,
                                 hardware_profile=args.hardware_profile,
                                 upload_artifacts=args.upload_artifacts,
                                 push_each=args.push_each, smoke=args.smoke)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
