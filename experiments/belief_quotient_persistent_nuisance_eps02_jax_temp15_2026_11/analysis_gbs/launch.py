"""Run a user-provisioned box plan. Does not create or rent instances."""
import argparse
import subprocess
import sys

from .data import LADDER, RR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box", type=int, choices=range(1, 7), required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("run_args", nargs=argparse.REMAINDER, help="run overrides after -- (e.g. -- --smoke --updates 10)")
    args = p.parse_args()
    ladder = [(LADDER, f"variant_{v}") for v in (1, 2, 3)]
    rr = [(RR, c) for c in ("reward_both", "reward_factor_1")]
    # Balanced plan: ladder boxes run policy then control; box 6 runs both
    # Reward Relevance controls with a single head width.
    boxes = {i + 1: [(*plan, "policy", []), (*plan, "control", [])] for i, plan in enumerate(ladder)}
    boxes.update({4: [(*rr[0], "policy", [])], 5: [(*rr[1], "policy", [])],
                  6: [(*plan, "control", ["--widths", "128"]) for plan in rr]})
    extra = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args
    for study, condition, mode, defaults in boxes[args.box]:
        command = [sys.executable, "-m", f"experiments.{study}.analysis_gbs.run",
                   "--condition", condition, "--seeds", "42", "43", "44",
                   "--run-name", f"{args.run_prefix}-{condition}-{mode}", *defaults]
        if mode == "control":
            command.append("--control")
        command.extend(extra)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
