"""Run a user-provisioned box plan. Does not create or rent instances."""
import argparse
import subprocess
import sys

from .data import LADDER, RR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box", type=int, choices=range(1, 7), required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--publish-and-destroy", action="store_true")
    args = p.parse_args()
    plans = [(LADDER, f"variant_{v}") for v in (1, 2, 3)] + [(RR, c) for c in ("reward_both", "reward_factor_1")]
    selected = plans if args.box == 6 else [plans[args.box-1]]
    for i, (study, condition) in enumerate(selected):
        mode = "control" if args.box == 6 else "policy"
        command = [sys.executable, "-m", f"experiments.{study}.analysis_gbs.run",
                   "--condition", condition, "--seeds", "42", "43", "44",
                   "--run-name", f"{args.run_prefix}-{condition}-{mode}"]
        if args.box == 6:
            command.append("--control")
        if args.publish_and_destroy and i == len(selected)-1:
            command.append("--publish-and-destroy")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
