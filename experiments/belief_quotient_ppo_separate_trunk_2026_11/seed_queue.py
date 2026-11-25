"""Run all matched arms sequentially on one seed's GPU."""
import argparse
import subprocess

from .shared import STUDY

LEAVES = ("trivial_shared", "trivial_separate", "full_state_shared",
          "full_state_separate", "trivial_shared_wide", "full_state_shared_wide")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    mode = ["--smoke", "--hardware", "cpu", "--no-upload-artifacts"] if args.smoke else [
        "--upload-artifacts",
    ]
    for leaf in LEAVES:
        print(f"PPO ladder: seed={args.seed} leaf={leaf}", flush=True)
        subprocess.run([
            "rl-harness", f"experiments.{STUDY}.{leaf}.experiment", "--seed", str(args.seed),
            *mode,
        ], check=True)


if __name__ == "__main__":
    main()
