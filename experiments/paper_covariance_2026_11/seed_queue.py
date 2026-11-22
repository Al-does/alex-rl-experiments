"""Run the covariance treatments sequentially on one GPU for one training seed."""

import argparse
import os
import subprocess


CONDITIONS = (
    "trivial_quotient", "reward_state_quotient", "full_state_quotient",
    "joint_reward", "marginal_reward", "rocksample",
)
LEAVES = tuple(f"{condition}_cov{weight}" for condition in CONDITIONS for weight in ("0p01", "0p1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--only", choices=LEAVES, nargs="+",
                        help="Run only these leaves when retrying part of a queue.")
    args = parser.parse_args()
    leaves = [leaf for leaf in LEAVES if args.only is None or leaf in args.only]
    env = {**os.environ, "JAX_PLATFORMS": "cuda",
           "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
    for index, leaf in enumerate(leaves):
        module = f"experiments.paper_covariance_2026_11.{leaf}.experiment"
        print(f"[seed {args.seed}] {index + 1}/{len(leaves)} starting {leaf}", flush=True)
        subprocess.run(["rl-harness", module, "--seed", str(args.seed), "--upload-artifacts"],
                       env=env, check=True)
        print(f"[seed {args.seed}] {index + 1}/{len(leaves)} completed {leaf}", flush=True)


if __name__ == "__main__":
    main()
