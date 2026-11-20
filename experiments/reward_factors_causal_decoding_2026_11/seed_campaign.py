"""One CPU-rich box per agent seed; smoke the same command before full budgets."""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-cpus", type=int, default=32)
    args = parser.parse_args()
    os.environ.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-m", "experiments.reward_factors_causal_decoding_2026_11.run",
               "--seed", str(args.seed), "--method", "all", "--campaign",
               "--workers", str(args.workers), "--max-cpus", str(args.max_cpus)]
    subprocess.run([*command, "--smoke"], check=True)
    print(f"SMOKE PASSED seed={args.seed}; starting 16-selection/32-confirmation campaign", flush=True)
    subprocess.run(command, check=True)
    print(f"CAMPAIGN COMPLETE seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
