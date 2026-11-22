"""Seed-42-only pilot; never launches the unapproved three-seed campaign."""

import argparse
import os
import subprocess
import sys

from experiments.reward_factors_causal_decoding_2026_11.seed_campaign import prepare_archives


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-cpus", type=int, default=32)
    args = parser.parse_args()
    if args.workers < 1 or args.max_cpus < 1:
        parser.error("workers and max-cpus must be positive")
    prepare_archives()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      XLA_PYTHON_CLIENT_PREALLOCATE="false")
    command = [sys.executable, "-m", "experiments.reward_factors_block2_steering_2026_11.run", "--seed", "42",
               "--site", "1", "--workers", str(args.workers), "--max-cpus", str(args.max_cpus)]
    subprocess.run([*command, "--test", "moving", "--method", "eq4", "context_exact_full", "action", "null", "--smoke"], check=True)
    print("REMOTE SMOKE PASSED; starting seed-42 B2 pilot, 16/32 episode budgets", flush=True)
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-m", "experiments.reward_factors_block2_steering_2026_11.encoder_run", "--seed", "42",
                    "--site", "1", "--workers", str(args.workers)], check=True)
    subprocess.run([*command, "--site", "3", "--method", "eq4", "eq4_joint", "random", "action", "das", "map"], check=True)
    subprocess.run([sys.executable, "-m", "experiments.reward_factors_block2_steering_2026_11.report"], check=True)
    print("SEED-42 PILOT COMPLETE; STOPPING BEFORE THREE-SEED CAMPAIGN", flush=True)


if __name__ == "__main__":
    main()
