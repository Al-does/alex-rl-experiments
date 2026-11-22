"""Approved remaining agent seeds; unchanged seed-42 scientific budgets."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.reward_factors_causal_decoding_2026_11.seed_campaign import prepare_archives


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", type=int, choices=(43, 44), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cpus", type=int, default=32)
    args = parser.parse_args()
    if min(args.workers, args.max_cpus) < 1:
        parser.error("workers and max-cpus must be positive")
    prepare_archives()
    os.environ.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    command = [sys.executable, "-m", "experiments.reward_factors_block2_steering_2026_11.run",
               "--seed", str(args.seed), "--site", "1", "--workers", str(args.workers),
               "--max-cpus", str(args.max_cpus)]
    subprocess.run([*command, "--test", "moving", "--method", "eq4", "context_exact_full", "action", "null", "--smoke"], check=True)
    print(f"SMOKE PASSED seed={args.seed}; starting full 16/32 episode budgets", flush=True)
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-m", "experiments.reward_factors_block2_steering_2026_11.encoder_run",
                    "--seed", str(args.seed), "--site", "1", "--workers", str(args.workers)], check=True)
    subprocess.run([*command, "--site", "3", "--method", "eq4", "eq4_joint", "random", "action", "das", "map"], check=True)
    from . import report
    result = report.aggregate(report.REPO, seeds=(args.seed,))
    destination = Path(__file__).parent / "results" / f"campaign_validation_seed{args.seed}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"CAMPAIGN COMPLETE seed={args.seed}; validated {len(result['files'])} compact inputs", flush=True)


if __name__ == "__main__":
    main()
