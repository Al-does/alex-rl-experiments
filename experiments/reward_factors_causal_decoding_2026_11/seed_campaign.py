"""One CPU-rich box per agent seed; smoke the same command before full budgets."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def prepare_archives():
    repo = Path(__file__).resolve().parents[2]
    shallow = subprocess.check_output(["git", "rev-parse", "--is-shallow-repository"], cwd=repo, text=True).strip()
    if shallow == "true":
        subprocess.run(["git", "fetch", "--unshallow", "--filter=blob:none", "origin"], cwd=repo, check=True)
    for revision in ("45517b13a67d746048a002f9db52ea55a654b219", "3504bfb72455a411b725ef368a18ff892760555f"):
        exists = subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=repo,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if exists.returncode:
            subprocess.run(["git", "fetch", "--filter=blob:none", "origin", revision], cwd=repo, check=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-cpus", type=int, default=32)
    args = parser.parse_args()
    prepare_archives()
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
