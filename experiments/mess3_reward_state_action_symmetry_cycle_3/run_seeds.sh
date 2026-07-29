#!/usr/bin/env bash
# Run five seeds for one cycle-3 arm. Intended for Vast --run.
# Usage: bash run_seeds.sh <module> <run_name_prefix>
set -euo pipefail
MODULE="${1:?module required, e.g. experiments....variant_2_decoupled_kelly.experiment}"
PREFIX="${2:?run-name prefix required}"
for s in 42 43 44 45 46; do
  echo "===== starting seed ${s} ====="
  rl-harness "${MODULE}" --seed "${s}" --upload-artifacts --run-id "${PREFIX}-seed${s}"
  echo "===== finished seed ${s} ====="
done
