#!/usr/bin/env bash
# Sequential multi-seed runner for one token-guess cycle-2 arm on a vast box.
# Truncates at ~0.66M steps (third checkpoint) via MESS3_TG_C2_MAX_ENV_STEPS.
#
# Usage:
#   bash experiments/mess3_token_guess_cycle_2/run_arm.sh decoupled_kelly 42 43 ... 56
set -euo pipefail

CONDITION="${1:?condition required}"
shift
if [ "$#" -lt 1 ]; then
  echo "usage: $0 <condition> <seed> [seed ...]" >&2
  exit 2
fi

STUDY="mess3_token_guess_cycle_2"
export MESS3_TG_C2_MAX_ENV_STEPS="${MESS3_TG_C2_MAX_ENV_STEPS:-700000}"

push_one() {
  local run_name="$1"
  python - "$run_name" <<'PY'
import os
import sys
from pathlib import Path

from devops.vast.self_destruct import push_results

run_name = sys.argv[1]
instance = None
path = Path("/root/vast_instance_id")
if path.is_file():
    instance = path.read_text().strip() or None
ok = push_results(
    branch=os.environ.get("VAST_RESULTS_BRANCH", "results"),
    run_name=run_name,
    instance_id=instance,
)
raise SystemExit(0 if ok else 2)
PY
}

for seed in "$@"; do
  run_id="${STUDY}-${CONDITION}-seed${seed}-0p66m"
  echo "[arm] start condition=${CONDITION} seed=${seed} run_id=${run_id} max_env_steps=${MESS3_TG_C2_MAX_ENV_STEPS}"
  rl-harness "experiments.${STUDY}.${CONDITION}.experiment" \
    --seed "${seed}" \
    --hardware cuda4090 \
    --upload-artifacts \
    --run-id "${run_id}"
  echo "[arm] pushing compact results for ${run_id}"
  push_one "${run_id}"
  echo "[arm] done seed=${seed}"
done

echo "[arm] queue complete condition=${CONDITION} seeds=$*"
