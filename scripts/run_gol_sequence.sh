#!/usr/bin/env bash
set -euo pipefail

# The cuda4090 auto profile over-subscribes env runners for this container's
# effective memory budget, so pin the harness to half the CPUs.  This keeps
# num_env_runners below the cgroup memory pressure threshold without changing
# the experiment's model or batch sizes.
CPUS=0-7

# Disable Ray's worker OOM killer; the container has a hard memory ceiling and
# the reduced runner count should keep usage below it.
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=1.0

cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

for v in variant_2 variant_3 variant_2_quarter variant_3_quarter; do
  echo "=== $v ==="
  taskset -c "$CPUS" rl-harness \
    "experiments.gol_reward_state_action_symmetry_cycle_1.${v}.experiment" \
    --seed 42 \
    --upload-artifacts
done
