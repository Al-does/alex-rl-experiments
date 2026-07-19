#!/usr/bin/env bash
# Run vast provisioning with local B2 secrets loaded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIBRARY="${RL_HARNESS_ROOT:-$ROOT/../rl-harness}"
if [ ! -d "$LIBRARY/devops/vast" ] && [ -d "$ROOT/../RL Harness/devops/vast" ]; then
  LIBRARY="$ROOT/../RL Harness"
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/load_env.sh"
cd "$LIBRARY"
exec uv run --group devops python -m devops.vast.provision "$@"
