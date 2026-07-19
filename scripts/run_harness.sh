#!/usr/bin/env bash
# Run rl-harness with local B2 secrets loaded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/load_env.sh"
cd "$ROOT"
exec uv run rl-harness "$@"
