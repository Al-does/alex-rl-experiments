#!/usr/bin/env bash
# Clone/link the shared library beside this repo and sync the editable env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
LIB_LINK="$PARENT/rl-harness"
LIB_SPACED="$PARENT/RL Harness"
LIB_CLONE="$PARENT/rl-harness-src"
LIBRARY_URL="${RL_HARNESS_URL:-https://github.com/Al-does/RL-Harness.git}"

if [ -d "$LIB_LINK/.git" ] || [ -L "$LIB_LINK" ]; then
  echo "Using existing library at $LIB_LINK"
elif [ -d "$LIB_SPACED/.git" ]; then
  echo "Linking $LIB_LINK -> RL Harness"
  ln -s "RL Harness" "$LIB_LINK"
elif [ -d "$LIB_CLONE/.git" ]; then
  echo "Linking $LIB_LINK -> rl-harness-src"
  ln -s "rl-harness-src" "$LIB_LINK"
else
  echo "Cloning $LIBRARY_URL -> $LIB_CLONE"
  git clone "$LIBRARY_URL" "$LIB_CLONE"
  ln -sfn "rl-harness-src" "$LIB_LINK"
fi

cd "$ROOT"
uv sync --group dev
echo
echo "Ready. Example:"
echo "  uv run rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --smoke"
echo
echo "Library edits: work in $LIB_LINK (branch + PR there)."
echo "Experiment edits: commit in this repo."
