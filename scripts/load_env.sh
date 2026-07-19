#!/usr/bin/env bash
# Source local secrets for rl-harness and vast provisioning.
set -a
if [ -f "$HOME/.rl_harness_b2_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.rl_harness_b2_env"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$ROOT/.env.local" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
fi
set +a
