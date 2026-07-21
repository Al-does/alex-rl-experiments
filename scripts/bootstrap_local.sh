#!/usr/bin/env bash
# Clone/link the shared library beside this repo and sync the editable env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
LIB_LINK="$PARENT/rl-harness"
LIB_SPACED="$PARENT/RL Harness"
LIB_CLONE="$PARENT/rl-harness-src"
LIBRARY_URL="${RL_HARNESS_URL:-https://github.com/Al-does/RL-Harness.git}"

# Vast provisioning needs a local OpenSSH client + keypair. Cursor Cloud images
# may lag behind Dockerfile rebuilds, so repair Linux installs at bootstrap time.
# Keys are generated per machine/session (not baked into the image) so concurrent
# agents do not share a private key.
ensure_local_ssh() {
  if [ "$(uname -s)" = "Linux" ] && ! command -v ssh >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing openssh-client (required for vast.ai provisioning)..."
      apt-get update
      apt-get install -y --no-install-recommends openssh-client
    else
      echo "WARNING: local ssh client missing; vast.ai provisioning will fail." >&2
    fi
  fi

  if ! command -v ssh-keygen >/dev/null 2>&1; then
    return 0
  fi
  mkdir -p "$HOME/.ssh"
  chmod 700 "$HOME/.ssh" 2>/dev/null || true
  if [ ! -f "$HOME/.ssh/id_rsa" ] || [ ! -f "$HOME/.ssh/id_rsa.pub" ]; then
    echo "Generating ~/.ssh/id_rsa for vast.ai SSH registration..."
    ssh-keygen -t rsa -b 4096 -N "" -f "$HOME/.ssh/id_rsa" -q
  fi
}

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

ensure_local_ssh

cd "$ROOT"
uv sync --group dev
echo
echo "Ready. Example:"
echo "  uv run rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --smoke"
echo
echo "Library edits: work in $LIB_LINK (branch + PR there)."
echo "Experiment edits: commit in this repo."
echo
echo "New colleagues should fork rl-experiments (entry point), not this repo:"
echo "  https://github.com/Al-does/rl-experiments"
