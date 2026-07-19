#!/usr/bin/env bash
# Write ~/.rl_harness_b2_env for local runs and vast provisioning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${RL_HARNESS_B2_ENV_FILE:-$HOME/.rl_harness_b2_env}"
EXAMPLE="$ROOT/.env.local.example"
AUTOLOAD_SCRIPT="$ROOT/scripts/b2_shell_autoload.sh"
SHELL_AUTOLOAD=1
SECRETS_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --no-shell-autoload   Do not add B2 auto-load to ~/.zshrc
  --shell-autoload-only Install ~/.zshrc auto-load only (skip secret prompts)
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-shell-autoload)
      SHELL_AUTOLOAD=0
      shift
      ;;
    --shell-autoload-only)
      SECRETS_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$SECRETS_ONLY" -eq 0 ]]; then
  echo "Backblaze B2 setup for rl-harness artifact upload"
  echo "Secrets will be written to: $TARGET"
  echo

  read -r -p "B2 bucket name: " B2_BUCKET
  read -r -p "B2 S3 endpoint [s3.us-west-004.backblazeb2.com]: " B2_ENDPOINT
  B2_ENDPOINT="${B2_ENDPOINT:-s3.us-west-004.backblazeb2.com}"
  if [[ "$B2_ENDPOINT" != http://* && "$B2_ENDPOINT" != https://* ]]; then
    B2_ENDPOINT="https://${B2_ENDPOINT}"
  fi
  read -r -p "B2 application key ID: " B2_APPLICATION_KEY_ID
  read -r -s -p "B2 application key (hidden): " B2_APPLICATION_KEY
  echo
  read -r -p "Optional B2 prefix inside bucket [alex]: " B2_PREFIX
  B2_PREFIX="${B2_PREFIX:-alex}"

  umask 077
  cat > "$TARGET" <<EOF
# rl-harness Backblaze B2 credentials (chmod 600)
export B2_BUCKET="${B2_BUCKET}"
export B2_ENDPOINT="${B2_ENDPOINT}"
export B2_APPLICATION_KEY_ID="${B2_APPLICATION_KEY_ID}"
export B2_APPLICATION_KEY="${B2_APPLICATION_KEY}"
export B2_PREFIX="${B2_PREFIX}"
EOF
  chmod 600 "$TARGET"

  echo
  echo "Wrote $TARGET"
fi

if [[ "$SHELL_AUTOLOAD" -eq 1 ]]; then
  if [[ "$SECRETS_ONLY" -eq 0 ]]; then
    read -r -p "Add B2 auto-load to ~/.zshrc for every new terminal? [Y/n] " reply
    reply="${reply:-Y}"
    case "$reply" in
      [Yy]|[Yy][Ee][Ss]|"") ;;
      *)
        SHELL_AUTOLOAD=0
        echo "Skipped ~/.zshrc auto-load."
        ;;
    esac
  fi
  if [[ "$SHELL_AUTOLOAD" -eq 1 ]]; then
    bash "$AUTOLOAD_SCRIPT" install
  fi
fi

if [[ "$SECRETS_ONLY" -eq 0 ]]; then
  echo
  echo "Local runs:"
  echo "  source scripts/load_env.sh"
  echo "  uv run rl-harness experiments....experiment --smoke"
  echo
  echo "Or use the wrapper:"
  echo "  ./scripts/run_harness.sh experiments....experiment --smoke"
  echo
  echo "Vast runs from this Mac automatically forward these vars to rented boxes"
  echo "when you run devops.vast.provision (no extra wiring needed)."
  echo
  echo "To remove ~/.zshrc auto-load later:"
  echo "  ./scripts/b2_shell_autoload.sh remove"
  echo
  echo "Template for reference: $EXAMPLE"
fi
