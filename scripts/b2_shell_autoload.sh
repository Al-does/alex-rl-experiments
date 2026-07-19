#!/usr/bin/env bash
# Install or remove rl-harness B2 auto-load in ~/.zshrc.
set -euo pipefail

MARKER_START="# >>> rl-harness b2 autoload >>>"
MARKER_END="# <<< rl-harness b2 autoload <<<"
BLOCK=$(
  cat <<'EOF'
# >>> rl-harness b2 autoload >>>
[ -f "$HOME/.rl_harness_b2_env" ] && set -a && source "$HOME/.rl_harness_b2_env" && set +a
# <<< rl-harness b2 autoload <<<
EOF
)
ZSHRC="${ZSHRC:-$HOME/.zshrc}"

usage() {
  cat <<EOF
Usage: $(basename "$0") {install|remove|status}

  install  Add B2 auto-load to \$ZSHRC (default: \$HOME/.zshrc)
  remove   Remove the managed block from \$ZSHRC
  status   Report whether the managed block is present
EOF
}

has_block() {
  [ -f "$ZSHRC" ] && grep -Fq "$MARKER_START" "$ZSHRC"
}

install_block() {
  if has_block; then
    echo "B2 auto-load already present in $ZSHRC"
    return 0
  fi
  if [ ! -f "$ZSHRC" ]; then
    printf '%s\n' "$BLOCK" > "$ZSHRC"
  else
    printf '\n%s\n' "$BLOCK" >> "$ZSHRC"
  fi
  echo "Added B2 auto-load to $ZSHRC"
  echo "Open a new terminal, or run: source \"$ZSHRC\""
}

remove_block() {
  if ! has_block; then
    echo "No B2 auto-load block found in $ZSHRC"
    return 0
  fi
  python3 - <<'PY' "$ZSHRC" "$MARKER_START" "$MARKER_END"
from pathlib import Path
import sys

path = Path(sys.argv[1])
start = sys.argv[2]
end = sys.argv[3]
lines = path.read_text().splitlines(keepends=True)
out = []
skip = False
for line in lines:
    if line.rstrip("\n") == start:
        skip = True
        continue
    if skip:
        if line.rstrip("\n") == end:
            skip = False
        continue
    out.append(line)
while out and out[-1].strip() == "":
    out.pop()
if out and not out[-1].endswith("\n"):
    out[-1] += "\n"
path.write_text("".join(out))
PY
  echo "Removed B2 auto-load from $ZSHRC"
}

status_block() {
  if has_block; then
    echo "installed ($ZSHRC)"
  else
    echo "not installed ($ZSHRC)"
  fi
}

case "${1:-}" in
  install) install_block ;;
  remove) remove_block ;;
  status) status_block ;;
  *) usage; exit 1 ;;
esac
