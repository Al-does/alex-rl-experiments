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

# Push compact experiments/ results onto origin/results without replaying the
# feature-branch commit history (which conflicts with the orphaned results tip).
# Always restore the training code commit afterward so later seeds still see
# MESS3_TG_C2_MAX_ENV_STEPS / early-stop shared.py.
push_one() {
  local run_name="$1"
  local train_ref
  train_ref="$(git -C "${VAST_EXPERIMENT_DIR:-.}" rev-parse HEAD)"
  python - "$run_name" "$train_ref" <<'PY'
import os
import subprocess
import sys
import time
from pathlib import Path

run_name = sys.argv[1]
train_ref = sys.argv[2]
repo = Path(os.environ.get("VAST_EXPERIMENT_DIR", Path.cwd()))
branch = os.environ.get("VAST_RESULTS_BRANCH", "results")


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(repo),
        capture_output=True,
        text=True,
    )


def log(msg: str) -> None:
    print(f"[push_one] {msg}", flush=True)


def restore_training_ref() -> None:
    """Return the worktree to the campaign code after a results checkout."""
    restored = run(["git", "checkout", "--detach", train_ref])
    if restored.returncode != 0:
        log(
            f"WARNING: could not restore training ref {train_ref}: "
            f"{restored.stderr.strip()}"
        )
    else:
        log(f"restored training ref {train_ref[:12]}")


# Clear any failed rebase from a previous push attempt.
run(["git", "rebase", "--abort"])
run(["git", "merge", "--abort"])

# Stage only compact experiment outputs (artifacts/ remains gitignored).
add = run(["git", "add", "-A", "--", "experiments/"])
if add.returncode != 0:
    log(f"git add failed: {add.stderr.strip()}")
    restore_training_ref()
    raise SystemExit(2)

staged = run(["git", "diff", "--cached", "--name-only"])
names = [line for line in staged.stdout.splitlines() if line.strip()]
if not names:
    log("no new compact experiment results to push")
    restore_training_ref()
    raise SystemExit(0)

# Snapshot staged blob contents, then rebuild a commit on top of origin/results.
snapshots: dict[str, bytes] = {}
for name in names:
    show = subprocess.run(
        ["git", "show", f":{name}"],
        cwd=str(repo),
        capture_output=True,
    )
    if show.returncode != 0:
        log(f"could not read staged {name}: {show.stderr.decode()}")
        restore_training_ref()
        raise SystemExit(2)
    snapshots[name] = show.stdout

# Drop the index so we can move HEAD onto the results tip cleanly.
run(["git", "reset", "--hard", "HEAD"])

fetched = run(["git", "fetch", "--depth", "1", "origin", branch])
if fetched.returncode != 0:
    log(f"fetch {branch} failed: {fetched.stderr.strip()}")
    restore_training_ref()
    raise SystemExit(2)

checkout = run(["git", "checkout", "-B", branch, "FETCH_HEAD"])
if checkout.returncode != 0:
    log(f"checkout {branch} failed: {checkout.stderr.strip()}")
    restore_training_ref()
    raise SystemExit(2)

for name, data in snapshots.items():
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

add2 = run(["git", "add", "-A", "--", "experiments/"])
if add2.returncode != 0:
    log(f"re-add failed: {add2.stderr.strip()}")
    restore_training_ref()
    raise SystemExit(2)

instance = None
id_path = Path("/root/vast_instance_id")
if id_path.is_file():
    instance = id_path.read_text().strip() or None
label = (
    f"results: {run_name} (vast {instance})"
    if instance
    else f"results: {run_name}"
)
commit = run(["git", "commit", "-m", label])
if commit.returncode != 0:
    # Nothing new relative to results tip.
    if "nothing to commit" in (commit.stdout + commit.stderr).lower():
        log("results tip already has these files")
        restore_training_ref()
        raise SystemExit(0)
    log(f"commit failed: {(commit.stderr or commit.stdout).strip()}")
    restore_training_ref()
    raise SystemExit(2)

delay = 1.0
pushed_ok = False
for attempt in range(1, 7):
    pushed = run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"])
    if pushed.returncode == 0:
        log(f"pushed {run_name} to {branch}")
        pushed_ok = True
        break
    log(f"push rejected (attempt {attempt}/6): {pushed.stderr.strip()}")
    # Concurrent boxes: refetch and retry with our tree on the new tip.
    fetched = run(["git", "fetch", "--depth", "1", "origin", branch])
    if fetched.returncode == 0:
        run(["git", "reset", "--soft", "FETCH_HEAD"])
        for name, data in snapshots.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        run(["git", "add", "-A", "--", "experiments/"])
        run(["git", "commit", "-m", label])
    time.sleep(delay)
    delay = min(delay * 2, 30.0)

restore_training_ref()
if not pushed_ok:
    log(f"push failed after retries for {run_name}")
    raise SystemExit(2)
raise SystemExit(0)
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
