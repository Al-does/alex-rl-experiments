"""Publish compact remote results without replaying feature-branch history."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time


def _run(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )


def publish_results(run_name: str) -> bool:
    """Snapshot experiment outputs onto the results branch, then restore them."""

    repo = Path(os.environ.get("VAST_EXPERIMENT_DIR", Path.cwd()))
    branch = os.environ.get("VAST_RESULTS_BRANCH", "results")
    train_ref_result = _run(repo, ["git", "rev-parse", "HEAD"])
    if train_ref_result.returncode:
        print(
            f"[publish] could not resolve training ref: "
            f"{train_ref_result.stderr.strip()}",
            flush=True,
        )
        return False
    train_ref = train_ref_result.stdout.strip()
    snapshots: dict[str, bytes] = {}

    def restore() -> None:
        restored = _run(repo, ["git", "checkout", "--detach", train_ref])
        if restored.returncode:
            print(
                f"[publish] WARNING: could not restore {train_ref[:12]}: "
                f"{restored.stderr.strip()}",
                flush=True,
            )
            return
        for name, data in snapshots.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f"[publish] restored training ref {train_ref[:12]}", flush=True)

    _run(repo, ["git", "rebase", "--abort"])
    _run(repo, ["git", "merge", "--abort"])
    added = _run(repo, ["git", "add", "-A", "--", "experiments/"])
    if added.returncode:
        print(f"[publish] git add failed: {added.stderr.strip()}", flush=True)
        return False
    staged = _run(repo, ["git", "diff", "--cached", "--name-only"])
    names = [name for name in staged.stdout.splitlines() if name.strip()]
    if not names:
        print("[publish] no compact results to publish", flush=True)
        return True
    for name in names:
        shown = subprocess.run(
            ["git", "show", f":{name}"],
            cwd=str(repo),
            capture_output=True,
            check=False,
        )
        if shown.returncode:
            print(f"[publish] could not snapshot {name}", flush=True)
            restore()
            return False
        snapshots[name] = shown.stdout

    reset = _run(repo, ["git", "reset", "--hard", "HEAD"])
    if reset.returncode:
        print(f"[publish] reset failed: {reset.stderr.strip()}", flush=True)
        restore()
        return False
    fetched = _run(repo, ["git", "fetch", "--depth", "1", "origin", branch])
    if fetched.returncode:
        print(f"[publish] fetch failed: {fetched.stderr.strip()}", flush=True)
        restore()
        return False
    checkout = _run(repo, ["git", "checkout", "-B", branch, "FETCH_HEAD"])
    if checkout.returncode:
        print(f"[publish] checkout failed: {checkout.stderr.strip()}", flush=True)
        restore()
        return False

    for name, data in snapshots.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if _run(repo, ["git", "add", "-A", "--", "experiments/"]).returncode:
        restore()
        return False
    instance_path = Path("/root/vast_instance_id")
    instance = (
        instance_path.read_text().strip()
        if instance_path.is_file()
        else os.environ.get("VAST_INSTANCE_ID")
    )
    label = (
        f"results: {run_name} (vast {instance})"
        if instance
        else f"results: {run_name}"
    )
    committed = _run(repo, ["git", "commit", "-m", label])
    if committed.returncode:
        combined = (committed.stdout + committed.stderr).lower()
        if "nothing to commit" not in combined:
            print(f"[publish] commit failed: {committed.stderr.strip()}", flush=True)
            restore()
            return False

    delay = 1.0
    pushed_ok = False
    for attempt in range(1, 7):
        pushed = _run(repo, ["git", "push", "origin", f"HEAD:refs/heads/{branch}"])
        if pushed.returncode == 0:
            pushed_ok = True
            print(f"[publish] pushed {run_name} to {branch}", flush=True)
            break
        print(
            f"[publish] push rejected ({attempt}/6): {pushed.stderr.strip()}",
            flush=True,
        )
        fetched = _run(repo, ["git", "fetch", "--depth", "1", "origin", branch])
        if fetched.returncode == 0:
            _run(repo, ["git", "reset", "--soft", "FETCH_HEAD"])
            for name, data in snapshots.items():
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            _run(repo, ["git", "add", "-A", "--", "experiments/"])
            _run(repo, ["git", "commit", "-m", label])
        time.sleep(delay)
        delay = min(2.0 * delay, 30.0)
    restore()
    return pushed_ok
