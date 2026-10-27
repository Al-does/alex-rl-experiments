"""Probe the rl_b10_aux_ce checkpoints (in-progress run; re-run as more land) and write probe_aux_ce.json.

Run: uv run python -m experiments.pusher_b.probe_analysis.probe_aux_ce
Checkpoints are synced from B2 into artifacts/ckpts/rl_b10_aux_ce/ preserving relative paths.
"""
import json
import re
import traceback

import experiments.pusher_b.learning  # noqa: F401  (registers ActorCriticWithNextTokenAux for restore)
from experiments.pusher_b.probe_analysis.probe_curves import (CKPTS, EPISODE_LENGTH, N_FIT, N_TEST, ROOT, SEEDS, load_module_only,
                          r2_pooled, rl_forward, sample, targets)

LEAF = "rl_b10_aux_ce"
PRESET = "b10"


def checkpoints():
    base = CKPTS / LEAF
    items = []
    if (base / "initial_checkpoint").is_dir():
        items.append((base / "initial_checkpoint", 0, "init"))
    for sub in ("log_spaced_checkpoints", "interval_checkpoints"):
        if not (base / sub).is_dir():
            continue
        for d in sorted((base / sub).iterdir()):
            m = re.match(r"iteration_(\d+)_steps_(\d+)", d.name)
            if m and d.is_dir():
                items.append((d, int(m.group(2)), f"iter {int(m.group(1))}"))
    final = next((base / "tune").glob("*/checkpoint_*"), None)
    if final is not None:
        run_id = json.loads((base / "compact-results" / "run_manifest.json").read_text())["run_id"] \
            if (base / "compact-results" / "run_manifest.json").is_file() else None
        curve_files = sorted((ROOT.parent / LEAF / "results").glob(f"{run_id or '*'}/training_curves.jsonl"))
        if curve_files:
            last = json.loads(curve_files[-1].read_text().splitlines()[-1])
            items.append((final, int(last["steps"]), f"iter {last['iteration']} (final)"))
    seen = set()
    out = []
    for path, steps, label in sorted(items, key=lambda t: t[1]):
        if steps not in seen:
            seen.add(steps)
            out.append((path, steps, label))
    return out


def main():
    out_path = ROOT / "results" / "probe_aux_ce.json"
    fit_t = sample(PRESET, N_FIT, SEEDS["fit"])
    test_t = sample(PRESET, N_TEST, SEEDS["test"])
    fb, fn = targets(fit_t, PRESET)
    tb, tn = targets(test_t, PRESET)
    out, skipped = [], []
    for path, env_steps, label in checkpoints():
        try:
            module = load_module_only(path).eval()
            a_fit, _ = rl_forward(module, fit_t)
            a_test, pred = rl_forward(module, test_t)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            skipped.append({"label": label, "env_steps": env_steps, "path": str(path.relative_to(CKPTS)), "error": repr(exc)})
            continue
        r2_fit = r2_pooled(a_fit, fb, a_fit, fb)
        r2 = r2_pooled(a_fit, fb, a_test, tb)
        acc = float((pred == tn).mean())
        out.append({"label": label, "env_steps": env_steps, "sequences_seen": env_steps / EPISODE_LENGTH,
                    "r2_fit": r2_fit, "r2": r2, "accuracy": acc})
        print(LEAF, label, env_steps, f"R2fit={r2_fit:.6f} R2={r2:.6f} 1-R2={1-r2:.3e} acc={acc:.4f}", flush=True)
        out_path.write_text(json.dumps({"runs": {LEAF: out}, "skipped": skipped, "settings": {
            "n_fit": N_FIT, "n_test": N_TEST, "positions": EPISODE_LENGTH, "seeds": SEEDS, "preset": PRESET,
            "probe": "affine least squares, residual stream after last block (pre-final-norm) -> 3-state Bayesian belief; pooled R^2 (r2_fit on fit set, r2 on held-out)",
            "accuracy": "greedy argmax next-token prediction on held-out sequences",
        }}, indent=2))


if __name__ == "__main__":
    main()
