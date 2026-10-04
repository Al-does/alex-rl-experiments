"""Probe the rl_b10_continue checkpoints and append them to probe_curves.json."""
import json
import re
from pathlib import Path

from experiments.pusher_b.probe_analysis.probe_curves import (CKPTS, EPISODE_LENGTH, N_FIT, N_TEST, ROOT, SEEDS, load_module_only,
                          r2_pooled, rl_forward, sample, targets)

LEAF = "rl_b10_continue"
PRESET = "b10"
PRIOR_STEPS = 15_134_336


def checkpoints():
    base = CKPTS / LEAF
    items = [(base / "initial_checkpoint", PRIOR_STEPS, "restore (iter 56)")]
    for d in sorted((base / "interval_checkpoints").iterdir()):
        m = re.match(r"iteration_(\d+)_steps_(\d+)", d.name)
        if m and d.is_dir():
            items.append((d, int(m.group(2)), f"iter {int(m.group(1))}"))
    final = next((base / "tune").glob("*/checkpoint_000000"))
    curves = [json.loads(l) for l in open(next(
        (ROOT.parent / LEAF / "results")
        .glob("*/training_curves.jsonl")))]
    if int(curves[-1]["steps"]) != items[-1][1]:
        items.append((final, int(curves[-1]["steps"]), f"iter {curves[-1]['iteration']} (final)"))
    return items


def main():
    out_path = ROOT / "results" / "probe_curves.json"
    results = json.loads(out_path.read_text())
    fit_t = sample(PRESET, N_FIT, SEEDS["fit"])
    test_t = sample(PRESET, N_TEST, SEEDS["test"])
    fb, fn = targets(fit_t, PRESET)
    tb, tn = targets(test_t, PRESET)
    out = []
    for path, env_steps, label in checkpoints():
        module = load_module_only(path).eval()
        a_fit, _ = rl_forward(module, fit_t)
        a_test, pred = rl_forward(module, test_t)
        r2 = r2_pooled(a_fit, fb, a_test, tb)
        acc = float((pred == tn).mean())
        out.append({"label": label, "env_steps": env_steps,
                    "sequences_seen": env_steps / EPISODE_LENGTH, "r2": r2, "accuracy": acc})
        print(LEAF, label, env_steps, f"R2={r2:.6f} 1-R2={1-r2:.3e} acc={acc:.4f}", flush=True)
        results["runs"][LEAF] = out
        out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
