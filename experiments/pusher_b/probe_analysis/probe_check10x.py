"""Refit the probe on 10x fit data for two random continuation checkpoints; compare to 2048-fit values."""
import json
import random
import sys
import time

from experiments.pusher_b.probe_analysis.probe_curves import N_FIT, N_TEST, ROOT, SEEDS, load_module_only, r2_pooled, rl_forward, sample, targets
from experiments.pusher_b.probe_analysis.probe_continue import LEAF, PRESET, checkpoints

N_BIG = 10 * N_FIT
BIG_SEED = 7_010

def main():
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
    cks = checkpoints()
    picks = sorted(rng.sample(range(len(cks)), 2))
    print("picked", [cks[i][2] for i in picks], flush=True)
    old = {p["label"]: p for p in json.loads((ROOT / "results" / "probe_curves.json").read_text())["runs"][LEAF]}
    t0 = time.time()
    fit_small = sample(PRESET, N_FIT, SEEDS["fit"])
    fit_big = sample(PRESET, N_BIG, BIG_SEED)
    test_t = sample(PRESET, N_TEST, SEEDS["test"])
    fsb, _ = targets(fit_small, PRESET)
    fbb, _ = targets(fit_big, PRESET)
    tb, _ = targets(test_t, PRESET)
    print(f"data generated in {time.time()-t0:.1f}s", flush=True)
    for i in picks:
        path, steps, label = cks[i]
        module = load_module_only(path).eval()
        t0 = time.time()
        a_small, _ = rl_forward(module, fit_small)
        a_big, _ = rl_forward(module, fit_big)
        a_test, _ = rl_forward(module, test_t)
        r2_small = r2_pooled(a_small, fsb, a_test, tb)
        r2_big = r2_pooled(a_big, fbb, a_test, tb)
        print(f"{label} steps={steps}  fit2048: 1-R2={1-r2_small:.4e} (stored {1-old[label]['r2']:.4e})"
              f"  fit20480: 1-R2={1-r2_big:.4e}  rel diff={(r2_big-r2_small)/(1-r2_small):+.3%}"
              f"  [{time.time()-t0:.1f}s]", flush=True)

if __name__ == "__main__":
    main()
