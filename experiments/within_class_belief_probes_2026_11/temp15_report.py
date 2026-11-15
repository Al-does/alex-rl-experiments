"""Record completed PR202 protocol checks, including hard-stop failures."""

import hashlib
import json
import subprocess

import jax
import numpy as np

from .temp15 import ROOT, RAW, gate_passes, rows


def main():
    path = ROOT / "results/ladder_temp15_gate.json"
    report = json.loads(path.read_text())
    report["provenance"] = {
        "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_commit": subprocess.check_output(["git", "-C", str(ROOT.parents[2]/"rl-harness"), "rev-parse", "HEAD"], text=True).strip(),
        "jax": jax.__version__, "numpy": np.__version__, "devices": str(jax.devices()),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"temp15.py", ROOT/"temp15_report.py", ROOT/"metrics.py")}}
    report["bootstrap_protocol"] = "2000 held-fixed-probe whole-test-episode resamples (84 episodes, last truncated), recompute target means; SE conditional on seed42 and fixed fit data. Threshold=max(0.01,5*SE) applies to both archived R2 and global MSE ratio."
    lines = ["## PR202: completed 24 recorded protocol checks", "",
             "Supersedes the earlier hard-stop rules: checks are recorded for every init/final arm; hard-stop threshold is max(0.01, 5 × CPU episode-bootstrap SE). No seed uncertainty is estimated: all observations condition on training seed 42 and fixed fit probes. Cross-device numerical execution is the motivation, not a verified attribution to TF32.", "",
             "| Condition / architecture | Stage | Archived R² | CPU R² | Delta | Bootstrap SE | Hard-stop limit | Status |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    flags = []
    failures = []
    names = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}
    for check in report["checks"]:
        arm = check["arm"]
        variant, arch = int(arm.split("_")[1]), arm.split("_", 2)[2]
        stage = "Init" if check["step"] == 0 else "Final"
        counts = []
        for split, n in (("fit", 60000), ("test", 80000)):
            with np.load(RAW/arm/f"{check['step']}_{split}.npz") as saved:
                actions = rows({"action": saved["action"]}, "action", n)
                counts.append(np.bincount(actions, minlength=3).tolist())
        check["cpu_fit_counts"], check["cpu_test_counts"] = counts
        check["one_class_fit_and_test"] = int(np.count_nonzero(np.sum(counts, axis=0))) == 1
        check["within_hard_stop_limit"] = gate_passes(check)
        if not check["within_hard_stop_limit"]:
            failures.append((arm, stage))
        if check["one_class_fit_and_test"]:
            action = ("Noop", "Tilt X", "Tilt Y")[int(np.argmax(counts[1]))]
            flags.append(f"{names[variant]} / {arch}, {stage.lower()}: {action}")
        lines.append(f"| {names[variant]} / {arch} | {stage} | {check['archived']:.9f} | {check['reproduced']:.9f} | {check['delta']:+.9f} | {check['bootstrap_se']:.7f} | {check['hard_stop_threshold']:.7f} | {'OK' if check['within_hard_stop_limit'] else 'HARD STOP'} |")
    report["status"] = "hard_stop" if failures else "protocol_checks_complete"
    report["failed_checks"] = failures
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    diagnosis = json.loads((ROOT/"results/ladder_temp15_final_diagnosis.json").read_text())
    lines += ["", "### Trivial Quotient / GRU128 final diagnosis", "",
              f"Same CPU rows, ridge 1e-6: float64 R² {diagnosis['float64']['r2']:.12f}; float32 R² {diagnosis['float32']['r2']:.12f}. Float64-centered unregularized design condition number {diagnosis['float64']['centered_design_condition']:.6g}; ridge-augmented condition number {diagnosis['float64']['ridge_augmented_condition']:.6g}. Float32 centering changes these to {diagnosis['float32']['centered_design_condition']:.6g} and {diagnosis['float32']['ridge_augmented_condition']:.6g}.", "",
              f"10,000-resample held-fixed-probe episode-bootstrap global-R² SE: {diagnosis['global_bootstrap']['r2_se']:.9f}; 95% interval {diagnosis['global_bootstrap']['r2_ci95']}. Conditions on one training seed, the CPU histories and fixed fit probe.", "",
              diagnosis["history_device_independence"], "",
              "### One-class policies (both fit and test)", "",
              *["- "+flag for flag in flags], "",
              "For these cases, pooled within-class R² equals global R² for the same fitted probe: there is no action-class conditioning benefit or within-region-versus-between-region distinction. Other arms are not one-class on these rows.", "",
              "All 24 checks completed; four exceed the user-authorized hard-stop threshold. Conditional metrics, matched-init replays, controls, paired differences and the requested within-class figure were not computed. Checkpoint bytes and SHA-256 were verified before restoration. Both R² and archived global 1−R² deltas are retained in the JSON."]
    readme = ROOT/"results/README.md"
    previous = readme.read_text().split("## PR202: completed 24 recorded protocol checks")[0].rstrip()
    readme.write_text(previous+"\n\n"+"\n".join(lines)+"\n")
    print(f"Protocol checks: {len(report['checks'])} recorded, {len(failures)} hard-stop failures; {len(flags)} one-class checkpoints.")


if __name__ == "__main__":
    main()
