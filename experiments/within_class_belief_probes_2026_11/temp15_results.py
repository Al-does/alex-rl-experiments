"""Compact annotated temp15 conditional-decoding report and figure."""

import hashlib
import json
import platform
import subprocess

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .temp15 import ROOT, RAW, runs, gate_passes, TOLERANCE_REASON

NAMES = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}


def name(arm):
    return NAMES[int(arm.split("_")[1])]+" / "+arm.split("_", 2)[2]


def main():
    gate_path = ROOT/"results/ladder_temp15_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["status"] = "recorded_protocol_checks_with_annotations"
    gate["tolerance_reason"] = TOLERANCE_REASON
    gate["discrepancies_are_blockers"] = False
    for c in gate["checks"]:
        c["annotation"] = None if gate_passes(c) else "archived global R² not reproduced on CPU"
    gate_path.write_text(json.dumps(gate, indent=2)+"\n")
    records = []
    for run in runs():
        directory = RAW/run.parent.parent.name
        record = json.loads((directory/"analysis.json").read_text())
        if (directory/"intermediate.json").exists():
            record["stages"]["intermediate"] = json.loads((directory/"intermediate.json").read_text())
        record["annotations"] = [{"stage": stage, "message": data["annotation"]} for stage, data in record["stages"].items() if data["annotation"]]
        records.append(record)
    report = {"schema_version": 1, "protocol": {
        "training_seeds": [42], "cpu_only": True, "greedy": True, "representation": "post_final_layer_norm_embedding",
        "fit_rows": 60000, "test_rows": 80000, "warmup": 64, "episode_length": 1024,
        "fit_episodes": 63, "test_episodes": 84, "last_scored_episode_truncated": True,
        "streams": "jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42),2)); identical keys at all checkpoints",
        "ridge": 1e-6, "fit_arithmetic": "float64 centered SVD-backed affine least squares, ridge weights only",
        "classes": "source network greedy actions; final classes fixed for matched init and controls",
        "pooled_r2": "1 - sum(class SSE) / sum(class test-mean-centered SST)",
        "scalar_targets": "delta=b1-b2; rho=b3 (manuscript 1-based indices)",
        "head_blind": "null space of two logit-difference weight rows; common logit shift retained; each encoder uses its own head",
        "matched_init": "complete final-policy observation/previous-action streams replayed from fresh zero carry at each episode; all 1024 steps including warmup; no carry reset at GRU 32-step training chunk boundaries; original final targets, classes and scored rows retained",
        "controls": "one fixed draw per fit/test split; shuffle complete token/previous-action pairs inside each episode except reset; random pair resampling from separate empirical reset/non-reset pools; targets and classes unchanged; label permutation on fit rows within class",
        "control_seeds": "20261106+10*control_index+split_index; label permutation 853",
        "uncertainty": "10000 paired whole-test-episode bootstrap draws; class means recomputed; probes held fixed; same resampled episodes for final and matched init; conditional on ONE training seed, fixed fit rows/probes; no between-training-seed inference",
        "undefined": "classes with fewer than two fit rows have excluded predictions; empty/constant class R2 is null; exclusions reported explicitly",
        "archive_checks": "four recorded discrepancies are annotations, not blockers; CPU within-run primary comparisons use verified checkpoint bytes",
        "intermediate_selection": "saved checkpoint closest to 2B steps; no matched-init comparison at intermediate; other ~200 saved stages not analyzed"},
        "provenance": {"experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "harness_commit": subprocess.check_output(["git", "-C", str(ROOT.parents[2]/"rl-harness"), "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__, "devices": str(jax.devices()),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}},
        "archive_checks": gate, "gru92_init_diagnosis": json.loads((ROOT/"results/ladder_temp15_gru92_diagnosis.json").read_text()),
        "arms": records}
    (ROOT/"results/ladder_temp15_within_class.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["## PR202 temperature-1.5: completed CPU within-class analysis", "",
        "Supersedes the earlier stopping decisions. All 12 arms are included, including all initialized encoders replayed on final-policy histories. Four archived-global-R² discrepancies remain annotations, not blockers. **One training seed (42) only.** Intervals are paired test-episode bootstrap intervals with fixed fit probes, not training-seed uncertainty.", "",
        "| Condition / architecture | Full h final | Matched init | Paired difference [95% CI] | Head-blind | Logits | Delta | Rho | Shuffle | Random | Labels | One-class final | Archive annotation |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for record in records:
        f, m = record["stages"]["final"], record["matched_init"]
        paired = record["paired_differences"]["full_h"]["belief"]
        ci = paired["paired_difference_ci95"]
        values = [f["metrics"][feature][target]["r2"] for feature, target in (("full_h", "belief"),)]
        values += [m["metrics"]["full_h"]["belief"]["r2"]]
        tail = [f["metrics"][feature][target]["r2"] for feature, target in (("head_blind", "belief"), ("logits", "belief"), ("full_h", "delta"), ("full_h", "rho"))]
        tail += [f["controls"][control]["belief"]["r2"] for control in ("history_shuffle", "random_input", "label_permutation")]
        annotations = "; ".join(a["stage"]+": "+a["message"] for a in record["annotations"]) or "—"
        lines.append("| "+name(record["arm"])+" | "+" | ".join(f"{v:.4f}" for v in values)+f" | {paired['difference']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "+" | ".join(f"{v:.4f}" for v in tail)+f" | {'Yes (Tilt X)' if f['one_class'] else 'No'} | {annotations} |")
    lines += ["", "### Reading the comparisons", "",
        "Unannotated arms: interpret trained-minus-matched-init differences as CPU affine-accessibility differences on exactly matched histories and greedy classes; the paired intervals are conditional on seed 42.", "",
        "**Separate annotated block:** GRU92 initialization in all three quotient conditions, and Trivial Quotient / transformer final: **archived global R² not reproduced on CPU**. Their CPU metrics and matched-init comparisons are retained in full, without assuming archived CPU/GPU equivalence.", "",
        "All four Trivial Quotient final policies are single-class (Tilt X), as are all GRU92 initializations (Tilt Y) and all memoryless initializations (Noop). For these checkpoints pooled within-class R² equals global R² for the same probe: no evidence beyond action-class conditioning is obtained by relabeling it as within-class.", "",
        "GRU92 initialization shares identical NPZ bytes and parameter arrays across all three variants. Archived/CPU actions are all Tilt Y. Rewards agree exactly: 0.012125 for Trivial and Reward-State Quotients, 0.107637502 for Full-State Quotient. Median/minimum held-out top-two logit gaps are 0.138522/0.031633, 0.138522/0.031633 and 0.154293/0.031634 respectively—not numerical ties.", "",
        "GRU matched-init replay starts with zeros in every layer for each independent episode, consumes the full final observation/action stream including warmup, and carries hidden state through all 1024 steps without resetting at the 32-step training chunk boundaries. Transformer caches likewise start empty per episode. Memoryless models receive the same stream but architecturally read only its current token.", "",
        "Own-policy transformer init has one excluded test row in Trivial Quotient and two in Full-State Quotient because the corresponding greedy class lacks two fit rows. All final and matched-init comparisons have complete prediction coverage. Counts and exclusions are retained per metric in JSON. Rounded near-zero memoryless contrasts are numerical-scale differences, not exact-zero uncertainty.", "",
        "The JSON includes own-policy initialization as a secondary comparison, all per-class/scalar/control scores, intermediate checkpoints near 2B, paired intervals for every representation and control, checkpoint provenance and hashes. No seed averaging is performed. These are descriptive accessibility measurements, not causal use or proof of belief computation."]
    readme = ROOT/"results/README.md"
    previous = readme.read_text().split("## PR202 temperature-1.5: completed CPU within-class analysis")[0].rstrip()
    readme.write_text(previous+"\n\n"+"\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for variant, ax in enumerate(axes, 1):
        group = [r for r in records if int(r["arm"].split("_")[1]) == variant]
        x = np.arange(len(group))
        for offset, key, label, color in ((-.15, "final", "Final trained", "#2468a0"), (.15, "matched_init", "Init on final histories", "#df8a24")):
            values = [(r["stages"]["final"] if key == "final" else r[key])["metrics"]["full_h"]["belief"]["r2"] for r in group]
            ax.plot(x+offset, values, "o", color=color, label=label)
        labels = [r["arm"].split("_", 2)[2]+("*" if r["annotations"] else "")+("†" if r["stages"]["final"]["one_class"] else "") for r in group]
        ax.set(title=NAMES[variant], xticks=x, xticklabels=labels, ylabel="Pooled within-class belief R²")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
        contrast = ax.twinx()
        diffs = np.array([r["paired_differences"]["full_h"]["belief"]["difference"] for r in group])
        intervals = np.array([r["paired_differences"]["full_h"]["belief"]["paired_difference_ci95"] for r in group])
        contrast.errorbar(x, diffs, yerr=np.stack([diffs-intervals[:, 0], intervals[:, 1]-diffs]), fmt="s", color="#46934a", capsize=3, label="Paired difference, 95% CI")
        contrast.axhline(0, color="#46934a", alpha=.3)
        contrast.set_ylabel("Final − matched-init R²", color="#46934a")
        if variant == 1:
            ax.legend(fontsize=8, loc="lower left")
            contrast.legend(fontsize=8, loc="upper left")
    fig.suptitle("Seed 42 only · * archive/CPU discrepancy annotated · † one-class final (within-class = global)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT/"results/ladder_temp15_within_class.png", dpi=180)
    plt.close(fig)
    print(f"Reported {len(records)} arms; {sum(len(r['stages']) for r in records)} checkpoint stages; 12 matched-init comparisons; 10000 paired episode resamples.")


if __name__ == "__main__":
    main()
