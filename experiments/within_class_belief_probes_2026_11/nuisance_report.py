"""Render the persistent-nuisance CPU results without restoring checkpoints."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .nuisance import ROOT, OUTPUT

NAMES = ["Trivial Quotient", "Reward-State Quotient", "Full-State Quotient"]
HEADER = "## Persistent-nuisance entropy 0.005: CPU within-class analysis"


def main():
    report = json.loads(OUTPUT.read_text())
    lines = [HEADER, "",
        "Three seed-42 transformer arms only, constant entropy 0.005, temperature 1.0, epsilon 0; no training. All checkpoints were verified against canonical durability-manifest SHA-256 and byte counts before restoration. Archive comparisons are annotations, not blockers, with threshold max(0.01, 5 × CPU episode-bootstrap SE) for both global R² and archived 1−R².", "",
        "**Target:** the study's full three-state filtered belief `(b1,b2,b3)`, not a nuisance-augmented latent product or a marginal. The persistent nuisance comes from the transition/emission structure. Scalar probes retain `delta=b1−b2` and `rho=b3`. In Trivial Quotient both coordinates are action-unnecessary; in Reward-State Quotient rho tracks the reward-state partition and delta is the within-nonreward-state contrast; in Full-State Quotient both contrasts can be action-relevant. These task partitions are distinct from the measured policy's greedy-action classes.", "",
        "### Archive protocol checks", "",
        "| Quotient | Stage | Step | Archived global R² | CPU global R² | Δ | Annotation |",
        "|---|---|---:|---:|---:|---:|---|"]
    for name, record in zip(NAMES, report["arms"]):
        for stage, data in record["stages"].items():
            c = data["archive_check"]
            lines.append(f"| {name} | {stage} | {data['step']} | {c['archived']:.9f} | {c['reproduced']:.9f} | {c['delta']:+.9f} | {c['annotation'] or '—'} |")
    lines += ["", "### Final versus matched initialization", "",
        "All columns are pooled class-centred within-class-refitted held-out R². Other than matched init and its paired contrast, columns describe the final trained encoder. Intervals resample whole test episodes 10,000 times with probes held fixed; **one training seed only**, not training-seed uncertainty.", "",
        "| Quotient | Final h | Matched init | Difference [95% CI] | Head-blind | Logits | Delta | Rho | Shuffle | Random | Labels | One-class final | Archive annotation |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for name, r in zip(NAMES, report["arms"]):
        f, m = r["stages"]["final"], r["matched_init"]
        d = r["paired_differences"]["full_h"]["belief"]
        lo, hi = d["paired_difference_ci95"]
        final = f["metrics"]["full_h"]["belief"]["r2"]
        init = m["metrics"]["full_h"]["belief"]["r2"]
        values = [f["metrics"][a][b]["r2"] for a, b in [("head_blind", "belief"), ("logits", "belief"), ("full_h", "delta"), ("full_h", "rho")]]
        values += [f["controls"][key]["belief"]["r2"] for key in ["history_shuffle", "random_input", "label_permutation"]]
        annotation = "; ".join(stage+": "+s["archive_check"]["annotation"] for stage, s in r["stages"].items() if s["archive_check"]["annotation"]) or "—"
        lines.append(f"| {name} | {final:.4f} | {init:.4f} | {d['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "+" | ".join(f"{v:.4f}" for v in values)+f" | {'Yes' if f['one_class'] else 'No'} | {annotation} |")
    lines += ["", "### Own-policy initialization (secondary)", "",
              "| Quotient | Full h | Head-blind | Logits | Delta | Rho |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, r in zip(NAMES, report["arms"]):
        m = r["stages"]["init"]["metrics"]
        values = [m[a][b]["r2"] for a, b in [("full_h", "belief"), ("head_blind", "belief"), ("logits", "belief"), ("full_h", "delta"), ("full_h", "rho")]]
        lines.append("| "+name+" | "+" | ".join(f"{v:.4f}" for v in values)+" |")
    annotations, one_class, exclusions = [], [], []
    for name, record in zip(NAMES, report["arms"]):
        for stage, data in record["stages"].items():
            if data["archive_check"]["annotation"]:
                annotations.append(name+" "+stage+": "+data["archive_check"]["annotation"])
            if data["one_class"]:
                one_class.append(name+" "+stage)
            count = data["metrics"]["full_h"]["belief"]["excluded_rows"]
            if count:
                exclusions.append(f"{name} {stage}: {count} rows")
    lines += ["", "**Separate archive-discrepancy block:** "+("; ".join(annotations) if annotations else "No checks exceeded the annotation threshold."), "",
        "**One-class checkpoint flags:** "+("; ".join(one_class)+"." if one_class else "None.")+" For a one-class checkpoint, pooled within-class R² equals global R² for the same fitted probe, not additional action-conditioned evidence.", "",
        "**Excluded predictions:** "+("; ".join(exclusions)+"." if exclusions else "None across saved stages.")+" Counts and per-class metrics are explicit in JSON.", "",
        "Matched init consumes exactly the final policy's token/previous-action stream from an empty transformer cache per independent episode, including all 64 warmup steps. It retains final beliefs, greedy classes and scored rows. History shuffle preserves reset and complete token/action pairs; random input samples from separate reset/nonreset empirical pair pools; label permutation changes fit labels within class only. These are frozen-input controls, not activation interventions. Each uses one fixed random draw.", "",
        "The head-blind probe removes the two action-logit-difference directions (62 dimensions remain), retaining common logit shifts. Each encoder uses its own policy head. Probe fit is float64 centered ridge least squares (ridge 1e−6). Last scored episodes are truncated to the 60K/80K budgets. Bootstrap uncertainty conditions on these CPU histories, fixed fit probes and seed 42; there is no across-training-seed inference. Accessibility differences do not establish causal use or Bayesian computation.", "",
        "Provenance: records imported with `git archive` from `"+report["provenance"]["source_ref"]+"` at `"+report["provenance"]["source_revision"]+"`, not merged/rebased. Only entropy-0.005 arm records and small shared recipe sources were brought in. The local greedy collector is equivalent for temperature 1/epsilon 0; upstream's only base collector change adds epsilon support. JSON includes source/runtime/checkpoint hashes, archived and CPU scores, all per-class/scalar/control scores, matched replay, and paired intervals."]
    readme = ROOT/"results/README.md"
    previous = readme.read_text().split(HEADER)[0].rstrip()
    readme.write_text(previous+"\n\n"+"\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(3)
    for offset, kind, label, color in [(-.12, "final", "Final trained", "#2468a0"), (.12, "matched_init", "Init on final histories", "#df8a24")]:
        values = [(r["stages"]["final"] if kind == "final" else r[kind])["metrics"]["full_h"]["belief"]["r2"] for r in report["arms"]]
        axes[0].plot(x+offset, values, "o", label=label, color=color)
    axes[0].set_ylabel("Pooled within-class belief R²")
    axes[0].legend(loc="best", fontsize=8)
    ds = [r["paired_differences"]["full_h"]["belief"] for r in report["arms"]]
    point = np.array([d["difference"] for d in ds])
    bounds = np.array([d["paired_difference_ci95"] for d in ds])
    axes[1].errorbar(x, point, yerr=np.stack([point-bounds[:, 0], bounds[:, 1]-point]), fmt="o", capsize=4, color="#46934a")
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_ylabel("Final − matched-init R² (95% paired CI)")
    for ax in axes:
        ax.set_xticks(x, [name.replace(" Quotient", "")+("†" if r["stages"]["final"]["one_class"] else "")+("*" if any(s["archive_check"]["annotation"] for s in r["stages"].values()) else "") for name, r in zip(NAMES, report["arms"])])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Persistent-nuisance · entropy 0.005 · seed 42 only\n† one-class final (within-class = global); * archive discrepancy annotated", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print("Reported 3 arms, 9 checkpoint stages, 3 matched-init comparisons; README appended and PNG written.")


if __name__ == "__main__":
    main()
