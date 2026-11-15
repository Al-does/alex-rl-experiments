"""Tables and matched-history figures for PR219's six frozen policies."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .nuisance_v2 import ROOT, OUTPUT

HEADER = "## Persistent-nuisance v2: temperature 1.5 with training epsilon 0 / 0.05"
NAMES = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}


def name(r):
    return NAMES[int(r["arm"].split("_")[1])]


def main():
    report = json.loads(OUTPUT.read_text())
    arms = report["arms"]
    lines = [HEADER, "",
        "Six seed-42 transformer policies only, target 2B steps, entropy 0, training temperature 1.5, training exploration epsilon 0 or 0.05. All probes are greedy argmax: both temperature and epsilon are ignored. No training, paid compute, commits or pushes.", "",
        "The source target remains the full filtered **three-state** belief, not a marginal or additional nuisance factor. Delta=b1−b2 and rho=b3 remain valid: delta contrasts the nonreward states; rho is reward-state probability. Both are action-unnecessary in Trivial Quotient; delta is the within-nonreward contrast in Reward-State Quotient; both may matter for Full-State actions. Classes are the policy's own greedy actions, not the latent partition.", "",
        "The v2 environment uses baseline transition rows (.995505,.002495,.002), (.002495,.995505,.002), (.005,.005,.99); emission rows (.1125,.0375,.85), (.0375,.1125,.85), (0,0,1); uniform initial belief; reward on pre-transition state 3; tilt strength 2.5. Inputs remain token/previous-action one-hots. The shared probe, models, env/filter implementation and training dispatch match the local base files; PR219's collector changes affect non-greedy training sampling/likelihoods only. Greedy key splitting and argmax are unchanged.", "",
        "### Archive checks (annotations only)", "",
        "Threshold max(0.01,5×CPU episode-bootstrap SE), checked for both global R² and archived 1−R². All checkpoint bytes and SHA-256 hashes are verified against canonical durability manifests before restoration. No discrepancy blocks analysis.", "",
        "| Quotient | Training epsilon | Stage | Step | Archived R² | CPU R² | Δ | Annotation |",
        "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in arms:
        for stage, s in r["stages"].items():
            c = s["archive_check"]
            lines.append(f"| {name(r)} | {r['training_epsilon']:.2f} | {stage} | {s['step']} | {c['archived']:.9f} | {c['reproduced']:.9f} | {c['delta']:+.9f} | {c['annotation'] or '—'} |")
    lines += ["", "### Final versus matched initialization", "",
        "Pooled within-greedy-class held-out R², class-specific refits and class-centred test denominators. Other than matched init and its paired contrast, columns describe the final trained encoder. Intervals use 10,000 paired episode resamples with probes fixed: **one training seed only**.", "",
        "| Quotient | Epsilon | Final h | Matched init | Difference [95% CI] | Head-blind | Logits | Delta | Rho | Shuffle | Random | Labels | One-class final | Archive annotation |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in arms:
        f, m = r["stages"]["final"], r["matched_init"]
        d = r["paired_differences"]["full_h"]["belief"]
        lo, hi = d["paired_difference_ci95"]
        values = [f["metrics"][a][b]["r2"] for a,b in [("head_blind","belief"),("logits","belief"),("full_h","delta"),("full_h","rho")]]
        values += [f["controls"][key]["belief"]["r2"] for key in ["history_shuffle","random_input","label_permutation"]]
        annotation = "; ".join(stage+": "+s["archive_check"]["annotation"] for stage,s in r["stages"].items() if s["archive_check"]["annotation"]) or "—"
        lines.append(f"| {name(r)} | {r['training_epsilon']:.2f} | {f['metrics']['full_h']['belief']['r2']:.4f} | {m['metrics']['full_h']['belief']['r2']:.4f} | {d['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "+" | ".join(f"{v:.4f}" for v in values)+f" | {'Yes' if f['one_class'] else 'No'} | {annotation} |")
    lines += ["", "### Own-policy initialization (secondary)", "",
              "| Quotient | Epsilon | Full h | Head-blind | Logits | Delta | Rho |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in arms:
        m=r["stages"]["init"]["metrics"]
        values=[m[a][b]["r2"] for a,b in [("full_h","belief"),("head_blind","belief"),("logits","belief"),("full_h","delta"),("full_h","rho")]]
        lines.append(f"| {name(r)} | {r['training_epsilon']:.2f} | "+" | ".join(f"{v:.4f}" for v in values)+" |")
    annotations, flags, exclusions = [], [], []
    for r in arms:
        for stage,s in r["stages"].items():
            label=f"{name(r)}, epsilon {r['training_epsilon']:.2f}, {stage}"
            if s["archive_check"]["annotation"]:annotations.append(label+": "+s["archive_check"]["annotation"])
            if s["one_class"]:flags.append(label)
            n=s["metrics"]["full_h"]["belief"]["excluded_rows"]
            if n:exclusions.append(label+f": {n} rows")
    lines += ["", "**Separate archive-discrepancy block:** "+("; ".join(annotations) if annotations else "None exceeded the annotation threshold.")+"", "",
        "**One-class stages:** "+("; ".join(flags)+"." if flags else "None.")+" Within-class R² equals global R² for a one-class policy; this is not additional action-conditioned evidence.", "",
        "**Excluded predictions:** "+("; ".join(exclusions)+"." if exclusions else "None across saved stages.")+" Full counts and per-class scores are retained in JSON.", "",
        "### Final-policy action counts", "",
        "Counts are [Noop, Tilt X, Tilt Y], across the scored fit/test rows.", ""]
    for r in arms:
        f=r["stages"]["final"]
        lines.append(f"- {name(r)}, epsilon {r['training_epsilon']:.2f}: fit {f['fit_counts']}; test {f['test_counts']}.")
    lines += ["", "Matched initialization replays exact final-policy token/previous-action streams with empty transformer caches per episode, including warmup; final beliefs, classes and scored rows stay fixed. History shuffle preserves resets and whole input pairs; random input samples separate reset/nonreset empirical pair pools; fit labels are permuted within class. Original targets/classes are retained; each control uses one fixed draw.", "",
        "Protocol: post-final-LayerNorm 64-D embeddings, 60K fit / 80K test rows, 64-step warmup, 1024-step episodes, ridge 1e−6, original split(fold_in(PRNGKey(42),2)) streams. Head-blinding removes two action-logit-difference directions (62 remain), retaining common logit shifts; each encoder uses its own head. Episode bootstrap recomputes class means but holds probes fixed, conditioning on the CPU histories and one seed. Last scored episodes are truncated to the row budgets. No training-seed or between-epsilon causal inference is supported. Accessibility does not establish causal use or Bayesian computation.", "",
        "Source: PR219 branch `"+report["provenance"]["source_branch"]+"` at `"+report["provenance"]["source_revision"]+"`. Imported selected transformer records and recipe/environment sources using git archive, no merge/rebase. The JSON includes hashes/bytes and source/runtime provenance, init/intermediate/final per-class/scalar/control scores, matched-init metrics and paired intervals. Checkpoint caches are separated by study to avoid collisions between epsilon settings."]
    readme=ROOT/"results/README.md"
    readme.write_text(readme.read_text().split(HEADER)[0].rstrip()+"\n\n"+"\n".join(lines)+"\n")
    fig, axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey="row")
    for col,epsilon in enumerate([0.,.05]):
        group=[r for r in arms if r["training_epsilon"]==epsilon]
        x=np.arange(3)
        for offset,kind,label,color in [(-.1,"final","Final trained","#2468a0"),(.1,"matched_init","Init on final histories","#df8a24")]:
            values=[(r["stages"]["final"] if kind=="final" else r[kind])["metrics"]["full_h"]["belief"]["r2"] for r in group]
            axes[0,col].plot(x+offset,values,"o",label=label,color=color)
        ds=[r["paired_differences"]["full_h"]["belief"] for r in group]
        points=np.array([d["difference"] for d in ds]);bounds=np.array([d["paired_difference_ci95"] for d in ds])
        axes[1,col].errorbar(x,points,yerr=np.stack([points-bounds[:,0],bounds[:,1]-points]),fmt="o",capsize=4,color="#46934a")
        axes[1,col].axhline(0,color="gray",lw=1)
        axes[0,col].set_title(f"Training epsilon {epsilon:.2f}")
        axes[0,col].legend(fontsize=8)
        labels=[name(r).replace(" Quotient","")+("†" if r["stages"]["final"]["one_class"] else "")+("*" if any(s["archive_check"]["annotation"] for s in r["stages"].values()) else "") for r in group]
        axes[1,col].set_xticks(x,labels)
        for row in range(2):axes[row,col].grid(axis="y",alpha=.2)
    axes[0,0].set_ylabel("Pooled within-class belief R²")
    axes[1,0].set_ylabel("Final − matched-init R² (95% paired CI)")
    fig.suptitle("Persistent-nuisance v2 · temperature 1.5 · seed 42 only\n† one-class final (within-class = global); * archive discrepancy annotated",fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".png"),dpi=180)
    plt.close(fig)
    print("Reported 6 arms, 18 checkpoint stages, 6 matched-init comparisons; README appended and PNG written.")


if __name__ == "__main__":
    main()
