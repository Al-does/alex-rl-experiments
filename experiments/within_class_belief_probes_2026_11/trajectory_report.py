"""Render checkpoint inventories, trajectories and exploratory peak contrasts."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .trajectory import ROOT, OUTPUT

HEADER = "## PR219 checkpoint trajectories: own histories versus fixed final histories"
NAMES = {1:"Trivial Quotient",2:"Reward-State Quotient",3:"Full-State Quotient"}


def main():
    report=json.loads(OUTPUT.read_text())
    lines=[HEADER,"",
        "All six seed-42 transformer runs have 102 durable checkpoints. The full per-run inventory (step, B2 key, bytes, SHA-256 and selection flag) is in the trajectory JSON. Twelve checkpoints per run are analyzed using a prespecified hybrid of early geometric and late even index spacing, including init, saved midpoint and final. No decoding values were used to choose the sampling grid. Peaks refer only to sampled checkpoints; an unsampled peak or drop cannot be ruled out.","",
        "### B2 inventory", "",
        "Each prefix below contains `metadata/durability_manifest.json`; checkpoint keys, hashes and byte counts come from that canonical manifest. All selected checkpoint files are verified before restoration.","",
        "| Quotient | Epsilon | Saved / analyzed | B2 prefix |",
        "|---|---:|---:|---|"]
    for r in report["arms"]:
        name=NAMES[int(r["arm"].split("_")[1])]
        lines.append(f"| {name} | {r['training_epsilon']:.2f} | {len(r['checkpoint_inventory'])} / {len(r['points'])} | `{r['b2']['base_uri']}` |")
    inventories=[tuple(e["step"] for e in r["checkpoint_inventory"]) for r in report["arms"]]
    if len(set(inventories))==1:
        lines += ["", "Every run has the following saved steps (shared inventory; each run's corresponding B2 objects are enumerated separately in JSON):", "", "```text", ", ".join(map(str,inventories[0])),"```"]
    lines += ["", "### Sampled trajectories", "",
        "R² values below are pooled class-centred within-greedy-class full-h belief scores. Own histories use each checkpoint's policy streams/classes; fixed histories use final-policy streams/targets/classes and refit the probe for every encoder. Reward is the realized greedy reward per scored own-policy step, not a reward earned by an encoder on fixed actions. JSON also includes global R², head-blind, logits, delta/rho, class counts, one-class flags and pointwise episode-bootstrap intervals for all scores and reward.","",
        "| Quotient | Epsilon | Steps | Own-history R² | Fixed-history R² | Greedy reward | One-class own |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for r in report["arms"]:
        name=NAMES[int(r["arm"].split("_")[1])]
        for p in r["points"]:
            a=p["own_history"]["metrics"]["full_h"]["belief"]["r2"]
            b=p["fixed_history"]["metrics"]["full_h"]["belief"]["r2"]
            lines.append(f"| {name} | {r['training_epsilon']:.2f} | {p['step']} | {a:.6f} | {b:.6f} | {p['greedy_reward']['mean']:.6f} | {'Yes' if p['own_history']['one_class'] else 'No'} |")
    lines += ["", "### Fixed-history peak minus final", "",
        "Exploratory paired intervals condition on the observed selected peak and fitted probes; they are **not corrected for choosing the maximum**, do not cover unsampled checkpoints, and are not training-seed intervals. A final-checkpoint peak has exactly zero paired difference by construction.","",
        "| Quotient | Epsilon | Peak checkpoint steps | Peak − final | Paired 95% CI |",
        "|---|---:|---:|---:|---|"]
    for r in report["arms"]:
        d=r["fixed_peak_minus_final"];lo,hi=d["paired_difference_ci95"]
        lines.append(f"| {NAMES[int(r['arm'].split('_')[1])]} | {r['training_epsilon']:.2f} | {d['peak_step']} | {d['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    declines=sum(r["fixed_peak_minus_final"]["peak_step"] < r["final_reference_step"] and r["fixed_peak_minus_final"]["difference"] > 0 for r in report["arms"])
    lines += ["",f"Descriptively, {declines}/{len(report['arms'])} sampled fixed-history trajectories peak before final. This is not a claim of monotonic decline, statistical confirmation after maximum selection, or loss of causally used belief information. Own-history changes additionally reflect visitation and action-class changes; reward is a separate measure."]
    exclusions=[];one=[]
    for r in report["arms"]:
        for p in r["points"]:
            label=f"{NAMES[int(r['arm'].split('_')[1])]}, epsilon {r['training_epsilon']:.2f}, step {p['step']}"
            if p["own_history"]["one_class"]:one.append(label)
            for track in ["own_history","fixed_history"]:
                n=p[track]["metrics"]["full_h"]["belief"]["excluded_rows"]
                if n:exclusions.append(label+f", {track}: {n}")
    lines += ["", "**One-class own-history points:** "+("; ".join(one)+"." if one else "None."),"",
        "A one-class within-class score equals the corresponding global affine score. All fixed-history points inherit their final policy's classes, so both Trivial fixed-history tracks are one-class throughout. Class changes make the own-history denominator vary; fixed-history tracks keep that denominator and visitation fixed.","",
        "**Excluded predictions:** "+("; ".join(exclusions)+"." if exclusions else "None."),"",
        f"Uncertainty: {report['protocol']['point_bootstrap']} pointwise whole-test-episode bootstrap resamples; {report['protocol']['peak_bootstrap']} paired peak-minus-final resamples; probes held fixed and class means recomputed. The final scored episode is truncated to the 80K-row budget. One training seed only; no fit-data, seed-to-seed, simultaneous-curve or selected-peak correction. The observation/action streams and post-warmup scoring follow the original probe protocol. Affine accessibility does not establish causal use or Bayesian computation.","",
        "Reproduce with CPU-only environment variables and `uv run python -m experiments.within_class_belief_probes_2026_11.trajectory`, then `uv run python -m experiments.within_class_belief_probes_2026_11.trajectory_report`. Flags `--points`, `--bootstrap`, and `--peak-bootstrap` set the sampling budget and bootstrap counts. Caches are parameterized by points/bootstrap under each study/arm's ignored artifact directory. No new training or paid compute."]
    readme=ROOT/"results/README.md"
    readme.write_text(readme.read_text().split(HEADER)[0].rstrip()+"\n\n"+"\n".join(lines)+"\n")
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    colors={0.:"#2468a0",.05:"#df8a24"}
    for variant in [1,2,3]:
        col=variant-1
        group=[r for r in report["arms"] if r["arm"]==f"variant_{variant}_transformer"]
        for r in group:
            color=colors[r["training_epsilon"]]
            x=np.array([p["step"] for p in r["points"]])/1e9
            for track,style in [("own_history","-"),("fixed_history","--")]:
                data=[p[track]["metrics"]["full_h"]["belief"] for p in r["points"]]
                values=np.array([m["r2"] for m in data]);ci=np.array([m["bootstrap"]["r2_ci95"] for m in data])
                axes[0,col].plot(x,values,style,color=color,marker=".",ms=4)
                axes[0,col].fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.09)
            reward=[p["greedy_reward"] for p in r["points"]]
            axes[1,col].plot(x,[p["mean"] for p in reward],"-",color=color,marker=".",ms=4)
            axes[1,col].fill_between(x,[p["ci95"][0] for p in reward],[p["ci95"][1] for p in reward],color=color,alpha=.12)
        axes[0,col].set_title(NAMES[variant])
        axes[1,col].set_xlabel("Training steps (billions)")
        axes[1,col].set_ylim(-.02,1.02)
        for row in range(2):axes[row,col].grid(alpha=.2)
    axes[0,0].set_ylabel("Within-class belief R²")
    axes[1,0].set_ylabel("Own-policy greedy reward / step")
    handles=[Line2D([0],[0],color=colors[e],label=f"Training ε={e:.2f}") for e in colors]
    handles += [Line2D([0],[0],color="black",ls=s,label=label) for s,label in [("-","Own histories"),("--","Fixed final histories")]]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),ncol=4,fontsize=9)
    fig.suptitle("PR219: sampled checkpoint trajectories · seed 42 only\nPointwise 95% episode-bootstrap bands; Trivial final/fixed-history scores are global R²",fontsize=11)
    fig.subplots_adjust(left=.065,right=.99,bottom=.12,top=.82,hspace=.12,wspace=.2)
    fig.text(.5,.01,"R² panels use independent y-axis ranges; lines connect sampled checkpoints only.",ha="center",fontsize=8)
    fig.savefig(OUTPUT.with_suffix(".png"),dpi=180,bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"),bbox_inches="tight",pad_inches=.02)
    plt.close(fig)
    print(f"Reported {len(report['arms'])} runs, {sum(len(r['points']) for r in report['arms'])} trajectory points; README, PNG and vector PDF written.")


if __name__=="__main__":
    main()
