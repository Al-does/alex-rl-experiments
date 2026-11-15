"""Own-policy-primary report; no fixed-history curves."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .own_trajectory import ROOT, OUTPUT

HEADER="## PR219 own-policy-only checkpoint trajectory (primary follow-up)"
NAMES={1:"Trivial",2:"Reward-State",3:"Full-State"}


def format_ci(m):
    lo,hi=m["bootstrap"]["r2_ci95"]
    return f"{m['r2']:.6f} [{lo:.6f}, {hi:.6f}]"


def main():
    report=json.loads(OUTPUT.read_text());lines=[HEADER,"",
        "**Primary result: each encoder is scored only on its own greedy-policy trajectories.** These curves do not use final-policy fixed histories. Each checkpoint has its own belief distribution and action partition; compare these as distribution-specific affine accessibility, not causal information use.","",
        report["protocol"]["bootstrap"],"",report["protocol"]["selection"],"",report["protocol"]["class_standardization"],"",
        "Controls retain original own-policy targets/classes and use one fixed perturbation draw. Input controls are distribution shifts, not required-to-be-zero nulls. Init−peak is signed init minus peak, so a negative value indicates a rise.",""]
    for r in report["arms"]:
        name=NAMES[int(r["arm"].split("_")[1])]
        lines += [f"### {name}, epsilon {r['training_epsilon']}","",
            "| Steps | Within-class R² [95% CI] | Head-blind | Logits | Delta | Rho | Reward | Test classes |",
            "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for p in r["points"]:
            m=p["metrics"];f=m["full_h"]
            lines.append(f"| {p['step']} | {format_ci(f['belief'])} | {m['head_blind']['belief']['r2']:.6f} | {m['logits']['belief']['r2']:.6f} | {f['delta']['r2']:.6f} | {f['rho']['r2']:.6f} | {p['greedy_reward']['mean']:.6f} | {sum(n>0 for n in p['test_counts'])} |")
        lines += ["",f"Peak checkpoint: `{r['peak_checkpoint']}`.","",
            "| Contrast | Left R² | Right R² | Difference [unpaired 95% CI] |",
            "|---|---:|---:|---|"]
        for key,label in [("peak_minus_final","Peak − final"),("init_minus_peak","Init − peak"),("class_standardized_peak_minus_final","Peak − final, final class proportions on shared support")]:
            d=r[key];lo,hi=d["ci95"]
            lines.append(f"| {label} | {d['left_r2']:.6f} | {d['right_r2']:.6f} | {d['difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
        d=r["class_standardized_peak_minus_final"]
        if not d["full_final_class_mix_identifiable"]:
            lines += ["",f"**Full-final-class-mix pooled contrast is undefined:** peak has no samples for final classes {d['missing_peak_classes']}. The numeric standardized contrast below/above is restricted to shared support; it must not be interpreted as covering the entire final policy distribution."]
        lines += ["",f"Shared classes: {d['shared_classes']}; standardized weights: {d['final_proportions_on_shared_support']}. Final mass covered: {d['final_mass_covered']:.6f}; peak mass covered: {d['peak_mass_covered']:.6f}.","",
            "| Shared action | Peak R² | Final R² | Difference [unpaired 95% CI] |",
            "|---:|---:|---:|---|"]
        for k,m in d["per_class"].items():
            lo,hi=m["ci95"]
            lines.append(f"| {k} | {m['left_r2']:.6f} | {m['right_r2']:.6f} | {m['difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
        lines += ["","| Control stage / steps | Shuffle R² [CI] | Random input R² [CI] | Label permutation R² [CI] |","|---|---|---|---|"]
        for p in r["points"]:
            if p["controls"]:
                label="init" if p["step"]==0 else "peak" if p["step"]==r["peak_step"] else "final"
                lines.append(f"| {label} / {p['step']} | "+" | ".join(format_ci(p["controls"][k]) for k in ["history_shuffle","random_input","label_permutation"])+" |")
        lines += [""]
    lines += ["One-class points are marked with crosses in the figure; their class-centred full-h scores equal global affine scores. Per-class point estimates/intervals, all global scores, action counts, head-blind/logits/scalar/reward intervals and checkpoint provenance are in JSON. Class reweighting does not fix within-class visitation, representation scaling effects, or training-seed uncertainty.","",
        "Reproduce: with the documented CPU-only/B2 environment, run `uv run python -m experiments.within_class_belief_probes_2026_11.own_trajectory` then `uv run python -m experiments.within_class_belief_probes_2026_11.own_trajectory_report`. Own rollouts are reused; missing predictions/controls are cached under each run's ignored `own_trajectory/` directory. No training or paid compute."]
    path=ROOT/"results/README.md";path.write_text(path.read_text().split(HEADER)[0].rstrip()+"\n\n"+"\n".join(lines)+"\n")
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    colors={0.:"#2468a0",.05:"#df8a24"};styles={"full_h":"-","head_blind":"--","logits":":"}
    for r in report["arms"]:
        col=int(r["arm"].split("_")[1])-1;color=colors[r["training_epsilon"]]
        x=[p["step"]/1e9 for p in r["points"]]
        for feature,style in styles.items():
            metrics=[p["metrics"][feature]["belief"] for p in r["points"]]
            axes[0,col].plot(x,[m["r2"] for m in metrics],style,color=color,lw=1.6)
            axes[0,col].fill_between(x,[m["bootstrap"]["r2_ci95"][0] for m in metrics],[m["bootstrap"]["r2_ci95"][1] for m in metrics],color=color,alpha=.06)
        single=[p for p in r["points"] if p["one_class"]]
        axes[0,col].scatter([p["step"]/1e9 for p in single],[p["metrics"]["full_h"]["belief"]["r2"] for p in single],marker="x",s=24,color=color)
        reward=[p["greedy_reward"] for p in r["points"]]
        axes[1,col].plot(x,[m["mean"] for m in reward],color=color)
        axes[1,col].fill_between(x,[m["ci95"][0] for m in reward],[m["ci95"][1] for m in reward],color=color,alpha=.12)
        axes[0,col].set_title(NAMES[col+1]+" Quotient")
        axes[1,col].set_xlabel("Training steps (billions)");axes[1,col].set_ylim(-.02,1.02)
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Own-history within-class belief R²");axes[1,0].set_ylabel("Own greedy reward / step")
    handles=[Line2D([0],[0],color=c,label=f"ε={e:.2f}") for e,c in colors.items()]
    handles += [Line2D([0],[0],color="black",ls=s,label=f) for f,s in styles.items()]
    handles += [Line2D([0],[0],color="black",marker="x",ls="none",label="One class (full h)")]
    fig.suptitle("PR219 primary follow-up: own-policy histories only · seed 42\nPointwise 95% episode-bootstrap bands; independent R² axis ranges",fontsize=12)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.92),ncol=6,fontsize=9)
    fig.subplots_adjust(left=.07,right=.99,bottom=.08,top=.81,hspace=.13,wspace=.22)
    for ext in ["png","pdf"]:fig.savefig(OUTPUT.with_suffix('.'+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    print("Own-only report: 6 runs / 72 points / 18 control stages; README, PNG and vector PDF written.")


if __name__=="__main__":main()
