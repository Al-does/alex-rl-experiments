"""Compact tables and figure for the symmetry-dose within-class probes (reads ignored raw analyses)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "artifacts/probe"
RESULTS = HERE / "results"
REWARD_GATE = .95
TARGETS = ("belief", "delta", "rho", "q_contrast")
LABEL = {"belief": "full belief", "delta": r"$b_1-b_2$", "rho": r"$b_3$", "q_contrast": "action-value contrasts"}


def arm_key(lam, kappa):
    return f"kappa={kappa:g}" if kappa else f"lam={lam:g}"


def reference(theory, lam, kappa):
    return theory["kappa"][f"{kappa:g}" if f"{kappa:g}" in theory["kappa"] else str(kappa)] if kappa else theory["lam"][
        next(k for k in theory["lam"] if float(k) == lam)]


def compact(analysis, theory):
    ref = reference(theory, analysis["lam"], analysis["kappa"])
    reward = analysis["reward"]["expected_per_step"]
    return {
        "run_id": analysis["run_id"], "study": analysis["study"], "arm": analysis["arm"], "seed": analysis["seed"],
        "lam": analysis["lam"], "kappa": analysis["kappa"], "final_step": analysis["final_step"],
        "checkpoint_sha256": {k: v["checkpoint"]["sha256"] for k, v in analysis["provenance"].items()},
        "source_git": analysis["provenance"]["final"]["source_git"],
        "reward_per_step": reward, "reward_se": analysis["reward"]["se"],
        "archived_gpu_reward_per_step": analysis["reward"]["archived_gpu_per_step"],
        "belief_mdp_reward": ref["optimal_reward"], "reward_fraction": reward / ref["optimal_reward"],
        "reward_qualified": bool(reward >= REWARD_GATE * ref["optimal_reward"]),
        "test_class_counts": analysis["test_class_counts"],
        "greedy_agrees_with_belief_mdp": analysis["greedy_agrees_with_belief_mdp"],
        "trained": analysis["trained"], "matched_init": analysis["matched_init"],
        "paired_differences": analysis["paired_differences"], "global_trained": analysis["global_trained"],
        "logit_margin_vs_q_margin_spearman": analysis["logit_margin_vs_q_margin_spearman"],
        "by_q_margin_tercile": analysis["by_q_margin_tercile"],
    }


def aggregate(runs):
    """Seed-level mean, sample SD and range per arm; no pooling of rows across seeds."""
    arms = {}
    for run in runs:
        arms.setdefault(arm_key(run["lam"], run["kappa"]), []).append(run)
    out = {}
    for name, members in arms.items():
        def stats(values):
            values = np.asarray(values, dtype=float)
            return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "min": float(values.min()), "max": float(values.max()), "n": len(values),
                    "positive": int((values > 0).sum())}
        out[name] = {
            "lam": members[0]["lam"], "kappa": members[0]["kappa"],
            "seeds": sorted(m["seed"] for m in members),
            "reward_qualified_seeds": sorted(m["seed"] for m in members if m["reward_qualified"]),
            "reward_fraction": stats([m["reward_fraction"] for m in members]),
            "action_classes_used": stats([sum(c > 0 for c in m["test_class_counts"]) for m in members]),
            "trained": {f: {t: stats([m["trained"][f][t] for m in members]) for t in TARGETS}
                        for f in ("full_h", "head_blind", "logits")},
            "matched_init": {f: {t: stats([m["matched_init"][f][t] for m in members]) for t in TARGETS}
                             for f in ("full_h", "head_blind", "logits")},
            "gain": {f: {t: stats([m["paired_differences"][f][t]["difference"] for m in members]) for t in TARGETS}
                     for f in ("full_h", "head_blind", "logits")},
            "logit_margin_vs_q_margin_spearman": {
                k: stats([m["logit_margin_vs_q_margin_spearman"][k] for m in members
                          if m["logit_margin_vs_q_margin_spearman"][k] is not None] or [np.nan])
                for k in ("trained", "matched_init")},
        }
    return out


def order(arms):
    return sorted(arms, key=lambda k: (arms[k]["kappa"], -arms[k]["lam"]))


def markdown(arms, theory):
    lines = ["| arm | seeds (reward-qualified) | reward / belief-MDP | delta-blind reward loss | classes | "
             "gain b1-b2 | gain b3 | gain belief | gain Q-contrast | head-blind gain b1-b2 | logits gain b1-b2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    def cell(s):
        return f"{s['mean']:+.3f} [{s['min']:+.3f}, {s['max']:+.3f}] ({s['positive']}/{s['n']}+)"
    for name in order(arms):
        a = arms[name]
        ref = reference(theory, a["lam"], a["kappa"])
        g = a["gain"]
        lines.append(f"| {name} | {len(a['seeds'])} ({len(a['reward_qualified_seeds'])}) | "
                     f"{a['reward_fraction']['mean']:.3f} | {ref['lumped_filter_loss'] / ref['optimal_reward']:.1%} | "
                     f"{a['action_classes_used']['min']:.0f}-{a['action_classes_used']['max']:.0f} | "
                     + " | ".join(cell(g["full_h"][t]) for t in ("delta", "rho", "belief", "q_contrast"))
                     + f" | {cell(g['head_blind']['delta'])} | {cell(g['logits']['delta'])} |")
    return "\n".join(lines) + "\n"


def figure(runs, arms, theory, trajectories):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42})
    names = order(arms)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 1.15]})
    ax = axes[0]
    for offset, target, color in ((-.17, "delta", "C3"), (.17, "rho", "C0")):
        for i, name in enumerate(names):
            members = [r for r in runs if arm_key(r["lam"], r["kappa"]) == name]
            values = [m["paired_differences"]["full_h"][target]["difference"] for m in members]
            ok = np.array([m["reward_qualified"] for m in members])
            values = np.asarray(values)
            ax.scatter(np.full(ok.sum(), i + offset), values[ok], s=9, color=color, alpha=.7, linewidths=0)
            ax.scatter(np.full((~ok).sum(), i + offset), values[~ok], s=9, facecolors="none", edgecolors=color, linewidths=.6)
            ax.plot([i + offset - .12, i + offset + .12], [values.mean()] * 2, color=color, lw=1.4,
                    label=LABEL[target] if i == 0 else None)
    ax.axhline(0, color="k", lw=.5)
    ax.set_xticks(x, [n.replace("lam=", r"$\lambda$=").replace("kappa=", r"$\kappa$=") for n in names], rotation=45, ha="right")
    ax.set_ylabel(r"within-class $R^2$: trained $-$ matched init")
    ax.legend(frameon=False)
    ax = axes[1]
    loss = [reference(theory, arms[n]["lam"], arms[n]["kappa"]) for n in names]
    ax.bar(x, [100 * r["lumped_filter_loss"] / r["optimal_reward"] for r in loss], color="0.5")
    ax.set_xticks(x, [n.replace("lam=", r"$\lambda$=").replace("kappa=", r"$\kappa$=") for n in names], rotation=45, ha="right")
    ax.set_ylabel(r"reward lost without $b_1-b_2$ (%)")
    ax = axes[2]
    for name, color in (("lam=1", "C0"), ("lam=-1", "C3")):
        first = True
        for t in trajectories:
            if arm_key(t["lam"], t["kappa"]) != name:
                continue
            steps = np.array([p["step"] for p in t["points"]]) / 1e6
            values = np.array([p["within_final_class_r2"]["delta"] for p in t["points"]])
            ax.plot(np.maximum(steps, 10), values - values[0], color=color, lw=.7, alpha=.8,
                    label=(r"$\lambda$=1 (Reward-State)" if name == "lam=1" else r"$\lambda$=$-$1 (Full-State)") if first else None)
            first = False
    ax.axhline(0, color="k", lw=.5)
    ax.set_xscale("log")
    ax.set_xlabel("training steps (millions; init at 10)")
    ax.set_ylabel(r"$b_1-b_2$ within-class $R^2$ $-$ init")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS / f"fig_within_class_symmetry_dose.{suffix}", bbox_inches="tight", pad_inches=.02, dpi=300)
    plt.close(fig)


def main():
    theory = json.loads((RESULTS / "theory.json").read_text())
    runs = sorted((compact(json.loads(p.read_text()), theory) for p in RAW.glob("*/analysis.json")),
                  key=lambda r: (r["kappa"], -r["lam"], r["seed"]))
    arms = aggregate(runs)
    trajectories = sorted((json.loads(p.read_text()) for p in RAW.glob("*/trajectory.json")), key=lambda t: t["run_id"])
    (RESULTS / "dose_probe.json").write_text(json.dumps(
        {"reward_gate": REWARD_GATE, "protocol": "60,000 fit / 80,000 test rows, 64-step warmup, class-specific affine ridge 1e-6, "
         "class-centred held-out R2, classes and histories from the final greedy policy, matched init replayed on the same histories",
         "arms": arms, "runs": runs}, indent=1, allow_nan=False) + "\n")
    (RESULTS / "dose_trajectory.json").write_text(json.dumps(trajectories, indent=1, allow_nan=False) + "\n")
    table = markdown(arms, theory)
    (RESULTS / "dose_table.md").write_text(table)
    figure(runs, arms, theory, trajectories)
    print(table)


if __name__ == "__main__":
    main()
