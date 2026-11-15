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
    if lam is None:
        return "trivial"
    return f"kappa={kappa:g}" if kappa else f"lam={lam:g}"


def position(lam, kappa):
    """Table order: Trivial Quotient, then lam from Reward-State to Full-State, then the kappa arms."""
    return (lam is not None, kappa, 0. if lam is None else -lam)


def loss(ref):
    """Reward given up by the filter that never separates states 1 and 2, with a conservative Monte Carlo SE."""
    return {"fraction": ref["lumped_filter_loss"] / ref["optimal_reward"],
            "absolute": ref["lumped_filter_loss"],
            "se": float(np.hypot(ref["optimal_reward_se"], ref["lumped_filter_reward_se"]))}


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def reference(theory, lam, kappa):
    if lam is None:
        return theory["trivial"]
    return theory["kappa"][f"{kappa:g}" if f"{kappa:g}" in theory["kappa"] else str(kappa)] if kappa else theory["lam"][
        next(k for k in theory["lam"] if float(k) == lam)]


def compact(analysis, theory, trajectory):
    ref = reference(theory, analysis["lam"], analysis["kappa"])
    reward = analysis["reward"]["expected_per_step"]
    curve = np.array([p["within_final_class_r2"]["delta"] for p in trajectory["points"]])
    return {
        "delta_trajectory": {"initial": float(curve[0]), "peak": float(curve.max()),
                             "peak_step": trajectory["points"][int(curve.argmax())]["step"],
                             "final": float(curve[-1]), "final_minus_peak": float(curve[-1] - curve.max())},
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
            "delta_trajectory": {k: stats([m["delta_trajectory"][k] for m in members])
                                 for k in ("initial", "peak", "final", "final_minus_peak")},
            "logit_margin_vs_q_margin_spearman": {
                k: stats([m["logit_margin_vs_q_margin_spearman"][k] for m in members
                          if m["logit_margin_vs_q_margin_spearman"][k] is not None] or [np.nan])
                for k in ("trained", "matched_init")},
        }
    return out


def order(arms):
    return sorted(arms, key=lambda k: position(arms[k]["lam"], arms[k]["kappa"]))


def predictors(arms, theory):
    """Arm-level rank correlation of candidate theory predictors with what training did to b1 - b2.

    Trivial Quotient is excluded (one action class, one seed, no future-action statistic).
    """
    names = [n for n in order(arms) if arms[n]["lam"] is not None]
    refs = [reference(theory, arms[n]["lam"], arms[n]["kappa"]) for n in names]
    candidates = {"delta_blind_reward_loss": [r["lumped_filter_loss"] for r in refs],
                  "next_optimal_action_share_needing_delta": [r["future_action"]["1"]["delta_increment"] for r in refs],
                  "q_gap_share_needing_delta": [r["qgap_share_needing_delta"] for r in refs]}
    outcomes = {"gain": [arms[n]["gain"]["full_h"]["delta"]["mean"] for n in names],
                "final_minus_peak": [arms[n]["delta_trajectory"]["final_minus_peak"]["mean"] for n in names]}
    return {"arms": names, "spearman": {c: {o: spearman(x, y) for o, y in outcomes.items()}
                                        for c, x in candidates.items()}}


def markdown(arms, theory):
    lines = ["| arm | seeds (reward-qualified) | reward / belief-MDP | reward lost without b1-b2 (MC SE) | classes | "
             "b1-b2 init | b1-b2 trained | gain b1-b2 | b1-b2 peak | final - peak | gain b3 | gain belief | "
             "head-blind gain b1-b2 | logits gain b1-b2 |",
             "|" + "---|" * 14]

    def cell(s):
        return f"{s['mean']:+.3f} [{s['min']:+.3f}, {s['max']:+.3f}] ({s['positive']}/{s['n']}+)"

    def level(s):
        return f"{s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}]"
    for name in order(arms):
        a = arms[name]
        ref = loss(reference(theory, a["lam"], a["kappa"]))
        g, t = a["gain"], a["delta_trajectory"]
        lines.append(
            f"| {name} | {len(a['seeds'])} ({len(a['reward_qualified_seeds'])}) | {a['reward_fraction']['mean']:.3f} | "
            f"{ref['absolute']:.4f} ({ref['se']:.4f}) = {ref['fraction']:.1%} | "
            f"{a['action_classes_used']['min']:.0f}-{a['action_classes_used']['max']:.0f} | "
            f"{level(a['matched_init']['full_h']['delta'])} | {level(a['trained']['full_h']['delta'])} | "
            f"{cell(g['full_h']['delta'])} | {level(t['peak'])} | {level(t['final_minus_peak'])} | "
            f"{cell(g['full_h']['rho'])} | {cell(g['full_h']['belief'])} | "
            f"{cell(g['head_blind']['delta'])} | {cell(g['logits']['delta'])} |")
    return "\n".join(lines) + "\n"


def figure(runs, arms, theory, trajectories):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42})
    names = [n for n in order(arms) if arms[n]["lam"] is not None]
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
    lost = [loss(reference(theory, arms[n]["lam"], arms[n]["kappa"])) for n in names]
    ax.bar(x, [100 * r["fraction"] for r in lost], color="0.5")
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
            ax.plot(np.maximum(steps, 10), values, color=color, lw=.7, alpha=.8,
                    label=(r"$\lambda$=1 (Reward-State)" if name == "lam=1" else r"$\lambda$=$-$1 (Full-State)") if first else None)
            first = False
    ax.set_xscale("log")
    ax.set_xlabel("training steps (millions; init at 10)")
    ax.set_ylabel(r"$b_1-b_2$ within-class $R^2$")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS / f"fig_within_class_symmetry_dose.{suffix}", bbox_inches="tight", pad_inches=.02, dpi=300)
    plt.close(fig)


def main():
    theory = json.loads((RESULTS / "theory.json").read_text())
    trajectories = sorted((json.loads(p.read_text()) for p in RAW.glob("*/trajectory.json")), key=lambda t: t["run_id"])
    by_run = {t["run_id"]: t for t in trajectories}
    runs = sorted((compact(a, theory, by_run[a["run_id"]])
                   for a in (json.loads(p.read_text()) for p in RAW.glob("*/analysis.json"))),
                  key=lambda r: (*position(r["lam"], r["kappa"]), r["seed"]))
    arms = aggregate(runs)
    (RESULTS / "dose_probe.json").write_text(json.dumps(
        {"reward_gate": REWARD_GATE, "protocol": "60,000 fit / 80,000 test rows, 64-step warmup, class-specific affine ridge 1e-6, "
         "class-centred held-out R2, classes and histories from the final greedy policy, matched init replayed on the same histories",
         "delta_blind_reward_loss": {n: loss(reference(theory, a["lam"], a["kappa"])) for n, a in arms.items()},
         "predictors_of_delta_change": predictors(arms, theory), "arms": arms, "runs": runs}, indent=1, allow_nan=False) + "\n")
    (RESULTS / "dose_trajectory.json").write_text(json.dumps(trajectories, indent=1, allow_nan=False) + "\n")
    table = markdown(arms, theory)
    (RESULTS / "dose_table.md").write_text(table)
    figure(runs, arms, theory, trajectories)
    print(table)


if __name__ == "__main__":
    main()
