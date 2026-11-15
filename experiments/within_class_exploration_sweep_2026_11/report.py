"""Tables and charts for the exploration-pressure within-class sweep."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze import OUTPUT

TRAJECTORY = OUTPUT.with_name("within_class_exploration_sweep_trajectory.json")
REWARD_GATE = OUTPUT.with_name("reward_gate.json")

RESULTS = OUTPUT.parent
NAMES = {2: "Reward-State", 3: "Full-State"}
COLORS = {2: "#c0504d", 3: "#2468a0"}
AXES = (("epsilon", "Training ε (entropy 0)"), ("entropy", "Constant entropy coef (ε = 0)"))


def axis_points(arms, axis):
    """Arms on one exploration axis; the shared (ε=0, entropy=0) baseline belongs to both."""
    if axis == "epsilon":
        group = [a for a in arms if a["training_entropy"] == 0]
        return sorted(group, key=lambda a: a["training_epsilon"]), "training_epsilon"
    group = [a for a in arms if a["training_epsilon"] == 0]
    return sorted(group, key=lambda a: a["training_entropy"]), "training_entropy"


def setting(a):
    if a["training_entropy"]:
        return f"entropy {a['training_entropy']:g}"
    return f"ε {a['training_epsilon']:g}"


def diff(a, feature, target="belief"):
    d = a["paired_differences"][feature][target]
    return d["difference"], *d["paired_difference_ci95"]


def r2(stage, feature, target="belief"):
    return stage["metrics"][feature][target]["r2"]


def fmt_ci(d):
    return f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"


def tables(arms):
    lines = ["| Quotient | Setting | Final reward/step | Final actions [Noop, X, Y] | Final h | Matched init | Δ full h [95% CI] | Δ head-blind [95% CI] | Δ logits [95% CI] | Final head-blind − logits |",
             "|---|---|---:|---|---:|---:|---|---|---|---:|"]
    for a in arms:
        f, m = a["stages"]["final"], a["matched_init"]
        policy = f["archive_check"]["archived_policy"]
        actions = "[" + ", ".join(f"{x:.2f}" for x in policy["action_fractions"]) + "]"
        lines.append(f"| {NAMES[a['variant']]} | {setting(a)} | {policy['reward_per_step']:.4f} | {actions} | {r2(f, 'full_h'):.4f} | "
                     f"{r2(m, 'full_h'):.4f} | {fmt_ci(diff(a, 'full_h'))} | {fmt_ci(diff(a, 'head_blind'))} | "
                     f"{fmt_ci(diff(a, 'logits'))} | {r2(f, 'head_blind')-r2(f, 'logits'):+.4f} |")
    lines += ["", "#### Scalar targets (full h)", "",
              "| Quotient | Setting | Final b1−b2 | Matched b1−b2 | Δ b1−b2 [95% CI] | Final b3 | Matched b3 | Δ b3 [95% CI] |",
              "|---|---|---:|---:|---|---:|---:|---|"]
    for a in arms:
        f, m = a["stages"]["final"], a["matched_init"]
        lines.append(f"| {NAMES[a['variant']]} | {setting(a)} | {r2(f, 'full_h', 'delta'):.4f} | {r2(m, 'full_h', 'delta'):.4f} | "
                     f"{fmt_ci(diff(a, 'full_h', 'delta'))} | {r2(f, 'full_h', 'rho'):.4f} | {r2(m, 'full_h', 'rho'):.4f} | "
                     f"{fmt_ci(diff(a, 'full_h', 'rho'))} |")
    lines += ["", "#### Controls (final encoder, full belief) and checks", "",
              "| Quotient | Setting | Shuffle | Random input | Label perm. | One-class final | Max archive |Δ| | Archive annotations |",
              "|---|---|---:|---:|---:|---|---:|---|"]
    for a in arms:
        f = a["stages"]["final"]
        c = [f["controls"][k]["belief"]["r2"] for k in ("history_shuffle", "random_input", "label_permutation")]
        checks = [s["archive_check"] for s in a["stages"].values()]
        notes = "; ".join(f"{k}: {s['archive_check']['annotation']}" for k, s in a["stages"].items() if s["archive_check"]["annotation"]) or "—"
        lines.append(f"| {NAMES[a['variant']]} | {setting(a)} | " + " | ".join(f"{v:+.4f}" for v in c)
                     + f" | {'Yes' if f['one_class'] else 'No'} | {max(abs(x['delta']) for x in checks):.1e} | {notes} |")
    return lines


def errorbar(ax, xs, values, color, label, offset):
    v = np.array(values)
    ax.errorbar(np.array(xs)+offset, v[:, 0], yerr=[v[:, 0]-v[:, 1], v[:, 2]-v[:, 0]], fmt="o-", capsize=3, color=color, label=label)


def figure(arms, rows, path, title):
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.4*len(rows)), sharey="row", squeeze=False)
    for col, (axis, xlabel) in enumerate(AXES):
        for variant in NAMES:
            group, key = axis_points([a for a in arms if a["variant"] == variant], axis)
            xs = [a[key] for a in group]
            span = (max(xs) or 1) * .012 * (1 if variant == 3 else -1)
            for row, (ylabel, fn) in enumerate(rows):
                errorbar(axes[row, col], xs, [fn(a) for a in group], COLORS[variant], NAMES[variant], span)
        for row, (ylabel, _) in enumerate(rows):
            ax = axes[row, col]
            ax.axhline(0, color="gray", lw=.8)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
        axes[-1, col].set_xlabel(xlabel)
        axes[0, col].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def absolute_figure(arms, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for col, (axis, xlabel) in enumerate(AXES):
        for row, variant in enumerate(NAMES):
            group, key = axis_points([a for a in arms if a["variant"] == variant], axis)
            xs = [a[key] for a in group]
            ax = axes[row, col]
            for label, fn, style in [("Final full h", lambda a: r2(a["stages"]["final"], "full_h"), "o-"),
                                     ("Final head-blind", lambda a: r2(a["stages"]["final"], "head_blind"), "s--"),
                                     ("Final logits only", lambda a: r2(a["stages"]["final"], "logits"), "^-"),
                                     ("Matched init full h", lambda a: r2(a["matched_init"], "full_h"), "x:")]:
                ax.plot(xs, [fn(a) for a in group], style, label=label)
            ax.set_title(f"{NAMES[variant]} Quotient", fontsize=10)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("Pooled within-class belief R²")
            if row == 1:
                ax.set_xlabel(xlabel)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Within-greedy-class belief R² by representation (seed 42, one model per point)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def knobs(a):
    return a["variant"], a["training_epsilon"], a["training_entropy"]


def load_gate():
    if not REWARD_GATE.exists():
        return {}, None
    gate = json.loads(REWARD_GATE.read_text())
    return {knobs(a): a for a in gate["arms"]}, gate["protocol"]


def reward_section(arms, gate, protocol, path):
    """Reward gate: final greedy reward vs the belief-grid reference, and Δ vs reward."""
    if not gate:
        return []
    reference = json.loads(REWARD_GATE.with_name("reward_reference.json").read_text())["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, variant in zip(axes, NAMES):
        group = [a for a in arms if a["variant"] == variant]
        for a in group:
            g = gate[knobs(a)]
            d = diff(a, "full_h")
            marker = "s" if a["training_entropy"] else "o"
            ax.errorbar(g["fraction_of_reference"], d[0], yerr=[[d[0]-d[1]], [d[2]-d[0]]],
                        xerr=[[g["fraction_of_reference"]-g["fraction_ci95"][0]], [g["fraction_ci95"][1]-g["fraction_of_reference"]]],
                        fmt=marker, color=COLORS[variant], capsize=2, mfc="white" if a["training_entropy"] else COLORS[variant])
            ax.annotate(setting(a), (g["fraction_of_reference"], d[0]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="gray", lw=.8)
        ax.axvline(float(protocol["gate"].split(">= ")[1].split("%")[0])/100, color="gray", ls="--", lw=.8)
        ax.set_title(f"{NAMES[variant]} Quotient", fontsize=10)
        ax.set_xlabel("Final greedy reward / belief-grid reference")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Δ full h within-class R²\n(final − matched init)")
    fig.suptitle("Training-added within-class resolution vs reward attained (circles: ε arms; squares: entropy arms;\n"
                 "dashed: reward gate; one seed per point)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    refs = {v: reference[str(v)] for v in NAMES}
    lines = ["#### Reward attained (gate for the belief-resolution comparison)", "",
             "Reference: belief-MDP value iteration on a simplex grid (`reward_reference.py`), its lookahead policy simulated "
             "with the exact filter for 4,096 episodes under the probe protocol (1,024 steps, 64-step warmup excluded). It is "
             "near-optimal (grid/discount sweep converged), not a certified Bayes bound. "
             + " ".join(f"{NAMES[v]}: reference {r['best_belief_grid']['mean']:.4f}, myopic-belief {r['myopic_belief']['mean']:.4f}, "
                        f"best current-token {r['current_token']['mean']:.4f}, Noop {r['constant_action']['0']['mean']:.4f}."
                        for v, r in refs.items()),
             f"Policies are greedy, {protocol['episodes']:,} fresh episodes each; gate: {protocol['gate']}. "
             "Arm − best arm uses common random numbers (shared rollout key) and a paired episode bootstrap. "
             "First gated step / gated peak use the sampled checkpoints: " + protocol["checkpoint_rewards"] + ".", "",
             "| Quotient | Setting | Final reward/step [95% CI] | / reference [95% CI] | Gate | Arm − best arm [95% CI] | First gated checkpoint | Gated peak (fixed hist.) | Gated peak − final |",
             "|---|---|---|---|---|---|---:|---|---:|"]
    for a in arms:
        g = gate[knobs(a)]
        b, p = g["minus_best_arm"], g["gated_peak"]
        first = "—" if g["first_gated_step"] is None else f"{g['first_gated_step']/1e6:,.0f}M"
        peak = "—" if p is None else f"{p['fixed_history_full_h_r2']:.4f} @ {p['step']/1e6:,.0f}M"
        lines.append(f"| {NAMES[a['variant']]} | {setting(a)} | {g['reward_mean']:.4f} [{g['reward_ci95'][0]:.4f}, {g['reward_ci95'][1]:.4f}] | "
                     f"{g['fraction_of_reference']:.3f} [{g['fraction_ci95'][0]:.3f}, {g['fraction_ci95'][1]:.3f}] | "
                     f"{'pass' if g['passes_gate'] else 'FAIL'} | {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
                     f"{first} | {peak} | {'—' if p is None else format(p['minus_final'], '+.4f')} |")
    return [*lines, "", f"![reward]({path.name})", ""]


def level(arm, axis):
    return arm["training_epsilon"] if axis == "epsilon" else arm["training_entropy"]


def trajectory_section(path, gate=None):
    """Fixed-final-history checkpoint trajectories (parent PR219 protocol), when that JSON exists."""
    gate = gate or {}
    if not TRAJECTORY.exists():
        return []
    runs = sorted(json.loads(TRAJECTORY.read_text())["arms"],
                  key=lambda r: (r["variant"], r["training_entropy"] > 0, r["training_epsilon"], r["training_entropy"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey="row")
    for col, (axis, xlabel) in enumerate(AXES):
        for row, variant in enumerate(NAMES):
            group, _ = axis_points([r for r in runs if r["variant"] == variant], axis)
            cmap = plt.get_cmap("viridis")
            ax = axes[row, col]
            for i, r in enumerate(group):
                x = np.array([p["step"] for p in r["points"]])/1e9
                m = [p["fixed_history"]["metrics"]["full_h"]["belief"] for p in r["points"]]
                ci = np.array([d["bootstrap"]["r2_ci95"] for d in m])
                color = cmap(i/max(1, len(group)-1)*.9)
                ax.plot(x, [d["r2"] for d in m], "-", marker=".", ms=4, color=color, label=f"{xlabel.split(' (')[0]} = {level(r, axis):g}")
                ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=.1)
                if knobs(r) in gate:
                    fails = [not c["passes_gate"] for c in gate[knobs(r)]["checkpoints"]]
                    ax.plot(x[fails], np.array([d["r2"] for d in m])[fails], "x", color=color, ms=6)
            ax.set_title(f"{NAMES[variant]} Quotient", fontsize=10)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel("Within-class belief R²\n(encoder at step, final-policy histories)", fontsize=9)
            if row == 1:
                ax.set_xlabel(f"Training steps (billions) — {xlabel}")
    fig.suptitle("Checkpoint trajectories on fixed final-policy histories (step 0 = matched init)\n"
                 "Pointwise 95% episode-bootstrap bands; seed 42 only; × = checkpoint policy fails the reward gate", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    lines = ["#### Checkpoint trajectories (fixed final-policy histories)", "",
             "Each run's 12 prespecified checkpoints (parent PR219 trajectory protocol) are replayed on that run's final-policy "
             "histories, classes and targets with the probe refit per encoder. Peaks are chosen after looking, so the paired "
             "peak − final interval is exploratory and not selection-adjusted. Peaks here ignore reward; "
             "see the reward-gate table for peaks restricted to checkpoints whose policy passes the gate.", "",
             "| Quotient | Setting | Init (fixed hist.) | Peak | Peak step | Final | Peak − final [95% CI] |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in runs:
        values = {p["step"]: p["fixed_history"]["metrics"]["full_h"]["belief"]["r2"] for p in r["points"]}
        d = r["fixed_peak_minus_final"]
        lines.append(f"| {NAMES[r['variant']]} | {setting(r)} | {values[min(values)]:.4f} | {values[d['peak_step']]:.4f} | "
                     f"{d['peak_step']/1e6:,.0f}M | {values[r['final_reference_step']]:.4f} | "
                     f"{fmt_ci((d['difference'], *d['paired_difference_ci95']))} |")
    return [*lines, "", f"![trajectory]({path.name})", ""]


def main():
    report = json.loads(OUTPUT.read_text())
    arms = sorted(report["arms"], key=lambda a: (a["variant"], a["training_entropy"] > 0, a["training_epsilon"], a["training_entropy"]))
    figure(arms, [("Δ full h R²\n(final − matched init)", lambda a: diff(a, "full_h")),
                  ("Δ head-blind R²\n(final − matched init)", lambda a: diff(a, "head_blind")),
                  ("Δ logits-only R²\n(final − matched init)", lambda a: diff(a, "logits"))],
           RESULTS/"gain_vs_exploration.png", "Training-added within-class belief resolution vs exploration pressure\n(95% paired episode-bootstrap CI; one training seed per point)")
    figure(arms, [("Δ b1−b2 R² (full h)", lambda a: diff(a, "full_h", "delta")),
                  ("Δ b3 R² (full h)", lambda a: diff(a, "full_h", "rho"))],
           RESULTS/"scalar_gain_vs_exploration.png", "Training-added within-class resolution of scalar belief coordinates\n(b1−b2: non-reward-state contrast; b3: reward-state probability)")
    absolute_figure(arms, RESULTS/"within_class_r2_by_representation.png")
    gate, protocol = load_gate()
    lines = ["# Results: exploration pressure vs within-class belief resolution", "",
             "Generated by `report.py` from `within_class_exploration_sweep.json`. Δ is final minus matched-init "
             "(init encoder replayed on the final policy's exact histories); 95% intervals are 10,000 paired "
             "whole-episode resamples conditional on one training seed (42).", "",
             *reward_section(arms, gate, protocol, RESULTS/"reward_vs_gain.png"), "#### Within-class probes", "", *tables(arms), "",
             "![gain](gain_vs_exploration.png)", "", "![scalar](scalar_gain_vs_exploration.png)", "",
             "![absolute](within_class_r2_by_representation.png)", "",
             *trajectory_section(RESULTS/"trajectory_vs_exploration.png", gate)]
    (RESULTS/"README.md").write_text("\n".join(lines))
    print("Reported", len(arms), "arms")


if __name__ == "__main__":
    main()
