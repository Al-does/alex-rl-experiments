"""Architecture comparison on the eps=0.02 Belief-Quotient Ladder and Reward Relevance tasks.

Reads committed condition summaries (ladder non-transformer results come from PR #238) and
writes manuscript-width PDFs (plus PNG previews) to ./results following the deep-beliefs
figure rules: 5.5 in width, 8/7 pt text, TrueType fonts.

    uv run python -m experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11.architecture_comparison.plot
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
LADDER = "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11"
RELEVANCE = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11"
WIDTH = 5.5
EXPERIMENTS = {
    "Belief-Quotient Ladder": [
        ("Trivial Quotient", LADDER, "variant_1", "belief", 0.966),
        ("Reward-State Quotient", LADDER, "variant_2", "belief", 0.468),
        ("Full-State Quotient", LADDER, "variant_3", "belief", 0.429),
    ],
    "Reward Relevance": [
        ("Joint Reward", RELEVANCE, "reward_both", "joint", None),
        ("Unrewarded Factor 2", RELEVANCE, "reward_factor_1", "joint", None),
    ],
}
ARCHS = [("transformer", "Transformer", "#4C78A8"), ("gru128", "GRU-128", "#F58518"),
         ("gru92", "GRU-92", "#54A24B"), ("memoryless", "Memoryless MLP", "#9D9D9D")]
SEED = 42
ERR_LIM = (1e-2, 1.5)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def load(study, leaf, target):
    runs = {}
    for path in glob.glob(str(EXP / study / leaf / "results" / "*" / "condition_summary.json")):
        summary = json.load(open(path))
        if summary.get("smoke"):
            continue
        rows = sorted(summary["checkpoint_reports"], key=lambda r: r["checkpoint_step"])
        runs[summary["seed"]] = (
            np.array([r["checkpoint_step"] / 1e9 for r in rows]),
            np.array([r["policy"]["reward_per_step"] for r in rows]),
            np.array([1 - r["targets"][target]["r_squared"] for r in rows]),
        )
    assert SEED in runs, leaf
    assert all(len(x) == 102 and abs(x[-1] - 2.012) < 0.01 for x, _, _ in runs.values()), leaf
    return runs


DATA = {(cond, arch): load(study, f"{cond}_{arch}", target)
        for conds in EXPERIMENTS.values() for _, study, cond, target, _ in conds
        for arch, _, _ in ARCHS}


def style(ax):
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def legend(fig, handles, labels):
    rank = [name for _, name, _ in ARCHS] + ["Belief-filter"]
    order = sorted(range(len(labels)), key=lambda i: (
        next(k for k, name in enumerate(rank) if labels[i].startswith(name)), "single" not in labels[i]))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside lower center",
               ncol=3, frameon=False, columnspacing=1.0, handlelength=1.6, handletextpad=0.5)


def curves(experiment, filename):
    conds = EXPERIMENTS[experiment]
    fig, axes = plt.subplots(2, len(conds), figsize=(WIDTH - 0.06, 3.6), sharex=True,
                             constrained_layout=True)
    for j, (name, _, cond, target, reference) in enumerate(conds):
        top, bottom = axes[0, j], axes[1, j]
        for arch, label, color in ARCHS:
            runs = DATA[cond, arch]
            x, reward, err = runs[SEED]
            if len(runs) > 1:
                rewards = np.array([r[1] for r in runs.values()])
                errs = np.array([r[2] for r in runs.values()])
                top.fill_between(x, rewards.min(0), rewards.max(0), color=color, alpha=0.18,
                                 linewidth=0, label=f"{label} ({len(runs)}-seed range)")
                bottom.fill_between(x, errs.min(0), errs.max(0), color=color, alpha=0.18, linewidth=0)
            top.plot(x, reward, color=color, linewidth=1.0, label=f"{label} (single seed)")
            bottom.plot(x, np.clip(err, ERR_LIM[0], None), color=color, linewidth=1.0)
        if reference:
            top.axhline(reference, color="black", linestyle="--", linewidth=0.7,
                        label="Belief-filter reference policy")
        top.set_title(f"{experiment}:\n{name}")
        top.set_ylim(0, 1)
        bottom.set_yscale("log")
        bottom.set_ylim(*ERR_LIM)
        bottom.set_xlim(0, 2.02)
        bottom.set_xlabel(r"Environment steps ($\times10^9$)")
        for ax in (top, bottom):
            style(ax)
    axes[0, 0].set_ylabel("Greedy reward\nper step")
    target_name = "joint belief" if experiment == "Reward Relevance" else "belief"
    axes[1, 0].set_ylabel(f"$1-R^2$ probe error\n({target_name})")
    fig.align_ylabels(axes[:, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend(fig, handles, labels)
    fig.savefig(OUT / filename, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / filename.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def final_bars(filename):
    groups = [(exp, name, cond, ref) for exp, conds in EXPERIMENTS.items()
              for name, _, cond, _, ref in conds]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(WIDTH, 3.9), sharex=True,
                                      constrained_layout=True)
    xs = np.arange(len(groups))
    width = 0.2
    for k, (arch, label, color) in enumerate(ARCHS):
        offset = (k - 1.5) * width
        rewards = [DATA[c, arch][SEED][1][-1] for _, _, c, _ in groups]
        errs = [DATA[c, arch][SEED][2][-1] for _, _, c, _ in groups]
        top.bar(xs + offset, rewards, width, color=color, label=f"{label} (single seed)")
        bottom.bar(xs + offset, errs, width, color=color)
        if arch == "transformer":
            for i, (_, _, c, _) in enumerate(groups):
                runs = DATA[c, arch].values()
                fr = np.array([r[1][-1] for r in runs])
                fe = np.array([r[2][-1] for r in runs])
                top.errorbar(xs[i] + offset, np.median(fr), yerr=[[np.median(fr) - fr.min()], [fr.max() - np.median(fr)]],
                             fmt="o", color="black", markersize=2.5, capsize=2, elinewidth=0.7,
                             label=f"Transformer ({len(fr)}-seed median, range)" if i == 0 else None)
                bottom.errorbar(xs[i] + offset, np.median(fe),
                                yerr=[[np.median(fe) - fe.min()], [fe.max() - np.median(fe)]],
                                fmt="o", color="black", markersize=2.5, capsize=2, elinewidth=0.7)
    for i, (_, _, _, ref) in enumerate(groups):
        if ref:
            top.hlines(ref, xs[i] - 2 * width, xs[i] + 2 * width, color="black", linestyle="--",
                       linewidth=0.7, label="Belief-filter reference policy" if i == 0 else None)
    top.set_ylim(0, 1.05)
    top.set_ylabel("Final greedy reward\nper step")
    bottom.set_yscale("log")
    bottom.set_ylim(*ERR_LIM)
    bottom.set_ylabel("Final $1-R^2$\nprobe error")
    fig.align_ylabels([top, bottom])
    bottom.set_xticks(xs, [name.replace(" ", "\n", 1) for _, name, _, _ in groups])
    split = len(EXPERIMENTS["Belief-Quotient Ladder"]) - 0.5
    for ax in (top, bottom):
        style(ax)
        ax.axvline(split, color="#bbbbbb", linewidth=0.6)
    for exp, conds in EXPERIMENTS.items():
        idx = [i for i, g in enumerate(groups) if g[0] == exp]
        top.text(np.mean(idx), 1.07, exp, ha="center", va="bottom", fontsize=8,
                 transform=top.get_xaxis_transform())
    handles, labels = top.get_legend_handles_labels()
    legend(fig, handles, labels)
    fig.savefig(OUT / filename, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / filename.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    curves("Belief-Quotient Ladder", "architecture_comparison_ladder_curves.pdf")
    curves("Reward Relevance", "architecture_comparison_reward_relevance_curves.pdf")
    final_bars("architecture_comparison_final.pdf")
