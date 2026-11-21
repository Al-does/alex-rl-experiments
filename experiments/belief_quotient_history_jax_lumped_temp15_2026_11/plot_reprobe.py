"""Render the complete temperature-1.5 ladder target re-probe reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.belief_quotient_history_jax_lumped_2026_11.shared import (
    ARCHITECTURES, CONDITION_NAMES,
)
from experiments.belief_quotient_history_jax_lumped_temp15_2026_11.reprobe import TARGET_DEFINITIONS

LABELS = {"transformer": "Transformer", "memoryless": "Memoryless",
          "gru128": "GRU 128 × 2", "gru92": "GRU 92 × 4"}
TITLES = {"belief": "Full belief", "reward_state": "Reward-state probability ρ",
          "state_distinction": "Fine-state contrast δ", "coarse_reward_class": "Coarse reward-class probability c[B]"}
COLORS = dict(zip(ARCHITECTURES, plt.rcParams["axes.prop_cycle"].by_key()["color"]))


def load_reports(root):
    reports, hashes = {}, {}
    for variant in (1, 2, 3):
        for architecture in ARCHITECTURES:
            leaf = f"variant_{variant}_{architecture}"
            paths = list((root / leaf / "results").glob("*/reprobe_targets.json"))
            if len(paths) != 1:
                raise ValueError(f"expected one report for {leaf}, got {len(paths)}")
            path = paths[0]
            report = json.loads(path.read_text())
            rows = report["checkpoint_reports"]
            if report["smoke"] or len(rows) != 202 or len({r["checkpoint_step"] for r in rows}) != 202:
                raise ValueError(f"incomplete full-budget report: {leaf}")
            if rows[0]["checkpoint_step"] != 0:
                raise ValueError(f"missing initialization: {leaf}")
            expected = set(TARGET_DEFINITIONS) - ({"coarse_reward_class"} if variant == 3 else set())
            if any(set(r["targets"]) != expected for r in rows):
                raise ValueError(f"wrong targets: {leaf}")
            reports[variant, architecture] = report
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return reports, hashes


def main():
    root = Path(__file__).parent
    reports, hashes = load_reports(root)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for target, title in TITLES.items():
        for scale in ("log", "linear"):
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True, sharey=True)
            for variant, ax in enumerate(axes, start=1):
                ax.set_title(CONDITION_NAMES[variant])
                ax.set_yscale("log")
                ax.set_ylim(2e-4, .6)
                ax.set_xscale(scale)
                ax.grid(True, which="both", alpha=.2)
                ax.set_xlabel("Environment steps")
                if target == "coarse_reward_class" and variant == 3:
                    ax.text(.5, .5, "Not lumpable\n(target omitted)", ha="center", va="center", transform=ax.transAxes)
                    continue
                for architecture in ARCHITECTURES:
                    rows = reports[variant, architecture]["checkpoint_reports"]
                    steps = np.array([r["checkpoint_step"] for r in rows])
                    error = np.array([1 - r["targets"][target]["r_squared"] for r in rows])
                    selected = steps > 0 if scale == "log" else np.ones(len(steps), dtype=bool)
                    ax.plot(steps[selected], error[selected], color=COLORS[architecture],
                            label=LABELS[architecture], linewidth=1.3)
                if scale == "linear":
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(9, 9))
            axes[0].set_ylabel("Probe 1 − R²")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.02))
            fig.suptitle(title)
            fig.tight_layout(rect=(0, .07, 1, .95))
            stem = figures / f"reprobe_{target}_{scale}x"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.02)
            fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=.02, dpi=180)
            plt.close(fig)

    maximum = max(r["max_abs_belief_r_squared_difference"] for r in reports.values())
    lines = ["# Additional belief-target probes — temperature-1.5 Belief-Quotient Ladder", "",
             "Seed 42; all 12 arms × 202 checkpoints (2,424 evaluations), including actual initialization. "
             "Greedy held-out embedding probes use the original fixed key, 60,000 fit / 80,000 test rows, "
             "64-step per-episode warmup, and ridge 10⁻⁶. Reward is the held-out post-warmup reward per step, not probe accuracy.", "",
             f"Maximum absolute full-belief R² difference from the archive: **{maximum:.12g}**.", "",
             "Uniform nonreward destinations and symmetric emissions make A/B tokens carry identical reward-class likelihood: "
             "c[B] equals ρ for Trivial Quotient and Reward-State Quotient, up to numerical precision. "
             "Both targets are retained and fitted independently. Full-State Quotient is not lumpable, so its coarse target is omitted.", "",
             "The quotient filter processes complete token/action episodes before warmup removal, "
             "conditions on the reset observation without a transition, and then uses the preceding executed action. "
             "These descriptive linear-accessibility measurements do not establish causal use. Single seed; no confidence intervals.", "",
             "## Final checkpoint", "",
             "All entries below except reward are **1 − R²** (smaller is better); final step = 4,004,511,744.", "",
             "| Condition / architecture | Full belief | Reward state ρ | Fine contrast δ | Coarse c[B] | Reward/step |",
             "|---|---:|---:|---:|---:|---:|"]
    for (variant, architecture), report in reports.items():
        final = report["checkpoint_reports"][-1]
        values = [f"{1-final['targets'][t]['r_squared']:.6g}" if t in final["targets"] else "—" for t in TITLES]
        lines.append(f"| {CONDITION_NAMES[variant]} / {LABELS[architecture]} | " + " | ".join(values) + f" | {final['reward_per_step']:.6f} |")
    lines += ["", "## Figures", "",
              "One figure per target and x-axis convention; shared log y-axis 2×10⁻⁴–0.6. "
              "Log-x plots omit initialization because log(0) is undefined; linear-x plots include it. "
              "Values outside the requested y range are not clipped in the underlying JSON or final table.", ""]
    for target in TITLES:
        for scale in ("log", "linear"):
            lines.append(f"![{TITLES[target]}, {scale} x](figures/reprobe_{target}_{scale}x.png)")
    lines += ["", "## Reproduction and CPU check", "",
              "Run `uv run python -m experiments.belief_quotient_history_jax_lumped_temp15_2026_11.reprobe --leaf LEAF` for each leaf, "
              "then `uv run python -m experiments.belief_quotient_history_jax_lumped_temp15_2026_11.plot_reprobe`. "
              "Input SHA-256 hashes are in `figures/reprobe_plot_manifest.json`; per-checkpoint parameter hashes, source run commits, "
              "target definitions, budgets, and archive parity differences are in each `reprobe_targets.json`.", "",
              "Local machine: 8 CPUs. Full-budget initialization checks took 14.17 s (Transformer) and 6.83 s (GRU 92 × 4), "
              "including fetch/load/JIT/fits. CPU full-belief archive differences were 2.54×10⁻⁵ and 0.07414 respectively. "
              "The same GRU checkpoint on an existing 4090 matched within 1.3×10⁻¹²; production reprobes therefore used existing "
              "idle 4090 boxes with their original JAX environments. No boxes were rented or destroyed. "
              "The three-checkpoint memoryless CPU validation is retained separately and matched within 1.25×10⁻⁸.", ""]
    (root / "reprobe_findings.md").write_text("\n".join(lines))
    (figures / "reprobe_plot_manifest.json").write_text(json.dumps({"inputs_sha256": hashes,
        "max_abs_belief_r_squared_difference": maximum, "checkpoint_count": 2424,
        "y_limits": [2e-4, .6], "log_x_initialization": "omitted"}, indent=2) + "\n")
    print(f"Validated 12 reports / 2424 checkpoints; max_abs_belief_r_squared_difference={maximum:.12g}")
    print("\n".join(lines[lines.index("| Condition / architecture | Full belief | Reward state ρ | Fine contrast δ | Coarse c[B] | Reward/step |"):
                               lines.index("## Figures")]))


if __name__ == "__main__":
    main()
