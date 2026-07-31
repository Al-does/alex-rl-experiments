"""Network-free sweep over the guess-feedback strength.

This arm answers the two structural questions before any policy is trained:
how hard the loop is at each strength, and whether the closed loop collapses
to one stacked, renormalized HMM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_feedback_cycle_1.composition import (
    myopic_ceiling,
    single_hmm_report,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext


# 0.5 is excluded: with exact argmax tie-breaking it is a measure-zero
# degeneracy where the loop drives itself into a near-uniform belief.
STRENGTH_GRID = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
SMOKE_STRENGTH_GRID = (0.0, 0.35, 0.7, 1.0)
THEORY_POLICIES = ("myopic_argmax", "probability_matching")
CONTEXT_LENGTH = 10


def sweep(
    *,
    strengths: tuple[float, ...],
    policies: tuple[str, ...],
    n_chains: int,
    n_steps: int,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Report the ceiling, single-HMM residual, and factorization loss."""

    rows: list[dict[str, Any]] = []
    for strength in strengths:
        row: dict[str, Any] = {
            "feedback_strength": float(strength),
            "ceiling_exact_filter": myopic_ceiling(
                strength,
                n_chains=n_chains,
                n_steps=n_steps,
                seed=seed,
            ),
            "ceiling_context_10": myopic_ceiling(
                strength,
                context_length=CONTEXT_LENGTH,
                n_chains=n_chains,
                n_steps=n_steps,
                seed=seed,
            ),
            "policies": {},
        }
        for policy in policies:
            row["policies"][policy] = single_hmm_report(
                strength,
                policy=policy,
                n_chains=n_chains,
                n_steps=n_steps,
                seed=seed,
            )
        rows.append(row)
    return rows


def plot_sweep(rows: list[dict[str, Any]], *, path: Path) -> None:
    """Draw ceiling, single-HMM identifiability, and factorization loss."""

    strengths = np.asarray([row["feedback_strength"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    axes[0].plot(
        strengths,
        [100.0 * row["ceiling_exact_filter"]["accuracy"] for row in rows],
        marker="o",
        color="#355c9a",
        label="Exact filter",
    )
    axes[0].plot(
        strengths,
        [100.0 * row["ceiling_context_10"]["accuracy"] for row in rows],
        marker="s",
        color="#c45135",
        label="10-step context",
    )
    axes[0].axhline(100.0 / 3.0, color="#888888", linestyle=":", label="Chance")
    axes[0].set_xlabel("Feedback strength kappa")
    axes[0].set_ylabel("Myopic Bayes accuracy (%)")
    axes[0].set_title("Task difficulty is an inverted U in kappa")

    colors = {"myopic_argmax": "#355c9a", "probability_matching": "#5aa17f"}
    for policy, color in colors.items():
        if policy not in rows[0]["policies"]:
            continue
        axes[1].plot(
            strengths,
            [row["policies"][policy]["block_tv_marginal_hmm"] for row in rows],
            marker="o",
            color=color,
            label=f"{policy}: stacked HMM",
        )
        axes[1].plot(
            strengths,
            [row["policies"][policy]["block_tv_sampling_floor"] for row in rows],
            marker=".",
            linestyle="--",
            color=color,
            alpha=0.6,
            label=f"{policy}: sampling floor",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Feedback strength kappa")
    axes[1].set_ylabel("Total variation over 4-token blocks")
    axes[1].set_title("Can one stacked HMM reproduce the loop?")

    reference = THEORY_POLICIES[0]
    losses = [row["policies"][reference]["executed_product_mse"] for row in rows]
    axes[2].plot(
        strengths,
        losses,
        marker="o",
        color="#355c9a",
        label="Product-state error on executed belief",
    )
    register = axes[2].twinx()
    register.plot(
        strengths,
        [row["policies"][reference]["register_entropy_nats"] for row in rows],
        marker="s",
        color="#c45135",
        label="Register entropy (nats)",
    )
    register.axhline(
        np.log(3.0),
        color="#888888",
        linestyle=":",
        label="Maximum register entropy",
    )
    axes[2].set_xlabel("Feedback strength kappa")
    axes[2].set_ylabel("Factorization loss", color="#355c9a")
    register.set_ylabel("Register entropy (nats)", color="#c45135")
    axes[2].set_title("Both endpoints factor; the interior does not")
    axes[2].set_ylim(-0.01, max(losses) * 1.9)
    register.set_ylim(-0.06, np.log(3.0) * 1.9)
    handles, labels = axes[2].get_legend_handles_labels()
    extra_handles, extra_labels = register.get_legend_handles_labels()
    axes[2].legend(
        handles + extra_handles,
        labels + extra_labels,
        fontsize=7,
        loc="upper center",
    )

    for axis in axes[:2]:
        axis.legend(fontsize=7, loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run_theory(context: RunContext) -> dict[str, Any]:
    """Run and record the analytic feedback sweep."""

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    strengths = SMOKE_STRENGTH_GRID if context.smoke else STRENGTH_GRID
    rows = sweep(
        strengths=strengths,
        policies=THEORY_POLICIES,
        n_chains=64 if context.smoke else 384,
        n_steps=384 if context.smoke else 3_072,
        seed=17 if context.seed is None else int(context.seed),
    )
    figure_path = context.results_dir / "feedback_theory_sweep.png"
    plot_sweep(rows, path=figure_path)
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "strengths": list(strengths),
        "policies": list(THEORY_POLICIES),
        "rows": rows,
        "figures": {"sweep": str(figure_path)},
        "reading": (
            "A stacked single HMM is adequate only where its 4-token total "
            "variation sits at the sampling floor."
        ),
    }
    outputs.write_json("theory_sweep.json", summary)
    return summary
