"""Network-free sweep over both axes of the composed generator.

This arm settles the structural questions before any policy is trained: how
hard each generator is, what a factored representation costs there, and whether
the closed loop collapses to one stacked, renormalized HMM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_feedback_factoring_cycle_1.composition import (
    myopic_ceiling,
    single_hmm_report,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext


OPERATING_STRENGTH = 0.7
NOISE_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.85, 0.95, 1.0)
# 0.5 is excluded from the strength grid: with exact argmax tie-breaking it is
# a measure-zero degeneracy where the loop drives itself into a near-uniform
# belief and never plays the register-inert guess.
STRENGTH_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
SMOKE_NOISE_GRID = (0.0, 0.5, 1.0)
SMOKE_STRENGTH_GRID = (0.0, 0.7, 1.0)
THEORY_POLICIES = ("myopic_argmax", "probability_matching")
CONTEXT_LENGTH = 10


def sweep(
    pairs: tuple[tuple[float, float], ...],
    *,
    policies: tuple[str, ...],
    n_chains: int,
    n_steps: int,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Report the ceiling, factoring cost, and single-HMM residual per point."""

    rows: list[dict[str, Any]] = []
    for strength, noise in pairs:
        row: dict[str, Any] = {
            "feedback_strength": float(strength),
            "register_noise": float(noise),
            "ceiling_context_10": myopic_ceiling(
                strength,
                noise,
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
                noise,
                policy=policy,
                n_chains=n_chains,
                n_steps=n_steps,
                seed=seed,
            )
        rows.append(row)
    return rows


def _series(rows: list[dict[str, Any]], policy: str, key: str) -> np.ndarray:
    return np.asarray([row["policies"][policy][key] for row in rows])


def plot_sweep(
    noise_rows: list[dict[str, Any]],
    strength_rows: list[dict[str, Any]],
    *,
    path: Path,
) -> None:
    """Draw the factoring-cost axis beside the feedback-strength axis."""

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    noise = np.asarray([row["register_noise"] for row in noise_rows])
    reference = THEORY_POLICIES[0]

    cost = _series(noise_rows, reference, "factored_cost_nats")
    axes[0].plot(noise, cost, marker="o", color="#355c9a", label="Cost of factoring")
    axes[0].set_ylabel("Extra nats per token from a product state", color="#355c9a")
    entropy = axes[0].twinx()
    entropy.plot(
        noise,
        _series(noise_rows, reference, "register_entropy_nats"),
        marker="s",
        color="#c45135",
        label="Register entropy",
    )
    entropy.axhline(np.log(3.0), color="#888888", linestyle=":", label="log 3")
    entropy.set_ylabel("Register entropy (nats)", color="#c45135")
    axes[0].set_xlabel("Register noise epsilon")
    axes[0].set_title(f"Factoring cost ramps with epsilon (kappa={OPERATING_STRENGTH})")
    axes[0].set_ylim(-0.01, max(cost.max(), 1e-6) * 1.75)
    entropy.set_ylim(-0.06, np.log(3.0) * 1.75)
    handles, labels = axes[0].get_legend_handles_labels()
    extra = entropy.get_legend_handles_labels()
    axes[0].legend(handles + extra[0], labels + extra[1], fontsize=7, loc="upper center")

    axes[1].plot(
        noise,
        [100.0 * row["ceiling_context_10"]["accuracy"] for row in noise_rows],
        marker="o",
        color="#355c9a",
        label="epsilon axis (kappa=0.7)",
    )
    strength = np.asarray([row["feedback_strength"] for row in strength_rows])
    axes[1].plot(
        strength,
        [100.0 * row["ceiling_context_10"]["accuracy"] for row in strength_rows],
        marker="s",
        color="#c45135",
        label="kappa axis (epsilon=1)",
    )
    axes[1].axhline(100.0 / 3.0, color="#888888", linestyle=":", label="Chance")
    axes[1].set_xlabel("epsilon (blue) or kappa (red)")
    axes[1].set_ylabel("Myopic Bayes accuracy (%)")
    axes[1].set_title("epsilon barely moves difficulty; kappa does")
    axes[1].legend(fontsize=7, loc="best")

    colors = {"myopic_argmax": "#355c9a", "probability_matching": "#5aa17f"}
    for policy, color in colors.items():
        if policy not in strength_rows[0]["policies"]:
            continue
        axes[2].plot(
            strength,
            _series(strength_rows, policy, "block_tv_marginal_hmm"),
            marker="o",
            color=color,
            label=f"{policy}: stacked HMM",
        )
        axes[2].plot(
            strength,
            _series(strength_rows, policy, "block_tv_sampling_floor"),
            marker=".",
            linestyle="--",
            alpha=0.6,
            color=color,
            label=f"{policy}: sampling floor",
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Feedback strength kappa")
    axes[2].set_ylabel("Total variation over 4-token blocks")
    axes[2].set_title("Can one stacked HMM reproduce the loop?")
    axes[2].legend(fontsize=7, loc="best")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run_theory(context: RunContext) -> dict[str, Any]:
    """Run and record the analytic sweep over both generator axes."""

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    noise_grid = SMOKE_NOISE_GRID if context.smoke else NOISE_GRID
    strength_grid = SMOKE_STRENGTH_GRID if context.smoke else STRENGTH_GRID
    scale = {
        "n_chains": 64 if context.smoke else 384,
        "n_steps": 384 if context.smoke else 3_072,
        "seed": 17 if context.seed is None else int(context.seed),
        "policies": THEORY_POLICIES,
    }
    noise_rows = sweep(
        tuple((OPERATING_STRENGTH, noise) for noise in noise_grid),
        **scale,
    )
    strength_rows = sweep(
        tuple((strength, 1.0) for strength in strength_grid),
        **scale,
    )
    figure_path = context.results_dir / "feedback_theory_sweep.png"
    plot_sweep(noise_rows, strength_rows, path=figure_path)
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "operating_strength": OPERATING_STRENGTH,
        "register_noise_grid": list(noise_grid),
        "feedback_strength_grid": list(strength_grid),
        "policies": list(THEORY_POLICIES),
        "register_noise_rows": noise_rows,
        "feedback_strength_rows": strength_rows,
        "figures": {"sweep": str(figure_path)},
        "reading": (
            "A stacked single HMM is adequate only where its 4-token total "
            "variation sits at the sampling floor; a factored representation "
            "is adequate only where the cost of factoring is near zero."
        ),
    }
    outputs.write_json("theory_sweep.json", summary)
    return summary
