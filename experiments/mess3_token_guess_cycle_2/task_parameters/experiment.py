"""Choose the MESS3 operating point for cycle 2."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_token_guess_cycle_2.operating_point import (
    OperatingPoint,
    accuracy_bounds,
    belief_r2_floor,
    evaluate,
    simulate_parallel,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext

FIT_STEPS = 60_000
TEST_STEPS = 30_000
SMOKE_FIT_STEPS = 8_000
SMOKE_TEST_STEPS = 4_000

GRID_ALPHAS = (0.5, 0.6, 0.7, 0.85, 0.95)
GRID_TRANSITIONS = (0.9, 0.97, 0.99, 0.995)
SMOKE_ALPHAS = (0.7, 0.85)
SMOKE_TRANSITIONS = (0.9, 0.995)

CANDIDATES = {
    "cycle_1": OperatingPoint(alpha=0.85, self_transition=0.90),
    "candidate_a": OperatingPoint(alpha=0.70, self_transition=0.99),
    "candidate_b": OperatingPoint(alpha=0.60, self_transition=0.995),
    "candidate_c": OperatingPoint(alpha=0.70, self_transition=0.995),
}

# Measured by training the study architecture (3 layers, d_model 96, context 64)
# on next-token prediction to the Bayes cross-entropy, then probing it. Recorded
# rather than recomputed because it takes about twenty minutes per point on CPU.
SUPERVISED_PROBE_R2 = {
    "cycle_1": {1_500: 0.9567, 3_000: 0.9532, 4_500: 0.9434, 6_000: 0.9318},
    "candidate_c": {1_500: 0.9487, 3_000: 0.9478, 4_500: 0.9439, 6_000: 0.9367},
}


def _plot_grid(grid: dict[str, Any], *, path) -> None:
    alphas = sorted({entry["alpha"] for entry in grid.values()})
    transitions = sorted({entry["self_transition"] for entry in grid.values()})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for axis, key, title in (
        (axes[0], "belief_r2_range", "belief-probe R² range (1 − floor)"),
        (axes[1], "accuracy_range", "greedy accuracy range (Bayes − repeat-last)"),
    ):
        for transition in transitions:
            values = [
                grid[f"{alpha}_{transition}"][key]
                for alpha in alphas
                if f"{alpha}_{transition}" in grid
            ]
            axis.plot(alphas[: len(values)], values, "o-", label=f"p = {transition}")
        axis.set_xlabel("emission concentration α")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("usable range of the metric")
    figure.suptitle(
        "Both metrics are near-degenerate at the parameters cycle 1 used "
        "(α = 0.85, p = 0.9)",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _findings(result: dict[str, Any]) -> str:
    lines = [
        "# Choosing the MESS3 operating point",
        "",
        "## Candidates",
        "",
        "`range` is what the belief-probe metric can move through; `ESS` is how "
        "many independent samples a 30,000-step probe rollout actually contains, "
        "given how slowly the chain mixes.",
        "",
        "| point | α | p | τ | R² floor | R² range | acc range | ESS | probe ±95% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in result["candidates"].items():
        lines.append(
            f"| `{name}` | {entry['alpha']:.2f} | {entry['self_transition']:.3f} | "
            f"{entry['state_correlation_time']:.0f} | "
            f"{entry['belief_r2_floor']:.4f} | {entry['belief_r2_range']:.4f} | "
            f"{entry['accuracy_range']:.4f} | "
            f"{entry['effective_sample_size']:.0f} | "
            f"{entry['probe_ci_half_width']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Context length",
            "",
            "Best belief R² available to a model that can see only the last k "
            "observations. No objective can beat this.",
            "",
            "| point | " + " | ".join(
                f"k={k}"
                for k in sorted(
                    next(iter(result["candidates"].values()))["context_requirement"]
                )
            ) + " |",
            "|---" * (1 + len(next(iter(result["candidates"].values()))["context_requirement"])) + "|",
        ]
    )
    for name, entry in result["candidates"].items():
        row = " | ".join(
            f"{value:.4f}" for _, value in sorted(entry["context_requirement"].items())
        )
        lines.append(f"| `{name}` | {row} |")
    lines.extend(
        [
            "",
            "The study's context length of 64 is sufficient at every candidate, so "
            "it does not need sweeping. The belief converges much faster than the "
            "hidden state does, because each observation is informative enough to "
            "wash out the prior well before the state decorrelates.",
            "",
            "## Supervised ceiling",
            "",
            "Training the study architecture on next-token prediction to the Bayes "
            "cross-entropy, then probing it:",
            "",
            "| point | 1.5k steps | 3k | 4.5k | 6k |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, curve in SUPERVISED_PROBE_R2.items():
        row = " | ".join(f"{curve[step]:.4f}" for step in sorted(curve))
        lines.append(f"| `{name}` | {row} |")
    lines.extend(
        [
            "",
            "Belief-probe R² falls with continued training at both points while "
            "cross-entropy stays at the Bayes floor. This is supervised training, "
            "so the decline cycle 1 saw over 20M PPO steps is not caused by "
            "reinforcement learning. It is optimiser-driven drift in a "
            "representation the task no longer constrains, which makes learning "
            "rate, optimiser, and training duration larger influences on the "
            "headline metric than most of the differences between arms.",
            "",
        ]
    )
    return "\n".join(lines)


def run(context: RunContext) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("choosing an operating point requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    fit_steps = SMOKE_FIT_STEPS if context.smoke else FIT_STEPS
    test_steps = SMOKE_TEST_STEPS if context.smoke else TEST_STEPS
    alphas = SMOKE_ALPHAS if context.smoke else GRID_ALPHAS
    transitions = SMOKE_TRANSITIONS if context.smoke else GRID_TRANSITIONS

    grid: dict[str, Any] = {}
    for alpha in alphas:
        for transition in transitions:
            point = OperatingPoint(alpha=alpha, self_transition=transition)
            fit = simulate_parallel(point, n_steps=fit_steps, seed=context.seed)
            test = simulate_parallel(point, n_steps=test_steps, seed=context.seed + 1)
            floor = belief_r2_floor(fit, test)
            accuracy_floor, accuracy_ceiling = accuracy_bounds(point, test)
            grid[f"{alpha}_{transition}"] = {
                "alpha": alpha,
                "self_transition": transition,
                "belief_r2_floor": floor,
                "belief_r2_range": 1.0 - floor,
                "accuracy_floor": accuracy_floor,
                "accuracy_ceiling": accuracy_ceiling,
                "accuracy_range": accuracy_ceiling - accuracy_floor,
            }

    candidates = {
        name: evaluate(
            point,
            fit_steps=fit_steps,
            test_steps=test_steps,
            seed=context.seed,
        )
        for name, point in CANDIDATES.items()
    }
    figure_path = context.results_dir / "operating_point_grid.png"
    _plot_grid(grid, path=figure_path)
    result = {
        "seed": context.seed,
        "smoke": context.smoke,
        "grid": grid,
        "candidates": candidates,
        "supervised_probe_r2": {
            name: {str(step): value for step, value in curve.items()}
            for name, curve in SUPERVISED_PROBE_R2.items()
        },
        "figure": str(figure_path),
    }
    outputs.write_json("task_parameters.json", result)
    (context.results_dir / "findings.md").write_text(_findings(result))
    return result
