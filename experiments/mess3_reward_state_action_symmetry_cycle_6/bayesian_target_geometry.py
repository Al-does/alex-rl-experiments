"""Closed-loop Bayesian-target belief geometry diagnostics for cycle 6."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm, Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import DEFAULT_VERTICES, to_xy
from envs.hmm.belief import measure
from envs.mess3.model import control_model
from experiments.mess3_reward_state_action_symmetry_cycle_6.design import (
    CYCLE_6_TRANSITION_MATRIX,
    EFFECT_SIZE,
    EXPECTED_ORACLE_POLICIES,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.task import (
    ActionSymmetryTask,
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
ROLLOUT_PATH = (
    EXPERIMENT_DIR
    / "artifacts"
    / "bayesian_target_geometry"
    / "beliefs_400k.npz"
)
OUTPUT_PATH = RESULTS_DIR / "bayesian_target_matched_support_views.png"
SUMMARY_PATH = RESULTS_DIR / "bayesian_target_matched_support_summary.json"

ACTION_LABELS = {
    NOOP_ACTION: "noop",
    POSITIVE_ACTION: "+",
    NEGATIVE_ACTION: "-",
}
VARIANTS = (1, 2, 3)


def policy_label(variant: int) -> str:
    """Return the compact target-policy label for one variant."""

    return "(" + ", ".join(
        ACTION_LABELS[action] for action in EXPECTED_ORACLE_POLICIES[variant]
    ) + ")"


def build_task(variant: int) -> tuple[object, ActionSymmetryTask]:
    """Build the cycle-6 HMM model and action task for one variant."""

    model = control_model(
        alpha=0.85,
        transition_matrix=CYCLE_6_TRANSITION_MATRIX,
    )
    task = ActionSymmetryTask(
        model=model,
        variant=variant,
        effect_size=EFFECT_SIZE,
    )
    return model, task


def bayesian_target_actions(beliefs: np.ndarray, variant: int) -> np.ndarray:
    """Map decision-time beliefs to the intended Bayesian-target actions."""

    policy = np.asarray(EXPECTED_ORACLE_POLICIES[variant], dtype=np.int64)
    return policy[np.argmax(beliefs, axis=1)]


def enumerate_policy_support(
    variant: int,
    *,
    depth: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Enumerate observation histories under the closed-loop policy.

    ``depth`` counts the reset observation. The returned arrays contain every
    prefix through ``depth`` so the finite-depth reachable set matches the
    paper-style context enumeration; summary metadata separately records the
    terminal-depth checks. Actions are recomputed from each node's belief before
    branching to the next observation.
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")

    model, task = build_task(variant)
    likelihood = np.asarray(model.emission_matrix, dtype=np.float64)
    initial = np.asarray(model.initial_distribution, dtype=np.float64)

    beliefs = np.stack(
        [measure(initial, likelihood, observation) for observation in range(3)]
    )
    probabilities = np.array(
        [float(initial @ likelihood[:, observation]) for observation in range(3)]
    )
    mass_by_depth = {1: float(probabilities.sum())}
    support_beliefs = [beliefs]
    support_probabilities = [probabilities]

    transition_by_action = {
        action: task.transition_matrix_for_action(action)
        for action in (NOOP_ACTION, POSITIVE_ACTION, NEGATIVE_ACTION)
    }
    for current_depth in range(1, depth):
        actions = bayesian_target_actions(beliefs, variant)
        predicted = np.empty_like(beliefs)
        for action, transition in transition_by_action.items():
            mask = actions == action
            if np.any(mask):
                predicted[mask] = beliefs[mask] @ transition

        child_count = beliefs.shape[0] * 3
        child_beliefs = np.empty((child_count, beliefs.shape[1]))
        child_probabilities = np.empty(child_count)
        for observation in range(3):
            start = observation * beliefs.shape[0]
            stop = start + beliefs.shape[0]
            observation_probabilities = predicted @ likelihood[:, observation]
            child_probabilities[start:stop] = (
                probabilities * observation_probabilities
            )
            child_beliefs[start:stop] = (
                predicted
                * likelihood[np.newaxis, :, observation]
                / observation_probabilities[:, np.newaxis]
            )

        beliefs = child_beliefs
        probabilities = child_probabilities
        mass_by_depth[current_depth + 1] = float(probabilities.sum())
        support_beliefs.append(beliefs)
        support_probabilities.append(probabilities)

    terminal_actions = bayesian_target_actions(beliefs, variant)
    summary = {
        "support_nodes_through_depth": int(
            sum(layer.shape[0] for layer in support_beliefs)
        ),
        "terminal_histories": int(beliefs.shape[0]),
        "probability_mass": float(probabilities.sum()),
        "max_mass_error_by_depth": float(
            max(abs(mass - 1.0) for mass in mass_by_depth.values())
        ),
        "min_path_probability": float(probabilities.min()),
        "max_path_probability": float(probabilities.max()),
        "terminal_action_counts": {
            ACTION_LABELS[action]: int(np.sum(terminal_actions == action))
            for action in (NOOP_ACTION, POSITIVE_ACTION, NEGATIVE_ACTION)
        },
        "belief_min": beliefs.min(axis=0).tolist(),
        "belief_max": beliefs.max(axis=0).tolist(),
        "max_simplex_sum_error": float(
            np.max(np.abs(beliefs.sum(axis=1) - 1.0))
        ),
    }
    return (
        np.concatenate(support_beliefs, axis=0),
        np.concatenate(support_probabilities),
        summary,
    )


def ilr_coordinates(beliefs: np.ndarray) -> np.ndarray:
    """Map 3-state simplex beliefs to isometric log-ratio coordinates."""

    clipped = np.clip(np.asarray(beliefs, dtype=np.float64), 1e-300, 1.0)
    first = np.sqrt(0.5) * np.log(clipped[:, 0] / clipped[:, 1])
    second = np.sqrt(2.0 / 3.0) * np.log(
        np.sqrt(clipped[:, 0] * clipped[:, 1]) / clipped[:, 2]
    )
    return np.column_stack([first, second])


def draw_simplex_outline(ax) -> None:
    """Draw the ordinary 3-state simplex boundary."""

    tri = np.vstack([DEFAULT_VERTICES, DEFAULT_VERTICES[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=0.8)
    for index, label in enumerate(("state 0", "state 1", "state 2")):
        offset = (DEFAULT_VERTICES[index] - DEFAULT_VERTICES.mean(axis=0)) * 0.13
        ax.annotate(
            label,
            DEFAULT_VERTICES[index] + offset,
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 0.94)
    ax.axis("off")


def plot_matched_views(
    stationary_beliefs: dict[int, np.ndarray],
    support_beliefs: dict[int, np.ndarray],
    support_probabilities: dict[int, np.ndarray],
    *,
    depth: int,
    output_path: Path,
) -> None:
    """Plot occupancy, equal-weight support, and log-ratio support panels."""

    rarity_values = np.concatenate(
        [-np.log10(support_probabilities[variant]) for variant in VARIANTS]
    )
    rarity_norm = Normalize(
        vmin=float(rarity_values.min()),
        vmax=float(rarity_values.max()),
    )

    figure, axes = plt.subplots(
        len(VARIANTS),
        3,
        figsize=(14.0, 12.0),
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.91,
        top=0.92,
        bottom=0.12,
        wspace=0.35,
        hspace=0.55,
    )
    density_mappable = None
    support_mappable = None

    for row, variant in enumerate(VARIANTS):
        row_label = f"variant {variant} target {policy_label(variant)}"

        ax = axes[row, 0]
        xy = to_xy(stationary_beliefs[variant])
        density_mappable = ax.hexbin(
            xy[:, 0],
            xy[:, 1],
            gridsize=120,
            mincnt=1,
            cmap="viridis",
            norm=LogNorm(),
            linewidths=0,
        )
        draw_simplex_outline(ax)
        ax.set_title(
            f"{row_label}\nstationary rollout density",
            fontsize=9,
        )

        ax = axes[row, 1]
        rarity = -np.log10(support_probabilities[variant])
        xy = to_xy(support_beliefs[variant])
        order = np.argsort(rarity)
        support_mappable = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            c=rarity[order],
            s=0.12,
            alpha=0.7,
            cmap="magma",
            norm=rarity_norm,
            linewidths=0,
            rasterized=True,
        )
        draw_simplex_outline(ax)
        ax.set_title(
            f"{row_label}\nequal-weight depth≤{depth} support",
            fontsize=9,
        )

        ax = axes[row, 2]
        ilr = ilr_coordinates(support_beliefs[variant])
        order = np.argsort(rarity)
        ax.scatter(
            ilr[order, 0],
            ilr[order, 1],
            c=rarity[order],
            s=0.12,
            alpha=0.7,
            cmap="magma",
            norm=rarity_norm,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.set_xlabel("ilr1")
        ax.set_ylabel("ilr2")
        ax.set_title(
            f"{row_label}\nsame support in ilr coordinates",
            fontsize=9,
        )

    assert density_mappable is not None
    assert support_mappable is not None
    figure.colorbar(
        density_mappable,
        ax=axes[:, 0],
        fraction=0.035,
        pad=0.01,
        label="stationary count per simplex bin",
    )
    figure.colorbar(
        support_mappable,
        ax=axes[:, 1:],
        fraction=0.025,
        pad=0.01,
        label="exact rarity: -log10 Pr(history prefix)",
    )
    figure.suptitle(
        "Cycle-6 Bayesian-target belief geometry: occupancy vs reachable support",
        fontsize=14,
    )
    note = textwrap.fill(
        (
            "Left: process-weighted 400k post-burn-in rollout beliefs. "
            "Middle/right: every observation-history prefix through the stated "
            "depth is plotted once, including finite-reset transients; "
            "the Bayesian-target action is recomputed at each branch. Color "
            "preserves true prefix probability, so rare branches are visible "
            "without treating them as common visitation."
        ),
        width=150,
    )
    figure.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_stationary_beliefs(path: Path) -> dict[int, np.ndarray]:
    """Load the existing 400k post-burn-in Bayesian-target rollouts."""

    with np.load(path) as payload:
        return {
            variant: np.asarray(payload[f"beliefs_{variant}"], dtype=np.float64)
            for variant in VARIANTS
        }


def write_summary(
    path: Path,
    *,
    depth: int,
    rollout_path: Path,
    stationary_beliefs: dict[int, np.ndarray],
    support_summaries: dict[int, dict[str, object]],
) -> None:
    """Write compact validation metadata for the generated figure."""

    try:
        rollout_label = str(rollout_path.resolve().relative_to(EXPERIMENT_DIR))
    except ValueError:
        rollout_label = str(rollout_path)
    payload = {
        "estimands": {
            "stationary_density": (
                "process-weighted post-burn-in closed-loop rollout beliefs"
            ),
            "support": (
                "equal-weight observation-history prefixes under the "
                "closed-loop Bayesian-target policy"
            ),
        "log_ratio": (
                "same support beliefs shown in isometric log-ratio coordinates"
            ),
        },
        "support_depth": depth,
        "stationary_rollout_path": rollout_label,
        "variants": {},
    }
    for variant in VARIANTS:
        beliefs = stationary_beliefs[variant]
        payload["variants"][str(variant)] = {
            "target_policy": [
                ACTION_LABELS[action]
                for action in EXPECTED_ORACLE_POLICIES[variant]
            ],
            "stationary_rollout_beliefs": int(beliefs.shape[0]),
            "stationary_max_simplex_sum_error": float(
                np.max(np.abs(beliefs.sum(axis=1) - 1.0))
            ),
            "support": support_summaries[variant],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="terminal observation-history depth, counting the reset token",
    )
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROLLOUT_PATH,
        help="NPZ file containing beliefs_1..beliefs_3 stationary rollouts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="PNG path for the matched geometry figure",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="JSON path for compact validation metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stationary_beliefs = load_stationary_beliefs(args.rollout_path)
    support_beliefs: dict[int, np.ndarray] = {}
    support_probabilities: dict[int, np.ndarray] = {}
    support_summaries: dict[int, dict[str, object]] = {}
    for variant in VARIANTS:
        beliefs, probabilities, summary = enumerate_policy_support(
            variant,
            depth=args.depth,
        )
        support_beliefs[variant] = beliefs
        support_probabilities[variant] = probabilities
        support_summaries[variant] = summary
    plot_matched_views(
        stationary_beliefs,
        support_beliefs,
        support_probabilities,
        depth=args.depth,
        output_path=args.output,
    )
    write_summary(
        args.summary,
        depth=args.depth,
        rollout_path=args.rollout_path,
        stationary_beliefs=stationary_beliefs,
        support_summaries=support_summaries,
    )


if __name__ == "__main__":
    main()
