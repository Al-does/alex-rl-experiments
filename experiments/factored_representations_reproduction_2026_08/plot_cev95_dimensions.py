"""Plot 95%-CEV dimensionality over training for all four study cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FuncFormatter,
    MaxNLocator,
    PercentFormatter,
)


STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "cev95_dimensions_over_training.png"
PANEL_SPECS = (
    ("PPO, 2 factors", "ppo_2_factors", 2),
    ("PPO, 3 factors", "ppo_3_factors", 3),
    ("PPO + CE, 2 factors", "ppo_aux_ce_2_factors", 2),
    ("PPO + CE, 3 factors", "ppo_aux_ce_3_factors", 3),
)
BAYES_ACCURACY_BY_FACTOR_COUNT = {
    # 50,000-episode, process-weighted estimates from
    # estimate_bayes_accuracy.py with seed 123.
    2: 0.2304,
    3: 0.1106,
}


def _load_panel(
    study_root: Path,
    condition_dir: str,
    factor_count: int,
) -> dict[str, Any]:
    points: dict[int, tuple[int, float]] = {}
    predictions: dict[str, int] | None = None
    pattern = "results/*/checkpoint_probes/steps_*/probe_battery.json"
    for path in sorted((study_root / condition_dir).glob(pattern)):
        report = json.loads(path.read_text())
        steps = int(report["agent_steps"])
        geometry = report["variance_geometry"]
        dimension = int(geometry["activation"]["cev95_dimension"])
        accuracy = float(report["task_accuracy_greedy"])
        if not np.isfinite(accuracy):
            raise ValueError(f"non-finite task accuracy in {path}")
        current_predictions = {
            key: int(value)
            for key, value in geometry[
                "algebraic_dimension_predictions"
            ].items()
        }
        if predictions is None:
            predictions = current_predictions
        elif predictions != current_predictions:
            raise ValueError(
                f"inconsistent dimension predictions under {condition_dir}"
            )
        previous = points.setdefault(steps, (dimension, accuracy))
        if previous[0] != dimension or not np.isclose(previous[1], accuracy):
            raise ValueError(
                f"conflicting probe values at {steps} steps under "
                f"{condition_dir}: {previous} and {(dimension, accuracy)}"
            )
    if not points or predictions is None:
        raise FileNotFoundError(
            f"no probe battery results found under {study_root / condition_dir}"
        )
    ordered = sorted(points.items())
    return {
        "steps": np.asarray([point[0] for point in ordered], dtype=np.int64),
        "dimensions": np.asarray(
            [point[1][0] for point in ordered], dtype=np.int64
        ),
        "accuracies": np.asarray(
            [point[1][1] for point in ordered], dtype=np.float64
        ),
        "factored_prediction": predictions["factored"],
        "joint_prediction": predictions["joint"],
        "bayes_accuracy": BAYES_ACCURACY_BY_FACTOR_COUNT[factor_count],
    }


def load_cev95_trajectories(
    study_root: Path = STUDY_ROOT,
) -> dict[str, dict[str, Any]]:
    """Load and deduplicate committed checkpoint probes for each panel."""

    return {
        title: _load_panel(study_root, condition_dir, factor_count)
        for title, condition_dir, factor_count in PANEL_SPECS
    }


def plot_cev95_dimensions(
    output: Path = DEFAULT_OUTPUT,
    *,
    study_root: Path = STUDY_ROOT,
) -> Path:
    """Write the requested four-panel 95%-CEV training trajectory."""

    trajectories = load_cev95_trajectories(study_root)
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), squeeze=False)
    step_formatter = FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
    )

    for axis, (title, _, _) in zip(axes.flat, PANEL_SPECS, strict=True):
        trajectory = trajectories[title]
        steps = trajectory["steps"]
        dimensions = trajectory["dimensions"]
        accuracies = trajectory["accuracies"]
        factored = trajectory["factored_prediction"]
        joint = trajectory["joint_prediction"]
        bayes_accuracy = trajectory["bayes_accuracy"]

        dimension_line = axis.plot(
            steps,
            dimensions,
            color="#1768ac",
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label="measured 95% CEV",
        )[0]
        factored_line = axis.axhline(
            factored,
            color="#2d7d46",
            linestyle="--",
            linewidth=1.7,
            label=f"factored prediction ({factored})",
        )
        joint_line = axis.axhline(
            joint,
            color="#9a3c3c",
            linestyle=":",
            linewidth=1.9,
            label=f"joint prediction ({joint})",
        )
        axis.annotate(
            f"{int(dimensions[-1])} dims",
            xy=(steps[-1], dimensions[-1]),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="#1768ac",
        )
        accuracy_axis = axis.twinx()
        accuracy_line = accuracy_axis.plot(
            steps,
            accuracies,
            color="#dc7c17",
            linewidth=1.9,
            marker="s",
            markersize=4.0,
            label="greedy task accuracy",
        )[0]
        accuracy_axis.set_ylim(0.0, bayes_accuracy)
        accuracy_axis.set_yticks(np.linspace(0.0, bayes_accuracy, 5))
        accuracy_axis.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=1)
        )
        accuracy_axis.set_ylabel(
            f"Task accuracy (Bayes max {bayes_accuracy:.1%})",
            color="#a85d0b",
        )
        accuracy_axis.tick_params(axis="y", colors="#a85d0b")
        accuracy_axis.spines["right"].set_color("#dc7c17")
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Dimensions for 95% explained variance")
        axis.xaxis.set_major_formatter(step_formatter)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(left=0)
        axis.set_ylim(0, max(joint, int(dimensions.max())) + 2)
        axis.grid(alpha=0.2)
        axis.legend(
            handles=[
                dimension_line,
                accuracy_line,
                factored_line,
                joint_line,
            ],
            fontsize=7.5,
            loc="best",
        )

    figure.suptitle(
        "Activation dimensionality over training (95% cumulative variance)",
        fontsize=14,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(plot_cev95_dimensions(args.output))


if __name__ == "__main__":
    main()
