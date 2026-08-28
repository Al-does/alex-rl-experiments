"""Plot SAC 95%-CEV dimensionality and greedy task accuracy over training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter  # noqa: E402

STUDY_ROOT = Path(__file__).resolve().parent
BAYES_ACCURACY_BY_FACTOR_COUNT = {
    2: 0.392**2,
    3: 0.392**3,
}
PANEL_SPECS = (
    ("SAC, 2 factors", "sac_2_factors", 2),
    ("SAC, 3 factors", "sac_3_factors", 3),
)
CEV_COLOR = "#1f77b4"
ACCURACY_COLOR = "#ff7f0e"


def _load_trajectory(
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
            for key, value in geometry["algebraic_dimension_predictions"].items()
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
        "dimensions": np.asarray([point[1][0] for point in ordered], dtype=np.int64),
        "accuracies": np.asarray(
            [point[1][1] for point in ordered], dtype=np.float64
        ),
        "factored_prediction": predictions["factored"],
        "joint_prediction": predictions["joint"],
        "bayes_accuracy": BAYES_ACCURACY_BY_FACTOR_COUNT[factor_count],
    }


def plot_panel(
    trajectory: dict[str, Any],
    *,
    title: str,
    output: Path,
) -> Path:
    steps = trajectory["steps"]
    dimensions = trajectory["dimensions"]
    accuracies = trajectory["accuracies"]
    joint = trajectory["joint_prediction"]
    bayes_accuracy = trajectory["bayes_accuracy"]

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    step_formatter = FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
    )

    dimension_line = axis.plot(
        steps,
        dimensions,
        color=CEV_COLOR,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="95% CEV dimensions",
    )[0]
    accuracy_axis = axis.twinx()
    accuracy_line = accuracy_axis.plot(
        steps,
        accuracies,
        color=ACCURACY_COLOR,
        linewidth=2.0,
        marker="s",
        markersize=4.0,
        label="greedy task accuracy",
    )[0]

    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Dimensions for 95% explained variance", color=CEV_COLOR)
    axis.tick_params(axis="y", colors=CEV_COLOR)
    axis.spines["left"].set_color(CEV_COLOR)
    axis.xaxis.set_major_formatter(step_formatter)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlim(left=0)
    axis.set_ylim(0, max(joint, int(dimensions.max())) + 2)
    axis.grid(alpha=0.2)

    accuracy_axis.set_ylim(0.0, bayes_accuracy)
    accuracy_axis.set_yticks(np.linspace(0.0, bayes_accuracy, 5))
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    accuracy_axis.set_ylabel(
        f"Task accuracy (Bayes max {bayes_accuracy:.1%})",
        color=ACCURACY_COLOR,
    )
    accuracy_axis.tick_params(axis="y", colors=ACCURACY_COLOR)
    accuracy_axis.spines["right"].set_color(ACCURACY_COLOR)

    axis.set_title(title)
    axis.legend(
        handles=[dimension_line, accuracy_line],
        loc="best",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_sac_cev95_accuracy(
    output_dir: Path,
    *,
    study_root: Path = STUDY_ROOT,
) -> list[Path]:
    outputs: list[Path] = []
    for title, condition_dir, factor_count in PANEL_SPECS:
        trajectory = _load_trajectory(study_root, condition_dir, factor_count)
        slug = condition_dir.replace("_", "-")
        output = output_dir / f"{slug}_cev95_accuracy_over_training.png"
        plot_panel(trajectory, title=title, output=output)
        outputs.append(output)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_ROOT / "results",
    )
    args = parser.parse_args(argv)
    for path in plot_sac_cev95_accuracy(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
