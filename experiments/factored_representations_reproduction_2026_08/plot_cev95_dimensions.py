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
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402


STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "cev95_dimensions_over_training.png"
PANEL_SPECS = (
    ("PPO, 2 factors", "ppo_2_factors"),
    ("PPO, 3 factors", "ppo_3_factors"),
    ("PPO + CE, 2 factors", "ppo_aux_ce_2_factors"),
    ("PPO + CE, 3 factors", "ppo_aux_ce_3_factors"),
)


def _load_panel(study_root: Path, condition_dir: str) -> dict[str, Any]:
    points: dict[int, int] = {}
    predictions: dict[str, int] | None = None
    pattern = "results/*/checkpoint_probes/steps_*/probe_battery.json"
    for path in sorted((study_root / condition_dir).glob(pattern)):
        report = json.loads(path.read_text())
        steps = int(report["agent_steps"])
        geometry = report["variance_geometry"]
        dimension = int(geometry["activation"]["cev95_dimension"])
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
        previous = points.setdefault(steps, dimension)
        if previous != dimension:
            raise ValueError(
                f"conflicting 95%-CEV dimensions at {steps} steps under "
                f"{condition_dir}: {previous} and {dimension}"
            )
    if not points or predictions is None:
        raise FileNotFoundError(
            f"no probe battery results found under {study_root / condition_dir}"
        )
    ordered = sorted(points.items())
    return {
        "steps": np.asarray([point[0] for point in ordered], dtype=np.int64),
        "dimensions": np.asarray(
            [point[1] for point in ordered], dtype=np.int64
        ),
        "factored_prediction": predictions["factored"],
        "joint_prediction": predictions["joint"],
    }


def load_cev95_trajectories(
    study_root: Path = STUDY_ROOT,
) -> dict[str, dict[str, Any]]:
    """Load and deduplicate committed checkpoint probes for each panel."""

    return {
        title: _load_panel(study_root, condition_dir)
        for title, condition_dir in PANEL_SPECS
    }


def plot_cev95_dimensions(
    output: Path = DEFAULT_OUTPUT,
    *,
    study_root: Path = STUDY_ROOT,
) -> Path:
    """Write the requested four-panel 95%-CEV training trajectory."""

    trajectories = load_cev95_trajectories(study_root)
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), squeeze=False)
    step_formatter = FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
    )

    for axis, (title, _) in zip(axes.flat, PANEL_SPECS, strict=True):
        trajectory = trajectories[title]
        steps = trajectory["steps"]
        dimensions = trajectory["dimensions"]
        factored = trajectory["factored_prediction"]
        joint = trajectory["joint_prediction"]

        axis.plot(
            steps,
            dimensions,
            color="#1768ac",
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label="measured 95% CEV",
        )
        axis.axhline(
            factored,
            color="#2d7d46",
            linestyle="--",
            linewidth=1.7,
            label=f"factored prediction ({factored})",
        )
        axis.axhline(
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
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Dimensions for 95% explained variance")
        axis.xaxis.set_major_formatter(step_formatter)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(left=0)
        axis.set_ylim(0, max(joint, int(dimensions.max())) + 2)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")

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
