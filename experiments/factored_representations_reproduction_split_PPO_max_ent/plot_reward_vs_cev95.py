"""Plot mean episode return against 95%-CEV activation dimension over training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    STUDY_ROOT / "results" / "20260831T184427Z-8826c329"
)
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "reward_vs_cev95_dimension.png"
PANEL_SPECS = (
    ("Split PPO max-ent, 2 factors", "2_factors"),
    ("Split PPO max-ent, 3 factors", "3_factors"),
)


def _load_return_by_steps(curves_path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    returns: list[float] = []
    for line in curves_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        steps.append(float(row["steps"]))
        returns.append(float(row["return_mean"]))
    if not steps:
        raise FileNotFoundError(f"no training curves in {curves_path}")
    order = np.argsort(steps)
    return np.asarray(steps, dtype=np.float64)[order], np.asarray(
        returns, dtype=np.float64
    )[order]


def _return_at_steps(steps: np.ndarray, returns: np.ndarray, target: int) -> float:
    index = int(np.argmin(np.abs(steps - float(target))))
    return float(returns[index])


def _load_panel(run_dir: Path, factor_subdir: str) -> dict[str, Any]:
    curves_path = run_dir / factor_subdir / "training_curves.jsonl"
    train_steps, train_returns = _load_return_by_steps(curves_path)

    dimensions: list[int] = []
    rewards: list[float] = []
    probe_steps: list[int] = []
    pattern = "checkpoint_probes/steps_*/probe_battery.json"
    for path in sorted((run_dir / factor_subdir).glob(pattern)):
        report = json.loads(path.read_text())
        step = int(report["agent_steps"])
        dimension = int(report["variance_geometry"]["activation"]["cev95_dimension"])
        reward = _return_at_steps(train_steps, train_returns, step)
        dimensions.append(dimension)
        rewards.append(reward)
        probe_steps.append(step)

    if not dimensions:
        raise FileNotFoundError(
            f"no probe battery results found under {run_dir / factor_subdir}"
        )

    return {
        "dimensions": np.asarray(dimensions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "steps": np.asarray(probe_steps, dtype=np.int64),
    }


def plot_reward_vs_cev95(
    run_dir: Path = DEFAULT_RUN_DIR,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    colors = ("#1768ac", "#dc7c17")

    for (title, factor_subdir), color in zip(PANEL_SPECS, colors, strict=True):
        panel = _load_panel(run_dir, factor_subdir)
        axis.plot(
            panel["dimensions"],
            panel["rewards"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5.0,
            label=title,
        )
        axis.scatter(
            panel["dimensions"][-1],
            panel["rewards"][-1],
            color=color,
            s=48,
            zorder=3,
        )

    axis.set_xlabel("Activation dimensions for 95% cumulative explained variance")
    axis.set_ylabel("Mean episode return (training reward)")
    axis.set_title(
        "Split PPO max-entropy (seed 42): reward vs 95%-CEV dimensionality"
    )
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(plot_reward_vs_cev95(args.run_dir, args.output))


if __name__ == "__main__":
    main()
