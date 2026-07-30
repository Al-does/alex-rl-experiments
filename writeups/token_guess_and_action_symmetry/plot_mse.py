"""Generate publication figures from the checked-in probe summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "figures"
TOKEN_SOURCE = DATA_DIR / "token_guess_cycle2_mse_curves.json"
CYCLE4_SOURCE = DATA_DIR / "action_symmetry_cycle4_mse_curves.json"

COLORS = {
    "ppo": "#0072B2",
    "predictive_loss": "#D55E00",
    "decoupled_kelly": "#009E73",
    "variant_1": "#0072B2",
    "variant_2": "#D55E00",
    "variant_3": "#009E73",
}
LABELS = {
    "ppo": "PPO",
    "predictive_loss": "PPO + next-token CE",
    "decoupled_kelly": "PPO + decoupled Kelly",
    "variant_1": "Variant 1",
    "variant_2": "Variant 2",
    "variant_3": "Variant 3",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _summarize(
    curves: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    sd_ddof: int,
) -> dict[str, list[dict[str, float | int]]]:
    summary: dict[str, list[dict[str, float | int]]] = {}
    for condition, seeds in curves.items():
        by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for points in seeds.values():
            for point in points:
                by_index[int(point["checkpoint_index"])].append(point)
        condition_points = []
        for checkpoint_index, points in sorted(by_index.items()):
            steps = np.asarray([point["agent_steps"] for point in points], dtype=float)
            values = np.asarray([point["mse"] for point in points], dtype=float)
            condition_points.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "n": len(values),
                    "agent_steps_mean": float(steps.mean()),
                    "agent_steps_min": int(steps.min()),
                    "agent_steps_max": int(steps.max()),
                    "mse_mean": float(values.mean()),
                    "mse_sd": float(values.std(ddof=sd_ddof)),
                    "sd_ddof": sd_ddof,
                }
            )
        summary[condition] = condition_points
    return summary


def _draw_mean_sd_curves(
    summary: dict[str, list[dict[str, float | int]]],
    *,
    title: str,
    output_stem: str,
    primary_step: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(6.7, 4.25))
    for condition, points in summary.items():
        x = np.asarray([point["agent_steps_mean"] for point in points]) / 1_000_000
        mean = np.asarray([point["mse_mean"] for point in points])
        sd = np.asarray([point["mse_sd"] for point in points])
        color = COLORS[condition]
        axis.fill_between(
            x,
            np.maximum(mean - sd, np.finfo(float).tiny),
            mean + sd,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        axis.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=LABELS[condition],
        )

    if primary_step is not None:
        axis.axvline(
            primary_step / 1_000_000,
            color="0.35",
            linestyle=(0, (2, 2)),
            linewidth=1.0,
            zorder=0,
        )
        axis.text(
            primary_step / 1_000_000,
            0.97,
            "primary comparison",
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            color="0.35",
            fontsize=8,
            rotation=90,
        )

    axis.set_yscale("log")
    axis.set_xlabel("Agent steps (millions; 0 = initialization)")
    axis.set_ylabel("Held-out affine-probe MSE")
    axis.set_title(title)
    axis.grid(axis="y", which="both", color="0.88", linewidth=0.6)
    axis.grid(axis="x", which="major", color="0.92", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    axis.margins(x=0.025)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT_DIR / f"{output_stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_token_guess() -> dict[str, list[dict[str, float | int]]]:
    payload = json.loads(TOKEN_SOURCE.read_text())
    summary = _summarize(payload["curves"], sd_ddof=0)
    _draw_mean_sd_curves(
        summary,
        title="Belief decodability during token-guess training",
        output_stem="token_guess_cycle2_mse",
        primary_step=659_185,
    )
    return summary


def plot_cycle4() -> dict[str, list[dict[str, float | int]]]:
    payload = json.loads(CYCLE4_SOURCE.read_text())
    summary = _summarize(payload["curves"], sd_ddof=0)
    _draw_mean_sd_curves(
        summary,
        title="Belief decodability during action-symmetry training",
        output_stem="action_symmetry_cycle4_mse",
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    summaries: dict[str, Any] = {
        "token_guess_cycle_2": plot_token_guess(),
        "reward_state_action_symmetry_cycle_4": plot_cycle4(),
    }
    (OUTPUT_DIR / "figure_data_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
