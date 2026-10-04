"""Compare Pusher-B belief decodability across training objectives."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parents[1]
BASELINE_RESULTS = ROOT / "results" / "probe_curves.json"
CONTROLLED_RESULTS = (
    EXPERIMENTS
    / "pusher_b_reward_state_action_symmetry_cycle_1"
    / "results"
    / "transducer_probe_summary.json"
)
OUTPUT_DIRECTORY = ROOT / "results"
EPISODE_LENGTH = 127

SERIES_ORDER = (
    "supervised",
    "token_guess_rl",
    "reward_a_variant_2",
    "reward_a_variant_3",
    "reward_b_variant_2",
    "reward_b_variant_3",
)
SERIES_STYLES = {
    "supervised": {
        "label": "Supervised next-token prediction",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    "token_guess_rl": {
        "label": "Token-guess PPO",
        "color": "#ff7f0e",
        "linestyle": "-",
        "marker": "s",
    },
    "reward_a_variant_2": {
        "label": "Reward A, variant 2",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "^",
    },
    "reward_a_variant_3": {
        "label": "Reward A, variant 3",
        "color": "#2ca02c",
        "linestyle": "--",
        "marker": "D",
    },
    "reward_b_variant_2": {
        "label": "Reward B, variant 2",
        "color": "#9467bd",
        "linestyle": "-",
        "marker": "v",
    },
    "reward_b_variant_3": {
        "label": "Reward B, variant 3",
        "color": "#9467bd",
        "linestyle": "--",
        "marker": "P",
    },
}


def _baseline_points(
    points: list[dict[str, object]],
    *,
    supervised: bool,
) -> list[dict[str, object]]:
    normalized = []
    for point in points:
        tokens_consumed = (
            int(point["sequences_seen"]) * EPISODE_LENGTH
            if supervised
            else int(point["env_steps"])
        )
        r_squared = float(point["r2"])
        normalized.append(
            {
                "checkpoint_name": str(point["label"]),
                "tokens_consumed": tokens_consumed,
                "r_squared": r_squared,
                "one_minus_r_squared": 1.0 - r_squared,
            }
        )
    return normalized


def _token_guess_points(
    runs: dict[str, list[dict[str, object]]],
    preset: str,
) -> list[dict[str, object]]:
    leaves = [f"rl_{preset}"]
    continuation = f"rl_{preset}_continue"
    if continuation in runs:
        leaves.append(continuation)
    by_tokens = {}
    for leaf in leaves:
        for point in _baseline_points(runs[leaf], supervised=False):
            by_tokens[int(point["tokens_consumed"])] = point
    return [by_tokens[tokens] for tokens in sorted(by_tokens)]


def load_combined_data(
    baseline_path: Path = BASELINE_RESULTS,
    controlled_path: Path = CONTROLLED_RESULTS,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    baseline = json.loads(baseline_path.read_text())
    controlled = json.loads(controlled_path.read_text())
    runs = baseline["runs"]
    arms = controlled["arms"]

    combined = {}
    for preset in ("b10", "b90"):
        series = {
            "supervised": _baseline_points(
                runs[f"supervised_{preset}"],
                supervised=True,
            ),
            "token_guess_rl": _token_guess_points(runs, preset),
        }
        for reward_state in ("a", "b"):
            for variant in (2, 3):
                key = f"reward_{reward_state}_variant_{variant}"
                arm = f"{preset}_reward_{reward_state}.variant_{variant}"
                series[key] = [
                    {
                        "checkpoint_name": str(point["checkpoint_name"]),
                        "tokens_consumed": int(point["env_steps"]),
                        "r_squared": float(point["r_squared"]),
                        "one_minus_r_squared": float(
                            point["one_minus_r_squared"]
                        ),
                    }
                    for point in arms[arm]
                ]
        if tuple(series) != SERIES_ORDER:
            raise ValueError(f"unexpected series for {preset}: {tuple(series)}")
        combined[preset] = series
    return combined


def plot_preset(
    preset: str,
    series: dict[str, list[dict[str, object]]],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for key in SERIES_ORDER:
        style = SERIES_STYLES[key]
        points = series[key]
        ax.plot(
            [int(point["tokens_consumed"]) for point in points],
            [float(point["one_minus_r_squared"]) for point in points],
            label=str(style["label"]),
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            marker=str(style["marker"]),
            linewidth=2,
            markersize=5,
        )

    ax.set_xscale("symlog", linthresh=1e5)
    ax.set_yscale("log")
    ax.set_xlim(-2e4, 1e9)
    ax.set_xlabel(
        "tokens consumed "
        "(supervised: updates × 512 sequences × 127 tokens; "
        "PPO: environment steps)"
    )
    ax.set_ylabel(
        r"1 - R$^2$ "
        "(held-out affine probe → Bayesian 3-state belief)"
    )
    ax.set_title(f"Pusher-B {preset}: belief decodability across objectives")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.text(
        0.5,
        0.01,
        "Probe protocols differ: baseline = 2,048 sequences/split; "
        "controlled tasks = 20,000 steps/split.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_summary(
    combined: dict[str, dict[str, list[dict[str, object]]]],
    output: Path,
) -> None:
    output.write_text(
        json.dumps(
            {
                "x_axis": "tokens_consumed",
                "y_axis": "one_minus_r_squared",
                "protocols": {
                    "supervised_and_token_guess_rl": (
                        "2048 fit and 2048 independent held-out sequences; "
                        "uncontrolled Bayesian belief target"
                    ),
                    "reward_state_action_symmetry": (
                        "20000 fit and 20000 independent held-out steps; "
                        "controlled action-conditioned Bayesian belief target"
                    ),
                },
                "note": (
                    "All series measure held-out affine belief decodability, "
                    "but the baseline and controlled studies use different "
                    "sampling budgets and task-appropriate belief filters."
                ),
                "presets": combined,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    combined = load_combined_data()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for preset in ("b10", "b90"):
        plot_preset(
            preset,
            combined[preset],
            OUTPUT_DIRECTORY / f"pusher_b_{preset}_combined_one_minus_r2.png",
        )
    write_summary(
        combined,
        OUTPUT_DIRECTORY / "pusher_b_combined_probe_curves.json",
    )


if __name__ == "__main__":
    main()
