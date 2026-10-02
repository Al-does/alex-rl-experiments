"""Compare repeated RockSample checkpoint trajectories from compact reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.rocksample_jax_2026_10.belief_probe import ROOT, sha256
from experiments.rocksample_jax_2026_10.report_beliefs import joint_metrics, scalar


def load_trajectories(directory: Path) -> list[list[dict]]:
    trajectories = []
    for folder in sorted(directory.glob("rep*_seed*")):
        reports = sorted(
            [json.loads(p.read_text()) for p in folder.glob("update*.json")],
            key=lambda r: r["env_steps"],
        )
        if not reports or reports[0]["env_steps"] != 0 or reports[-1]["update"] != 29:
            raise ValueError(f"incomplete trajectory: {folder}")
        if [r["update"] for r in reports] != [0, 1, 2, 3, 4, 5, 6, 9, 14, 20, 29]:
            raise ValueError(f"unexpected checkpoint schedule: {folder}")
        if any(r["smoke"] for r in reports):
            raise ValueError("smoke reports cannot enter the scientific report")
        identities = {(r["run_id"], r["seed"]) for r in reports}
        if len(identities) != 1:
            raise ValueError("mixed run/seed identity")
        for distribution in ("on_policy", "common_checks"):
            if any(
                r[distribution]["n_fit_episodes"] != 256
                or r[distribution]["n_test_episodes"] != 256
                for r in reports
            ):
                raise ValueError("inconsistent episode budgets")
        trajectories.append(reports)
    if len(trajectories) != 8:
        raise ValueError("expected eight complete trajectories")
    if (
        len({r["common_checks"]["history_sha256"] for t in trajectories for r in t})
        != 1
    ):
        raise ValueError("common histories differ between encoders")
    for seed in range(4):
        matched = [t for t in trajectories if t[0]["seed"] == seed]
        if (
            len(matched) != 2
            or len({t[0]["initialization_parameter_sha256"] for t in matched}) != 1
        ):
            raise ValueError("initialization differs between matched seeds")
    return trajectories


def rock_values(report: dict, distribution: str, mask: str) -> list[float]:
    return [
        np.nan
        if rock[mask] is None or rock[mask]["r_squared"] is None
        else rock[mask]["r_squared"]
        for rock in report[distribution]["marginals"]["final"]["per_rock"]
    ]


def render(directory: Path) -> None:
    trajectories = load_trajectories(directory)
    destination = directory / "report"
    destination.mkdir()
    files = []
    colors = plt.get_cmap("tab10")
    runs = sorted({t[0]["run_id"] for t in trajectories})

    def label(report):
        return f"run {runs.index(report['run_id']) + 1} · seed {report['seed']}"

    def style(report):
        return {
            "color": colors(report["seed"]),
            "linestyle": "-" if report["run_id"] == runs[0] else "--",
        }

    def save(fig, name):
        path = destination / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, target in zip(axes, ("M", "J"), strict=True):
        for trajectory in trajectories:
            scores = [
                r["on_policy"]["marginals"]["final"]["metrics"]
                if target == "M"
                else joint_metrics(r["on_policy"], "final")
                for r in trajectory
            ]
            steps = np.array([r["env_steps"] for r in trajectory]) / 1e6
            ax.plot(
                steps,
                [s["normalized_mse"] for s in scores],
                marker=".",
                label=label(trajectory[-1]),
                **style(trajectory[-1]),
            )
            if target == "M":
                ci = np.array([s["normalized_mse_ci"] for s in scores])
                ax.fill_between(
                    steps,
                    ci[:, 0],
                    ci[:, 1],
                    color=colors(trajectory[-1]["seed"]),
                    alpha=0.07,
                )
        ax.set(
            xlabel="Training environment steps (millions)",
            ylabel=rf"{target} $1-R^2$",
            title=f"{'Marginal' if target == 'M' else 'Joint'} posterior",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("All held-out policy timesteps")
    fig.tight_layout()
    save(fig, "probe_vs_steps.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for col, target in enumerate(("M", "J")):
        ax = axes[col]
        return_ax = ax.twinx()
        for trajectory in trajectories:
            steps = np.array([r["env_steps"] for r in trajectory]) / 1e6
            error = [
                r["on_policy"]["marginals"]["final"]["metrics"]["normalized_mse"]
                if target == "M"
                else joint_metrics(r["on_policy"], "final")["normalized_mse"]
                for r in trajectory
            ]
            returns = [r["on_policy"]["behavior"]["return_mean"] for r in trajectory]
            ax.plot(
                steps,
                error,
                marker="o",
                markersize=3,
                label=label(trajectory[-1]),
                **style(trajectory[-1]),
            )
            return_ax.plot(
                steps,
                returns,
                marker="^",
                markersize=3,
                alpha=0.55,
                **style(trajectory[-1]),
            )
            return_ax.errorbar(
                steps[-1],
                returns[-1],
                yerr=1.96 * trajectory[-1]["on_policy"]["behavior"]["return_sem"],
                fmt="^",
                color=colors(trajectory[-1]["seed"]),
                alpha=0.55,
            )
        ax.set(
            xlabel="Training environment steps (millions)",
            ylabel=rf"{target} $1-R^2$",
            title=f"{'Marginal' if target == 'M' else 'Joint'} posterior",
        )
        return_ax.set_ylabel("Held-out sampled-policy episode return")
        ax.grid(alpha=0.2)
    agent_handles, agent_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        agent_handles,
        agent_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
        ncols=4,
        frameon=False,
    )
    fig.legend(
        handles=[
            Line2D([], [], color="black", marker="o", label=r"Probe $1-R^2$"),
            Line2D(
                [],
                [],
                color="black",
                marker="^",
                alpha=0.55,
                label="Sampled-policy return",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=2,
        frameon=False,
    )
    fig.suptitle("Checkpoint trajectories", y=0.99)
    fig.tight_layout(rect=(0, 0.13, 1, 0.89))
    save(fig, "reward_vs_probe.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), layout="constrained")
    for ax, trajectory in zip(axes.flat, trajectories, strict=True):
        values = np.array([rock_values(r, "on_policy", "all") for r in trajectory]).T
        im = ax.imshow(values, vmin=-0.2, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(trajectory)), [r["update"] for r in trajectory])
        ax.set_yticks(range(7), [f"rock {i}" for i in range(7)])
        ax.set(xlabel="PPO update (0 = initialization)", title=label(trajectory[-1]))
    fig.colorbar(im, ax=axes, label="Per-rock R²", shrink=0.7)
    fig.suptitle("All held-out policy timesteps · color clipped at −0.2")
    save(fig, "per_rock_trajectory_on_policy.png")

    finals = [t[-1] for t in trajectories]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    values = np.array([rock_values(r, "on_policy", "all") for r in finals])
    im = ax.imshow(values, vmin=-0.2, vmax=1, cmap="viridis", aspect="auto")
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        ax.text(
            col,
            row,
            "n/a" if np.isnan(value) else f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black" if value > 0.65 else "white",
        )
    ax.set_xticks(
        range(7),
        [f"{i}\n{tuple(p)}" for i, p in enumerate(finals[0]["env"]["rocks"])],
    )
    ax.set_yticks(range(8), [label(r) for r in finals])
    ax.set_title("All held-out policy timesteps")
    fig.colorbar(im, ax=ax, label="Final per-rock R²", shrink=0.8)
    save(fig, "per_rock_r_squared.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, (metric, title) in zip(
        axes,
        (
            ("sample_given_initial_good", "Sampled, given initially Good"),
            ("check_episode_fraction", "Checked at least once"),
            ("prior_at_end_fraction", "Posterior remains at prior at episode end"),
        ),
        strict=True,
    ):
        values = np.array([r["independent_behavior"][metric] for r in finals])
        im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if value > 0.65 else "white",
            )
        ax.set_xticks(range(7), [f"rock {i}" for i in range(7)])
        ax.set_yticks(range(8), [label(r) for r in finals])
        ax.set_title(title)
    fig.colorbar(
        im,
        ax=axes,
        label="Fraction · 4,096 independent confirmation episodes",
        shrink=0.8,
    )
    save(fig, "rollout_rocks.png")

    fig, axes = plt.subplots(8, 7, figsize=(18, 17), squeeze=False)
    predictions = [
        np.array(report["on_policy"]["example"]["predictions"])
        for report in finals
    ]
    lower = min(-0.1, min(values.min() for values in predictions)) - 0.05
    upper = max(1.1, max(values.max() for values in predictions)) + 0.05
    for row, report in enumerate(finals):
        example = report["on_policy"]["example"]
        for rock in range(7):
            ax = axes[row, rock]
            ax.plot(
                example["times"],
                np.array(example["targets"])[:, rock],
                color="black",
                linewidth=1,
            )
            ax.plot(
                example["times"],
                np.array(example["predictions"])[:, rock],
                color=colors(report["seed"]),
                linewidth=1,
            )
            ax.set_title(f"{label(report)} · rock {rock}", fontsize=8)
            ax.set_ylim(lower, upper)
            ax.grid(alpha=0.2)
    fig.suptitle("Sampled or representative episode", y=0.995)
    fig.legend(
        handles=[
            Line2D([], [], color="black", label="Black: Bayesian target"),
            Line2D([], [], color=colors(0), label="Colored: affine decoded belief"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncols=2,
        frameon=False,
    )
    fig.supxlabel("Episode time steps")
    fig.supylabel("P(Rock is Good)")
    fig.tight_layout(rect=(0.015, 0.02, 1, 0.93))
    save(fig, "coordinates_on_policy.png")

    rows = []
    rock_rows = []
    for trajectory in trajectories:
        for r in trajectory:
            data = r["on_policy"]
            m = data["marginals"]["final"]["metrics"]
            j = joint_metrics(data, "final")
            rows.append(
                {
                    "run_id": r["run_id"],
                    "seed": r["seed"],
                    "update": r["update"],
                    "env_steps": r["env_steps"],
                    "return_mean": data["behavior"]["return_mean"],
                    "return_sem": data["behavior"]["return_sem"],
                    "M_1-R2": m["normalized_mse"],
                    "M_ci_low": m["normalized_mse_ci"][0],
                    "M_ci_high": m["normalized_mse_ci"][1],
                    "J_1-R2": j["normalized_mse"],
                }
            )
            for rock, scores in enumerate(data["marginals"]["final"]["per_rock"]):
                change = data["check_updates"][rock]
                rock_rows.append(
                    {
                        "run_id": r["run_id"],
                        "seed": r["seed"],
                        "update": r["update"],
                        "env_steps": r["env_steps"],
                        "rock": rock,
                        "R2": None
                        if scores["all"] is None
                        else scores["all"]["r_squared"],
                        "sample_fraction": data["behavior"][
                            "sample_episode_fraction"
                        ][rock],
                        "prior_at_end": data["behavior"]["prior_at_end_fraction"][
                            rock
                        ],
                        "n_informative_checks": change["n_informative"],
                        "update_R2": None
                        if change["metrics"] is None
                        else change["metrics"]["r_squared"],
                        "target_change": change["mean_absolute_target_change"],
                        "decoded_change": change["mean_absolute_decoded_change"],
                    }
                )
    for name, entries in (("summary.csv", rows), ("per_rock.csv", rock_rows)):
        path = destination / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(entries[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(entries)
        files.append(path)

    lines = [
        "# Matched r7 RockSample belief trajectories",
        "",
        "88 measurements: exact initialization + 10 archived checkpoints for each of eight agents.",
        "",
        "| Agent | Independent return ± SEM | M R² on policy | J R² on policy | Rocks sampled in <5% of initially-Good episodes |",
        "|---|---:|---:|---:|---|",
    ]
    for r in finals:
        b = r["independent_behavior"]
        missing = [i for i, p in enumerate(b["sample_given_initial_good"]) if p < 0.05]
        lines.append(
            f"| {label(r)} | {b['return_mean']:.3f} ± {b['return_sem']:.3f} | {scalar(r['on_policy']['marginals']['final']['metrics']['r_squared'])} | {scalar(joint_metrics(r['on_policy'], 'final')['r_squared'])} | {missing} |"
        )
    lines += [
        "",
        "## Measurement contract",
        "",
        "512 complete held-out episodes sampled from each saved policy; 256 fit/256 test, split seed 831; CV seed 551 uses fit episodes only; raw affine predictions; 200 fixed-probe episode bootstrap resamples. Full joint null battery at final only. Policy histories vary with each checkpoint. Reset rows included; exit ends histories.",
        "",
        "Representation: post-final LayerNorm; exact action/observation-only targets M and J; previous action in observations. All held-out policy timesteps are scored. Histories start at reset, and terminal/post-exit rows are excluded. Predictions are not clipped. Read BELIEF_PROBES.md for reproduction commands.",
        "",
        "Each nominal seed has identical reconstructed initialization hashes across the two runs. Policy histories vary with checkpoint; their 1−R² curves change with visitation as well as representations. Joint scores use 128 configurations; marginal scores use seven coordinates. Bootstrap bands measure held-out episode uncertainty conditional on each fitted decoder, not training variability.",
        "",
        "The <5% sampling rule is a descriptive behavioral flag, not a threshold for absence of a representation. Coordinate-level target variances and undefined scores are preserved in JSON.",
        "",
        "## Run comparison",
        "",
    ]
    for seed in range(4):
        matched = [t for t in trajectories if t[0]["seed"] == seed]
        different = next(
            (
                a["update"]
                for a, b in zip(*matched, strict=True)
                if a.get("parameter_sha256") != b.get("parameter_sha256")
            ),
            None,
        )
        lines.append(
            f"- Seed {seed}: first saved parameter divergence at update {different}; matched initialization identical."
        )
    path = destination / "report.md"
    path.write_text("\n".join(lines) + "\n")
    files.append(path)
    inputs = list(directory.glob("rep*_seed*/update*.json"))
    if (directory / "findings.md").exists():
        inputs.append(directory / "findings.md")
    sources = [
        Path(__file__),
        ROOT / "belief_probe.py",
        ROOT / "checkpoint_beliefs.py",
        ROOT / "report_beliefs.py",
    ]
    manifest = {
        "inputs": {str(p.relative_to(directory)): sha256(p) for p in inputs},
        "outputs": {p.name: sha256(p) for p in files},
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "run_ids": runs,
        "verified_measurements": sum(map(len, trajectories)),
        "common_history_sha256": trajectories[0][0]["common_checks"]["history_sha256"],
        "training_metadata": "https://github.com/Al-does/alex-rl-experiments/pull/175",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
