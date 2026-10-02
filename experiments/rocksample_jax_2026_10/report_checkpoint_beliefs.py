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
        return f"rep {runs.index(report['run_id']) + 1} · seed {report['seed']}"

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, distribution in enumerate(("common_checks", "on_policy")):
        for col, target in enumerate(("M", "J")):
            ax = axes[row, col]
            for trajectory in trajectories:
                scores = [
                    r[distribution]["marginals"]["final"]["metrics"]
                    if target == "M"
                    else joint_metrics(r[distribution], "final")
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
                ylabel=r"$1-R^2$",
                title=f"{target} · {'fixed forced-check histories' if row == 0 else 'each checkpoint’s own policy histories'}",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Actual initialization and ten archived checkpoints · M bands: fixed-probe episode bootstrap"
    )
    fig.tight_layout()
    save(fig, "probe_vs_steps.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, target in enumerate(("M", "J")):
        ax = axes[col]
        for trajectory in trajectories:
            x = [
                r["on_policy"]["marginals"]["final"]["metrics"]["normalized_mse"]
                if target == "M"
                else joint_metrics(r["on_policy"], "final")["normalized_mse"]
                for r in trajectory
            ]
            y = [r["on_policy"]["behavior"]["return_mean"] for r in trajectory]
            ax.plot(
                x,
                y,
                marker=".",
                alpha=0.7,
                label=label(trajectory[-1]),
                **style(trajectory[-1]),
            )
            ax.errorbar(
                x[-1],
                y[-1],
                yerr=1.96 * trajectory[-1]["on_policy"]["behavior"]["return_sem"],
                fmt="o",
                color=colors(trajectory[-1]["seed"]),
            )
            ax.annotate(
                f"r{runs.index(trajectory[-1]['run_id']) + 1}s{trajectory[-1]['seed']}",
                (x[-1], y[-1]),
                fontsize=8,
            )
        ax.set(
            xlabel=f"{target} on-policy 1−R²",
            ylabel="Fresh stochastic-policy episode return",
        )
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Checkpoint trajectories · final vertical bars: 1.96 × return SEM")
    fig.tight_layout()
    save(fig, "reward_vs_probe.png")

    for distribution, mask in (
        ("on_policy", "all"),
        ("common_checks", "informed_relevant"),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
        for ax, trajectory in zip(axes.flat, trajectories, strict=True):
            values = np.array(
                [rock_values(r, distribution, mask) for r in trajectory]
            ).T
            im = ax.imshow(values, vmin=-0.2, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(trajectory)), [r["update"] for r in trajectory])
            ax.set_yticks(range(7), [f"rock {i}" for i in range(7)])
            ax.set(
                xlabel="PPO update (0 = initialization)", title=label(trajectory[-1])
            )
        fig.colorbar(im, ax=axes, label="Per-rock R²", shrink=0.7)
        fig.suptitle(
            f"{distribution} · {mask} · NaN/constant coordinates blank; color clipped at −0.2"
        )
        save(fig, f"per_rock_trajectory_{distribution}.png")

    finals = [t[-1] for t in trajectories]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (distribution, mask) in zip(
        axes,
        [
            ("on_policy", "all"),
            ("on_policy", "informed_relevant"),
            ("common_checks", "informed_relevant"),
        ],
        strict=True,
    ):
        values = np.array([rock_values(r, distribution, mask) for r in finals])
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
        ax.set_title(f"{distribution}\n{mask}")
    fig.colorbar(im, ax=axes, label="Final per-rock R²", shrink=0.8)
    save(fig, "per_rock_r_squared.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(
        axes,
        (
            "sample_given_initial_good",
            "check_episode_fraction",
            "prior_at_end_fraction",
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
        ax.set_title(metric.replace("_", " "))
    fig.colorbar(
        im,
        ax=axes,
        label="Fraction · 4,096 independent confirmation episodes",
        shrink=0.8,
    )
    save(fig, "rollout_rocks.png")

    for distribution in ("on_policy", "common_checks"):
        fig, axes = plt.subplots(8, 7, figsize=(18, 17), squeeze=False)
        for row, report in enumerate(finals):
            example = report[distribution]["example"]
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
                ax.set_ylim(-0.3, 1.3)
                ax.grid(alpha=0.2)
        fig.suptitle(
            f"{distribution}: first held-out episode (no selection) · black: Bayesian target · colored: affine decode"
        )
        fig.tight_layout()
        save(fig, f"coordinates_{distribution}.png")

    rows = []
    rock_rows = []
    for trajectory in trajectories:
        for r in trajectory:
            for distribution in ("on_policy", "common_checks"):
                data = r[distribution]
                m = data["marginals"]["final"]["metrics"]
                j = joint_metrics(data, "final")
                rows.append(
                    {
                        "run_id": r["run_id"],
                        "seed": r["seed"],
                        "update": r["update"],
                        "env_steps": r["env_steps"],
                        "distribution": distribution,
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
                            "distribution": distribution,
                            "rock": rock,
                            "R2_all": None
                            if scores["all"] is None
                            else scores["all"]["r_squared"],
                            "R2_informed": None
                            if scores["informed_relevant"] is None
                            else scores["informed_relevant"]["r_squared"],
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
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
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
        finals[0]["protocol"],
        "",
        "Representation: post-final LayerNorm; exact action/observation-only targets M and J; previous action in observations. Histories start at reset, and terminal/post-exit rows are excluded. Predictions are not clipped. Read BELIEF_PROBES.md for commands and distribution caveats.",
        "",
        "All fixed-route history hashes agree across all 88 encoders. Each nominal seed has identical reconstructed initialization hashes across the two repetitions. Own-policy histories vary with checkpoint; their 1−R² curves change with visitation as well as representations. Joint scores use 128 configurations; marginal scores use seven coordinates. Bootstrap bands measure held-out episode uncertainty conditional on each fitted decoder, not training variability.",
        "",
        "The <5% sampling rule is a descriptive behavioral flag, not a threshold for absence of a representation. Coordinate-level target variances and undefined scores are preserved in JSON. Forced-history probes are separately refitted and do not establish transfer or causal use.",
        "",
        "## Repetition comparison",
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
