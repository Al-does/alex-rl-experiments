"""Render compact probe reports without reopening the archived checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.rocksample_jax_2026_10.belief_probe import sha256, source


def joint_metrics(report: dict, name: str) -> dict:
    joint = report["joint"]
    if "representations" in joint:
        return (
            joint["representations"]["final"]["metrics"]
            if name == "final"
            else joint["initialization"]["final"]["metrics"]
        )
    return joint[name]["metrics"]


def scalar(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def render(reports: list[dict], destination: Path) -> None:
    destination.mkdir()
    colors = plt.get_cmap("tab10")
    files = []

    def save(fig, name):
        path = destination / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for report in reports:
        seed = report["seed"]
        data = report["on_policy"]
        for ax, target in zip(axes, ("M", "J"), strict=True):
            if target == "M":
                scores = [
                    data["marginals"][name]["metrics"]
                    for name in ("initialization", "final")
                ]
                low, high = np.asarray([s["normalized_mse_ci"] for s in scores]).T
            else:
                scores = [
                    joint_metrics(data, name) for name in ("initialization", "final")
                ]
                low = high = np.asarray([s["normalized_mse"] for s in scores])
            values = np.asarray([s["normalized_mse"] for s in scores])
            steps = [0, report["env_steps"] / 1e6]
            ax.plot(steps, values, "o-", color=colors(seed), label=f"seed {seed}")
            ax.vlines(steps, low, high, color=colors(seed))
            ax.set(
                xlabel="Training environment steps (millions)",
                ylabel=r"$1-R^2$",
                title=f"{'Marginal' if target == 'M' else 'Joint'} posterior {target}",
            )
            ax.axhline(1, color="grey", linestyle=":", linewidth=1)
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Initialization and final checkpoint · paired final-policy histories")
    fig.tight_layout()
    save(fig, "probe_vs_steps.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for report in reports:
        seed = report["seed"]
        data = report["on_policy"]
        reward = data["behavior"]["return_mean"]
        sem = data["behavior"]["return_sem"]
        init_reward = report["initial_policy"]["behavior"]["return_mean"]
        for ax, target in zip(axes, ("M", "J"), strict=True):
            scores = [
                data["marginals"][name]["metrics"]
                if target == "M"
                else joint_metrics(data, name)
                for name in ("initialization", "final")
            ]
            x = [s["normalized_mse"] for s in scores]
            ax.plot(x, [init_reward, reward], "--", color=colors(seed), alpha=0.4)
            ax.scatter(
                x[0],
                init_reward,
                facecolors="none",
                edgecolors=[colors(seed)],
                marker="o",
            )
            ax.errorbar(
                x[1],
                reward,
                yerr=1.96 * sem,
                fmt="o",
                color=colors(seed),
                label=f"seed {seed}",
            )
            ax.annotate(
                str(seed), (x[1], reward), xytext=(6, 3), textcoords="offset points"
            )
            ax.set(
                xlabel=r"$1-R^2$" + f" ({target})",
                ylabel="Fresh stochastic-policy episode return",
                title=f"{'Marginal' if target == 'M' else 'Joint'} posterior {target}",
            )
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Open: initialization · filled: final · errors: 1.96 × return SEM")
    fig.tight_layout()
    save(fig, "reward_vs_probe.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    rocks = reports[0]["env"]["rocks"]
    for ax, (distribution, mask, title) in zip(
        axes,
        [
            ("on_policy", "all", "Final policy · all rows"),
            ("on_policy", "informed_relevant", "Final policy · informed/relevant"),
            ("common_checks", "informed_relevant", "Forced checks · informed/relevant"),
        ],
        strict=True,
    ):
        values = np.array(
            [
                [
                    np.nan if r is None or r["r_squared"] is None else r["r_squared"]
                    for r in [
                        rock[mask]
                        for rock in report[distribution]["marginals"]["final"][
                            "per_rock"
                        ]
                    ]
                ]
                for report in reports
            ]
        )
        im = ax.imshow(values, vmin=-0.2, vmax=1, cmap="viridis", aspect="auto")
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            ax.text(
                col,
                row,
                "n/a" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="black" if value > 0.65 else "white",
                fontsize=8,
            )
        ax.set_xticks(
            range(len(rocks)), [f"{i}\n{tuple(p)}" for i, p in enumerate(rocks)]
        )
        ax.set_yticks(range(len(reports)), [f"seed {r['seed']}" for r in reports])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label=r"Per-rock marginal $R^2$", shrink=0.8)
    save(fig, "per_rock_r_squared.png")

    for distribution in ("on_policy", "common_checks"):
        fig, axes = plt.subplots(
            len(reports), len(rocks), figsize=(17, 2.3 * len(reports)), squeeze=False
        )
        for row, report in enumerate(reports):
            example = report[distribution]["example"]
            for rock in range(len(rocks)):
                ax = axes[row, rock]
                ax.plot(
                    example["times"],
                    np.array(example["targets"])[:, rock],
                    color="black",
                    label="Bayes target",
                    linewidth=1.3,
                )
                ax.plot(
                    example["times"],
                    np.array(example["predictions"])[:, rock],
                    color=colors(report["seed"]),
                    label="Affine decode",
                    linewidth=1,
                )
                ax.set_title(f"seed {report['seed']} · rock {rock}", fontsize=9)
                ax.grid(alpha=0.2)
                if row == len(reports) - 1:
                    ax.set_xlabel("Decision step")
                if rock == 0:
                    ax.set_ylabel("P(Good)")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"{distribution.replace('_', ' ')} · first held-out episode (fixed selection)"
        )
        fig.tight_layout()
        save(fig, f"coordinates_{distribution}.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for report in reports:
        seed = report["seed"]
        _, recipe = source(seed)
        records = [
            json.loads(line)
            for line in (recipe.parent / "training_curves.jsonl")
            .read_text()
            .splitlines()
        ]
        records = [
            r for r in records if r["arm"] == "d128_kl0.1" and r["seed_index"] == seed
        ]
        ax.plot(
            [r["env_steps"] / 1e6 for r in records],
            [r["episode_return_mean"] for r in records],
            color=colors(seed),
            label=f"seed {seed}",
        )
    ax.set(
        xlabel="Training environment steps (millions)",
        ylabel="Training episode return",
        title="Archived reward curves (29 updates; no intermediate probes)",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, "training_reward.png")

    path = destination / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed",
                "return_mean",
                "return_sem",
                "marginal_init_1_minus_r2",
                "marginal_final_1_minus_r2",
                "joint_init_1_minus_r2",
                "joint_final_1_minus_r2",
                "common_marginal_final_1_minus_r2",
            ]
        )
        for r in reports:
            d = r["on_policy"]
            writer.writerow(
                [
                    r["seed"],
                    d["behavior"]["return_mean"],
                    d["behavior"]["return_sem"],
                    *[
                        d["marginals"][n]["metrics"]["normalized_mse"]
                        for n in ("initialization", "final")
                    ],
                    *[
                        joint_metrics(d, n)["normalized_mse"]
                        for n in ("initialization", "final")
                    ],
                    r["common_checks"]["marginals"]["final"]["metrics"][
                        "normalized_mse"
                    ],
                ]
            )
    files.append(path)
    lines = [
        "# RockSample d128_kl0.1 belief probes",
        "",
        "## Main measurements",
        "",
        "All probe points below use identical final-policy histories for initialization and final representations.",
        "",
        "| Seed | Fresh return ± SEM | M init 1−R² | M final 1−R² | J init 1−R² | J final 1−R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        d = r["on_policy"]
        lines.append(
            f"| {r['seed']} | {d['behavior']['return_mean']:.3f} ± {d['behavior']['return_sem']:.3f} | "
            + " | ".join(
                scalar(s["normalized_mse"])
                for s in [
                    d["marginals"]["initialization"]["metrics"],
                    d["marginals"]["final"]["metrics"],
                    joint_metrics(d, "initialization"),
                    joint_metrics(d, "final"),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Per-rock diagnostics",
        "",
        "Undefined fits have zero target variance or no eligible rows. Rock indices are zero based.",
        "",
        "| Seed | Rock | Check episodes | Good-sample episodes | Prior at end | On-policy R² (all) | On-policy R² (informed/relevant) | Forced-check R² (informed/relevant) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        d = r["on_policy"]
        for i, scores in enumerate(d["marginals"]["final"]["per_rock"]):
            forced = r["common_checks"]["marginals"]["final"]["per_rock"][i][
                "informed_relevant"
            ]
            informed = scores["informed_relevant"]
            lines.append(
                f"| {r['seed']} | {i} {tuple(rocks[i])} | {d['behavior']['check_episode_fraction'][i]:.3f} | {d['behavior']['good_sample_episode_fraction'][i]:.3f} | {d['behavior']['prior_at_end_fraction'][i]:.3f} | {scalar(scores['all']['r_squared'])} | {scalar(None if informed is None else informed['r_squared'])} | {scalar(None if forced is None else forced['r_squared'])} |"
            )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "M is the seven-dimensional vector of current Good probabilities. J is the 128-configuration product posterior. An affine map to M need not linearly represent the nonlinear product J.",
        "",
        "R² is 1 − held-out MSE / held-out coordinate-averaged target variance; predictions are unprojected. Raw MSE, target variance, sample coverage, training-only SVD-cutoff selection, and episode-bootstrap comparisons are in the seed JSONs.",
        "",
        "Relevant means unsampled and P(Good)>0. Informed/relevant additionally excludes the unchanged 0.5 prior. Certainty at 1 remains relevant. Subset scores evaluate the unrestricted probe without refitting on test rows.",
        "",
        "The primary distribution is independently reset episodes under each final stochastic policy. The matched initialization encoder sees the same actions and observations. Initial-policy rollouts provide an additional initialization-distribution baseline, recorded separately; task returns always use each policy's own rollouts.",
        "",
        "The common-check control uses identical histories across seeds: check all rocks remotely, visit each rock, check at distance zero, sample, then exit. Probes are refit on disjoint episodes in this off-policy distribution. It tests accessibility when evidence is supplied, not causal use or on-policy transfer.",
        "",
        "The initialization parameters are reconstructed from archived runtime seed 42 and both original key splits, with equality tested against ppo.init. There was no saved initialization pickle. Only the step-zero and 30,408,704-step probes exist; connecting lines are visual guides, not intermediate measurements.",
        "",
        "Joint-posterior controls include observable/current-input plus sampled-status/time features, next-sensor Good probabilities, training-label permutations, and training-covariance Gaussian feature nulls (three repetitions). Prediction of sensor readings is closely tied to marginals and is not an independent representation claim.",
        "",
        "Coordinate traces select the first test episode by the fixed episode split. Check-update metrics measure decoded differences before/after informative checks; no terminal observation is included. Privileged qualities appear only in calibration diagnostics; rewards appear only in behavioral summaries.",
        "",
        "Calibration bins and predictive sensor totals are descriptive aggregates of correlated episode rows. Bootstrap intervals resample episodes with fitted probes held fixed; they do not capture training-seed variability or refit uncertainty.",
        "",
        f"Evaluation uses {reports[0]['episodes_per_distribution']} complete episodes per policy/distribution, split equally for probe fitting and evaluation. No warmup is removed; exact cache replay begins at reset and retains the complete model receptive field.",
    ]
    path = destination / "report.md"
    path.write_text("\n".join(lines) + "\n")
    files.append(path)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "renderer_sha256": sha256(Path(__file__)),
                "inputs": {
                    f"seed{r['seed']}.json": sha256(
                        destination.parent / f"seed{r['seed']}.json"
                    )
                    for r in reports
                },
                "outputs": {p.name: sha256(p) for p in files},
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    reports = [
        json.loads((args.directory / f"seed{i}.json").read_text()) for i in range(4)
    ]
    render(reports, args.directory / "report")


if __name__ == "__main__":
    main()
