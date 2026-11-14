"""Training curves for the v2 ctx32 reward-factors transformer runs, with and without epsilon-greedy.

Reads the committed ``results/<run-id>/`` folders of both studies and writes one PNG per study
under ``results/figures/``: one panel per condition and seed, linear env steps on x, factor and
joint 1-R^2 on a log left axis with the same range in all eight panels, reward per step on the
right axis.

    uv run python -m experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11.plot_training_curves
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
OUT = HERE / "results" / "figures"
STUDIES = {
    "reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11": ("no_eps", "Entropy 0, T=1.5, no ε"),
    "reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11": (
        "eps05",
        "Entropy 0, T=1.5, ε-greedy ε=0.05",
    ),
}
CONDITIONS = ("reward_both", "reward_factor_1")
TARGETS = {"factor_1": "tab:blue", "factor_2": "tab:green", "joint": "tab:purple"}


def load(run):
    summary = json.loads((run / "condition_summary.json").read_text())
    curves = [json.loads(line) for line in (run / "training_curves.jsonl").read_text().splitlines()]
    reports = summary["checkpoint_reports"]
    return {
        "seed": summary["seed"],
        "condition": summary["condition"],
        "checkpoint_steps": [r["checkpoint_step"] for r in reports],
        "errors": {t: [r["targets"][t]["global_mse_ratio"] for r in reports] for t in TARGETS},
        "greedy_reward": [r["policy"]["reward_per_step"] for r in reports],
        "train_steps": [c["env_steps"] for c in curves],
        "train_reward": [c["reward_per_step"] for c in curves],
    }


def load_study(study):
    runs = []
    for condition in CONDITIONS:
        results = EXPERIMENTS / study / f"{condition}_transformer" / "results"
        runs += sorted((load(run) for run in results.glob("2026*")), key=lambda r: r["seed"])
    return runs


def plot(data):
    errors = [e for runs in data.values() for run in runs for es in run["errors"].values() for e in es if e > 0]
    ylim = (min(errors) / 1.3, max(errors) * 1.3)
    xmax = max(s for runs in data.values() for run in runs for s in run["checkpoint_steps"]) / 1e9
    billions = lambda steps: [s / 1e9 for s in steps]
    for study, (slug, title) in STUDIES.items():
        runs = data[study]
        fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6))
        for run, ax in zip(runs, axes):
            for target, color in TARGETS.items():
                ax.plot(billions(run["checkpoint_steps"]), run["errors"][target], color=color, lw=1.6,
                        label=f"{target.replace('_', ' ')} 1−R² (greedy probe)")
            ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.set_xlim(0, xmax)
            ax.set_xlabel("training env steps (billions)")
            ax.set_ylabel("belief 1−R² (log)")
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{run['condition']}, seed {run['seed']}")
            right = ax.twinx()
            right.plot(billions(run["checkpoint_steps"]), run["greedy_reward"], color="tab:red", lw=1.6,
                       label="reward/step (greedy eval)")
            right.plot(billions(run["train_steps"]), run["train_reward"], color="tab:orange", lw=1.0, ls="--",
                       label="reward/step (training rollouts)")
            right.set_ylim(0, 1)
            right.set_ylabel("reward per step", color="tab:red")
            right.tick_params(axis="y", colors="tab:red")
        left_handles, left_labels = axes[0].get_legend_handles_labels()
        right_handles, right_labels = right.get_legend_handles_labels()
        fig.legend(left_handles + right_handles, left_labels + right_labels, loc="lower center", ncol=5,
                   frameon=False)
        fig.suptitle(f"Persistent-nuisance v2 reward factors, ctx-32 transformer, 2B steps — {title}")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(OUT / f"training_curves_{slug}.png", dpi=150)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot({study: load_study(study) for study in STUDIES})


if __name__ == "__main__":
    main()
