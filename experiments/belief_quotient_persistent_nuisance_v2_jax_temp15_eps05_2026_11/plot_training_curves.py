"""Training curves and final action shares for the v2 temp15 arms, with and without epsilon-greedy.

Reads the committed ``results/<run-id>/`` folders of both studies and writes, under
``results/figures/``: one PNG per study and architecture (variants 1-3 side by side; belief
1-R^2 on a log left axis shared across the six subplots of that architecture, reward per
step on the right axis) and ``action_shares.md``. Transformer figures keep their original
``training_curves_<study>.png`` names.

    uv run python -m experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11.plot_training_curves
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
    "belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11": ("no_eps", "no ε", "Entropy 0, T=1.5, no ε"),
    "belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11": ("eps05", "ε=0.05", "Entropy 0, T=1.5, ε-greedy ε=0.05"),
}
VARIANTS = (1, 2, 3)
ARCHITECTURES = ("transformer", "memoryless", "gru128", "gru92")
ACTIONS = ("Noop", "Tilt X", "Tilt Y")


def load(study, variant, architecture):
    (run,) = sorted((EXPERIMENTS / study / f"variant_{variant}_{architecture}" / "results").glob("2026*"))
    summary = json.loads((run / "condition_summary.json").read_text())
    curves = [json.loads(line) for line in (run / "training_curves.jsonl").read_text().splitlines()]
    reports = summary["checkpoint_reports"]
    return {
        "checkpoint_steps": [r["checkpoint_step"] for r in reports],
        "belief_error": [r["targets"]["belief"]["global_mse_ratio"] for r in reports],
        "greedy_reward": [r["policy"]["reward_per_step"] for r in reports],
        "greedy_actions": reports[-1]["policy"]["action_fractions"],
        "train_steps": [c["env_steps"] for c in curves],
        "train_reward": [c["reward_per_step"] for c in curves],
        "sampled_actions": curves[-1]["action_fractions"],
        "final_step": reports[-1]["checkpoint_step"],
    }


def plot(data, architecture):
    errors = [e for runs in data.values() for run in runs.values() for e in run["belief_error"]]
    ylim = (min(errors) / 1.3, max(errors) * 1.3)
    billions = lambda steps: [s / 1e9 for s in steps]
    for study, (slug, _, title) in STUDIES.items():
        fig, axes = plt.subplots(1, len(VARIANTS), figsize=(16, 4.6), sharex=True)
        for variant, ax in zip(VARIANTS, axes):
            run = data[study][variant]
            ax.plot(billions(run["checkpoint_steps"]), run["belief_error"], color="tab:blue", lw=1.6,
                    label="belief 1−R² (greedy probe)")
            ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.set_xlim(0, run["final_step"] / 1e9)
            ax.set_xlabel("training env steps (billions)")
            ax.set_ylabel("belief 1−R² (log)", color="tab:blue")
            ax.tick_params(axis="y", colors="tab:blue")
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"Variant {variant} — {architecture}")
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
        handles, labels = left_handles + right_handles, left_labels + right_labels
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"Persistent-nuisance v2, seed 42, 2B steps — {title}")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        suffix = "" if architecture == "transformer" else f"_{architecture}"
        fig.savefig(OUT / f"training_curves_{slug}{suffix}.png", dpi=150)
        plt.close(fig)


def action_table(data):
    fmt = lambda fractions: " / ".join(f"{f:.3f}" for f in fractions)
    lines = [
        (
            "Final-checkpoint action shares (2.01B steps, seed 42), "
            f"actions 0/1/2 = {' / '.join(ACTIONS)}. Greedy is the argmax evaluation policy; "
            "sampled is the last training iteration's rollouts (T=1.5, plus the ε mixture in the ε arms)."
        ),
        "",
        f"| Architecture | Arm | Greedy {' / '.join(ACTIONS)} | Sampled {' / '.join(ACTIONS)} |",
        "|---|---|---|---|",
    ]
    for architecture in ARCHITECTURES:
        for study, (_, label, _) in STUDIES.items():
            for variant in VARIANTS:
                run = data[architecture][study][variant]
                lines.append(
                    f"| {architecture} | {label} v{variant} | {fmt(run['greedy_actions'])} | {fmt(run['sampled_actions'])} |"
                )
    (OUT / "action_shares.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        architecture: {study: {v: load(study, v, architecture) for v in VARIANTS} for study in STUDIES}
        for architecture in ARCHITECTURES
    }
    for architecture, runs in data.items():
        plot(runs, architecture)
    action_table(data)


if __name__ == "__main__":
    main()
