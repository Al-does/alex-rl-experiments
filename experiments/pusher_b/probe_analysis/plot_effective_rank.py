"""Pusher-B b10: probe 1-R^2 (left axis) vs activation effective rank (right axis) across PPO checkpoints.

Run: uv run python -m experiments.pusher_b.probe_analysis.plot_effective_rank
Reads results/effective_rank.json (effective_rank.py) plus the probe curves (probe_curves.json /
probe_aux_ce.json) and writes results/pusher_b_b10_erank_{ppo,ppo_ce}.png (fig-1 twins with the
accuracy axis replaced by effective rank) and results/pusher_b_b10_grokking_diagnostics.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
PROBE_STAGE = "block_4"
CKPT_COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
CHECKPOINTS = {
    "PPO": {"A": "iter 4", "B": "iter 8", "C": "iter 143"},
    "PPO+CE": {"A": "iter 4", "B": "iter 8", "C": "iter 211"},
}
FAMILY_LEAVES = {"PPO": ("rl_b10", "rl_b10_continue"), "PPO+CE": ("rl_b10_aux_ce",)}
FAMILY_FILES = {"PPO": "ppo", "PPO+CE": "ppo_ce"}
ERR_COLOR, RANK_COLOR = "0.3", "tab:red"


def _by_steps(rows: list[dict]) -> dict[int, dict]:
    return {int(r["env_steps"]): r for r in rows}


def family_points(family: str) -> list[dict]:
    erank = json.loads((RESULTS / "effective_rank.json").read_text())["runs"]
    base = json.loads((RESULTS / "probe_curves.json").read_text())["runs"]
    aux = json.loads((RESULTS / "probe_aux_ce.json").read_text())["runs"]
    probes = {**base, **aux}
    merged: dict[int, dict] = {}
    for leaf in FAMILY_LEAVES[family]:
        probe_rows = _by_steps(probes[leaf])
        for steps, row in _by_steps(erank[leaf]).items():
            if steps in merged:
                continue
            probe = probe_rows.get(steps)  # two continuation log-spaced checkpoints were never probed
            merged[steps] = {
                "label": row["label"], "env_steps": steps,
                "one_minus_r2": None if probe is None else 1.0 - probe["r2"],
                "accuracy": None if probe is None else probe["accuracy"],
                "activations": row["activations"], "weights": row["weights"],
                "aux_head_excess_nll": row.get("aux_head_excess_nll"),
            }
    return [merged[s] for s in sorted(merged)]


def training_curves(family: str) -> list[dict]:
    rows: list[dict] = []
    for leaf in FAMILY_LEAVES[family]:
        path = sorted((ROOT.parent / leaf / "results").glob("*/training_curves.jsonl"))[-1]
        rows += [json.loads(line) for line in path.read_text().splitlines() if line]
    return sorted(rows, key=lambda r: r["steps"])


def _ckpt(points: list[dict], label: str) -> dict:
    return next(p for p in points if p["label"].startswith(label))


def render_twin(family: str, out: Path) -> None:
    points = family_points(family)
    x = [p["env_steps"] / 1e6 for p in points]
    probed = [p for p in points if p["one_minus_r2"] is not None]
    x_err = [p["env_steps"] / 1e6 for p in probed]
    err = [p["one_minus_r2"] for p in probed]
    rank = [p["activations"][PROBE_STAGE]["centered_erank"] for p in points]

    fig, ax_err = plt.subplots(figsize=(8, 4.8))
    ax_rank = ax_err.twinx()
    ax_err.plot(x_err, err, "-o", color=ERR_COLOR, lw=1.5, ms=3.5, label="probe $1-R^2$ (held-out, left axis)")
    ax_rank.plot(x, rank, "--s", color=RANK_COLOR, lw=1.5, ms=3.5, label="effective rank of probe stream (right axis)")
    offsets = {"A": (-14, 12), "B": (6, 12), "C": (-6, 12)}
    for letter, ckpt_label in CHECKPOINTS[family].items():
        p = _ckpt(points, ckpt_label)
        px = p["env_steps"] / 1e6
        for ax, py in ((ax_err, p["one_minus_r2"]), (ax_rank, p["activations"][PROBE_STAGE]["centered_erank"])):
            ax.plot(px, py, "o", color=CKPT_COLORS[letter], ms=10, mec="k", mew=0.8, zorder=5)
        ax_err.annotate(letter, (px, p["one_minus_r2"]), textcoords="offset points", xytext=offsets[letter],
                        ha="center", fontsize=12, fontweight="bold", color=CKPT_COLORS[letter])
        ax_err.plot([], [], "o", color=CKPT_COLORS[letter], ms=8, mec="k", mew=0.8,
                    label=f"{letter}: {family} {ckpt_label} ({px:.1f}M steps)")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("probe $1-R^2$ (held-out)", color=ERR_COLOR)
    ax_err.set_xlabel("PPO env steps (millions)")
    ax_rank.set_ylabel(f"effective rank (centered residual stream after last block, $d$=128)", color=RANK_COLOR)
    ax_rank.tick_params(axis="y", colors=RANK_COLOR)
    ax_rank.set_ylim(0, max(rank) * 1.15)
    ax_err.set_title(f"Task 1 (Pusher-B b=0.1), {family}, seed {SEED}: belief probe vs effective rank")
    ax_err.grid(alpha=0.3, which="both")
    handles = ax_err.get_legend_handles_labels()[0] + ax_rank.get_legend_handles_labels()[0]
    labels = ax_err.get_legend_handles_labels()[1] + ax_rank.get_legend_handles_labels()[1]
    ax_err.legend(handles, labels, fontsize=8, loc="center right", bbox_to_anchor=(0.99, 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def render_diagnostics(out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True)
    styles = {"PPO": ("-", "o"), "PPO+CE": ("--", "s")}
    stage_colors = {"embed": "0.6", "block_1": "tab:blue", "block_2": "tab:cyan", "block_3": "tab:olive",
                    "block_4": "tab:red", "final_norm": "tab:purple"}
    for family in CHECKPOINTS:
        pts = family_points(family)
        x = [max(p["env_steps"], 1e5) / 1e6 for p in pts]  # init shown at 0.1M on the log axis
        ls, mk = styles[family]
        ax = axes[0, 0]
        for stage, color in stage_colors.items():
            ax.plot(x, [p["activations"][stage]["centered_erank"] for p in pts], ls, marker=mk, ms=3, color=color,
                    label=f"{stage} ({family})" if family == "PPO" else None)
        ax.set_title("effective rank per residual-stream stage (centered)")
        ax.set_ylabel("effective rank")
        ax = axes[0, 1]
        ax.plot(x, [p["activations"][PROBE_STAGE]["centered_participation_ratio"] for p in pts], ls, marker=mk, ms=3,
                color="tab:red", label=f"participation ratio ({family})")
        ax.plot(x, [p["activations"][PROBE_STAGE]["centered_n99"] for p in pts], ls, marker=mk, ms=3,
                color="tab:gray", label=f"PCs for 99% variance ({family})")
        ax.set_title(f"probe stream ({PROBE_STAGE}): dimensionality")
        ax = axes[0, 2]
        probed = [p for p in pts if p["one_minus_r2"] is not None]
        ax.plot([max(p["env_steps"], 1e5) / 1e6 for p in probed], [p["one_minus_r2"] for p in probed], ls, marker=mk,
                ms=3, color="0.3", label=f"probe 1-R² ({family})")
        if any(p["aux_head_excess_nll"] is not None for p in pts):
            ax.plot(x, [p["aux_head_excess_nll"] for p in pts], ls, marker=mk, ms=3, color="tab:green",
                    label=f"aux-head excess NLL, nats ({family})")
        ax.set_yscale("log")
        ax.set_title("probe 1-R² and (PPO+CE) aux-head excess NLL")
        ax = axes[1, 0]
        ax.plot(x, [p["weights"]["encoder_l2_norm"] for p in pts], ls, marker=mk, ms=3, color="tab:blue",
                label=f"encoder ‖θ‖₂ ({family})")
        ax.plot(x, [p["weights"]["encoder_distance_from_init"] for p in pts], ls, marker=mk, ms=3, color="tab:orange",
                label=f"encoder ‖θ−θ₀‖₂ ({family})")
        ax.set_title("weight norm and distance from initialization")
        ax.set_ylabel("L2 norm")
        ax = axes[1, 1]
        ax.plot(x, [p["weights"]["encoder_mean_erank"] for p in pts], ls, marker=mk, ms=3, color="tab:blue",
                label=f"mean over encoder matrices ({family})")
        for name, color in (("encoder.blocks.3.mlp.output.weight", "tab:red"), ("encoder.blocks.3.attention.v.weight", "tab:green")):
            ax.plot(x, [p["weights"]["per_matrix"][name]["erank"] for p in pts], ls, marker=mk, ms=3, color=color,
                    label=f"{name.removeprefix('encoder.')} ({family})")
        ax.set_title("effective rank of weight matrices")
        ax = axes[1, 2]
        ax.plot(x, [p["activations"][PROBE_STAGE]["rms_norm"] for p in pts], ls, marker=mk, ms=3, color="tab:red",
                label=f"RMS activation norm ({family})")
        ax.plot(x, [p["activations"][PROBE_STAGE]["mean_norm"] for p in pts], ls, marker=mk, ms=3, color="tab:gray",
                label=f"‖mean activation‖ ({family})")
        ax.set_yscale("log")
        ax.set_title(f"probe stream ({PROBE_STAGE}): activation scale")
        curves = training_curves(family)
        xc = [r["steps"] / 1e6 for r in curves]
        ax = axes[0, 3]
        ax.plot(xc, [r["return_mean"] for r in curves], ls, lw=1, color="tab:blue", label=f"episode return ({family})")
        ax.plot([max(p["env_steps"], 1e5) / 1e6 for p in probed], [p["accuracy"] * 127 for p in probed], ls, marker=mk,
                ms=3, color="tab:green", label=f"held-out greedy accuracy × 127 ({family})")
        ax.set_title("task performance (return per 127-step episode)")
        ax = axes[1, 3]
        ax.plot(xc, [r["entropy"] for r in curves], ls, lw=1, color="tab:purple", label=f"policy entropy, nats ({family})")
        ax.set_yscale("log")
        ax.set_title("policy entropy (training curves, every iteration)")
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("PPO env steps (millions, log; init at 0.1)")
    fig.suptitle(f"Pusher-B b=0.1 task-1 pretraining, seed {SEED}: grokking diagnostics (solid PPO, dashed PPO+CE)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    for family, stem in FAMILY_FILES.items():
        render_twin(family, RESULTS / f"pusher_b_b10_erank_{stem}.png")
    render_diagnostics(RESULTS / "pusher_b_b10_grokking_diagnostics.png")
    print("saved")


if __name__ == "__main__":
    main()
