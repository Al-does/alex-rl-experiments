"""Decision boundaries on the belief simplex for the three v2 conditions.

Top row: greedy action of the exact-filter reference policy (barycentric Bellman
interpolation, training discount) over the whole simplex. Lower rows: greedy actions
of the seed-42 final transformer checkpoints (no ε and ε=0.05) plotted at the exact
decision-time belief of each visited history, with the reference boundary overlaid.

    uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.plot_decision_boundary
"""
import argparse
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_history_jax_2026_11.shared import model_spec
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering import STUDIES, restore
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.environment import ladder_process
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import empty_maps, rollout
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_refinement import InterpolatedPolicy

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "figures"
VARIANTS = {1: "Trivial", 2: "Reward-State Quotient", 3: "Full-State Quotient"}
ACTIONS = ("Noop", "Tilt X", "Tilt Y")
COLORS = ("#4c72b0", "#dd8452", "#55a868")
ROWS = (("reference", "Exact-filter reference"), (STUDIES[0], "Agent, no ε"), (STUDIES[1], "Agent, ε=0.05"))
SQ3 = np.sqrt(3) / 2


def project(b):
    return b[..., 1] + b[..., 2] / 2, b[..., 2] * SQ3


def raster(resolution):
    x, y = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, SQ3, int(resolution * SQ3)))
    b3 = y / SQ3
    b2 = x - b3 / 2
    b = np.stack((1 - b2 - b3, b2, b3), -1)
    inside = (b >= -1e-12).all(-1)
    return x, y, np.clip(b, 0, 1), inside


def agent_records(study, variant, episodes, length, warmup):
    leaf = HERE.parent / study / f"variant_{variant}_transformer"
    checkpoint, info = restore(leaf, study, leaf / "artifacts" / "decision_boundary")
    process = ladder_process(variant)
    spec = model_spec("transformer", process)
    params = load_params(checkpoint, spec)
    records = rollout(spec, process, params, empty_maps(), jax.random.PRNGKey(2042), episodes, length)
    belief = np.asarray(records["belief"])[warmup:].reshape(-1, 3)
    action = np.asarray(records["action"])[warmup:].ravel()
    return belief, action, info["sha256"]


def draw_frame(ax):
    ax.plot([0, 1, .5, 0], [0, 0, SQ3, 0], color="k", lw=.8)
    for (x, y), label, ha, va in (((0, 0), "S1", "right", "top"), ((1, 0), "S2", "left", "top"),
                                  ((.5, SQ3), "S3 (reward)", "center", "bottom")):
        ax.text(x, y, label, ha=ha, va=va, fontsize=8)
    ax.set_xlim(-.08, 1.08)
    ax.set_ylim(-.07, SQ3 + .07)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_boundary(ax, x, y, actions, inside, **style):
    for k in range(3):
        indicator = np.where(inside, (actions == k).astype(float), np.nan)
        if np.nanmin(indicator) < .5 < np.nanmax(indicator):
            ax.contour(x, y, indicator, levels=[.5], colors=style.get("color", "k"),
                       linewidths=style.get("lw", 1.), linestyles=style.get("ls", "-"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="reference simplex grid resolution")
    parser.add_argument("--gamma", type=float, default=.99, help="reference discount (training uses .99)")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=500)
    args = parser.parse_args()
    x, y, grid_b, inside = raster(args.resolution)
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.2))
    summary = dict(reference=dict(gamma=args.gamma, n=args.n), episodes=args.episodes,
                   length=args.length, warmup=args.warmup, seed=2042, panels={})
    for column, (variant, name) in enumerate(VARIANTS.items()):
        policy = InterpolatedPolicy(args.n, args.gamma, variant=variant)
        reference = policy(grid_b.reshape(-1, 3)).reshape(grid_b.shape[:2])
        alternative = InterpolatedPolicy(args.n, .995, variant=variant)(grid_b.reshape(-1, 3)).reshape(grid_b.shape[:2])
        summary["panels"][f"v{variant}_reference"] = dict(
            convergence=policy.convergence,
            simplex_area_fractions=[float((reference[inside] == k).mean()) for k in range(3)],
            area_disagreeing_with_gamma_995=float((reference != alternative)[inside].mean()))
        for row, (source, label) in enumerate(ROWS):
            ax = axes[row, column]
            draw_frame(ax)
            if source == "reference":
                ax.imshow(np.where(inside, reference, np.nan), origin="lower", cmap=cmap, vmin=-.5, vmax=2.5,
                          extent=(0, 1, 0, SQ3), interpolation="nearest", alpha=.85)
                draw_boundary(ax, x, y, reference, inside, lw=1.2)
                ax.set_title(f"{name}\n{label}", fontsize=10)
                continue
            belief, action, sha = agent_records(source, variant, args.episodes, args.length, args.warmup)
            agreement = float((policy(belief) == action).mean())
            px, py = project(belief)
            order = np.random.default_rng(0).permutation(len(action))
            ax.scatter(px[order], py[order], c=np.asarray(COLORS)[action[order]], s=2, lw=0, alpha=.5, rasterized=True)
            draw_boundary(ax, x, y, reference, inside, lw=.9)
            ax.set_title(f"{label}: agrees with reference {agreement:.1%}", fontsize=9)
            summary["panels"][f"v{variant}_{'eps05' if 'eps05' in source else 'no_eps'}"] = dict(
                checkpoint_sha256=sha, decisions=int(len(action)), reference_agreement=agreement,
                action_fractions=(np.bincount(action, minlength=3) / len(action)).tolist())
            print(variant, source, agreement, flush=True)
    handles = [Line2D([], [], marker="s", ls="", color=c, label=a) for c, a in zip(COLORS, ACTIONS)]
    handles.append(Line2D([], [], color="k", label=f"reference boundary (γ={args.gamma:g})"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("V2 persistent-nuisance env: greedy decision boundaries on the belief simplex (seed 42, final checkpoints)", fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .97))
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"decision_boundary_simplex.{suffix}", dpi=200)
    (OUT / "decision_boundary_simplex.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
