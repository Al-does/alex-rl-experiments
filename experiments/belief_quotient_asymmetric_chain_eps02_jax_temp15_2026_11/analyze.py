"""Post-hoc analysis for the asymmetric-future uncontrolled chain (D_V vs D_R)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from analysis.probes.linear import fit_affine_probe, probe_predict, r2_score
from experiments.belief_quotient_asymmetric_chain_eps02_jax_temp15_2026_11 import (
    shared as chain,
)
from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.probe import PROBE_RIDGE, collect
from experiments.belief_quotient_history_jax_2026_11.shared import model_spec
from harness.storage.b2 import B2StorageConfig

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
FINAL_STEP = 2012217344
SEEDS = (42, 43, 44)
ARMS = ("ppo_gamma99", "ppo_gamma0", "reinforce_gamma99")
ANGLES_DEG = list(range(-90, 91, 5))


def _discover() -> list[dict]:
    runs = []
    for arm in ARMS:
        arm_runs = []
        for manifest_path in sorted((ROOT / arm / "results").glob("*/run_manifest.json")):
            m = json.loads(manifest_path.read_text())
            if m["runtime"]["smoke"] or m["status"] != "completed":
                continue
            seed = int(m["runtime"]["seed"])
            if seed not in SEEDS:
                continue
            arm_runs.append({
                "arm": arm,
                "seed": seed,
                "run_id": m["run_id"],
                "results_dir": manifest_path.parent,
                "manifest": m,
            })
        arm_runs.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in arm_runs] != list(SEEDS):
            raise ValueError(f"Expected seeds {SEEDS} for {arm}, got {[r['seed'] for r in arm_runs]}")
        runs.extend(arm_runs)
    return runs


def _load_checkpoint(run: dict, spec, client) -> tuple[dict, dict | None]:
    remote = run["manifest"]["remote_artifacts"]
    durability = json.loads(
        client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read()
    )
    rel = f"checkpoints/step_{FINAL_STEP:010d}.npz"
    entry = next(f for f in durability["files"] if f["relative_path"] == rel)
    local_path = ROOT / "artifacts" / "analysis_checkpoints" / run["run_id"] / Path(rel).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        client.download_file(remote["bucket"], entry["key"], str(local_path))
    digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"SHA mismatch for {local_path}")
    template = models.init_params(spec, jax.random.PRNGKey(0))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    with np.load(local_path) as saved:
        values = [jnp.asarray(saved[jax.tree_util.keystr(k)]) for k, _ in leaves]
        actor_params = jax.tree_util.tree_unflatten(treedef, values)
        value_head = None
        if "['value']['w']" in saved:
            value_head = {
                "w": np.asarray(saved["['value']['w']"], dtype=np.float64),
                "b": np.asarray(saved["['value']['b']"], dtype=np.float64),
            }
    return actor_params, value_head


def _eval_embeddings(spec, params, obs: np.ndarray) -> np.ndarray:
    chunks = np.array_split(obs, max(1, len(obs) // 32))
    out = []
    for chunk in chunks:
        _, _, emb = models.sequence(spec, params, jnp.asarray(chunk))
        out.append(np.asarray(emb, dtype=np.float64))
    return np.concatenate(out, axis=0)


def _probe_1mr2(h_tr: np.ndarray, y_tr: np.ndarray, h_te: np.ndarray, y_te: np.ndarray) -> float:
    w, b = fit_affine_probe(h_tr, y_tr, ridge=PROBE_RIDGE)
    pred = probe_predict(w, b, h_te)
    return float(1.0 - r2_score(pred, y_te))


def _generate_matched_histories(process, spec, *, n_episodes: int = 84, episode_length: int = 1024, warmup: int = 64, seed: int = 999):
    """Generate uncontrolled chain histories (actions have identical transitions)."""
    rng = np.random.default_rng(seed)
    T = chain.TRANSITION_MATRIX
    E = np.asarray(process.emission, dtype=np.float64)
    init = np.asarray(process.initial, dtype=np.float64)
    init = init / init.sum()
    states = np.empty((n_episodes, episode_length), dtype=np.int32)
    tokens = np.empty((n_episodes, episode_length), dtype=np.int32)
    actions = rng.integers(0, process.n_actions, size=(n_episodes, episode_length), dtype=np.int32)
    beliefs = np.empty((n_episodes, episode_length, 3), dtype=np.float64)
    s = rng.choice(3, size=n_episodes, p=init)
    b = np.broadcast_to(init, (n_episodes, 3)).copy()
    for t in range(episode_length):
        states[:, t] = s
        o = (rng.random(n_episodes)[:, None] > np.cumsum(E[s], axis=1)).sum(axis=1).clip(0, 2)
        tokens[:, t] = o
        if t > 0:
            b = b @ T
        b = b * E[:, o].T
        b /= b.sum(axis=1, keepdims=True)
        beliefs[:, t] = b
        s = (rng.random(n_episodes)[:, None] > np.cumsum(T[s], axis=1)).sum(axis=1).clip(0, 2)
    prev_a = np.concatenate([np.full((n_episodes, 1), -1, dtype=np.int32), actions[:, :-1]], axis=1)
    obs = np.concatenate([
        np.eye(process.n_tokens, dtype=np.float32)[tokens],
        np.where(prev_a[..., None] >= 0, np.eye(process.n_actions, dtype=np.float32)[np.maximum(prev_a, 0)], 0.0),
    ], axis=-1)
    return obs, beliefs[:, warmup:]


def main():
    jax.config.update("jax_default_matmul_precision", "highest")
    runs = _discover()
    process = chain.chain_process()
    spec = model_spec("transformer", process)
    client = B2StorageConfig.from_env().s3_client()

    # Shared matched histories (fit & test) for apples-to-apples angular comparison.
    warmup = 64
    obs_tr, b_tr_ep = _generate_matched_histories(process, spec, n_episodes=64, warmup=warmup, seed=1001)
    obs_te, b_te_ep = _generate_matched_histories(process, spec, n_episodes=84, warmup=warmup, seed=2002)
    b_tr = b_tr_ep.reshape(-1, 3)[:60_000]
    b_te = b_te_ep.reshape(-1, 3)[:80_000]

    # Basis on the simplex: rho_dir (0 deg) and delta_dir (+90 deg).
    unit_dirs = {}
    for deg in ANGLES_DEG:
        rad = np.deg2rad(deg)
        unit_dirs[deg] = np.cos(rad) * chain.RHO_DIR + np.sin(rad) * chain.DELTA_DIR

    per_run = []
    for run in runs:
        cs = json.loads((run["results_dir"] / "condition_summary.json").read_text())
        init_rep = cs["checkpoint_reports"][0]
        final_rep = cs["checkpoint_reports"][-1]
        actor_params, value_head = _load_checkpoint(run, spec, client)

        h_tr = _eval_embeddings(spec, actor_params, obs_tr)[:, warmup:, :].reshape(-1, spec.width)[:60_000]
        h_te = _eval_embeddings(spec, actor_params, obs_te)[:, warmup:, :].reshape(-1, spec.width)[:80_000]

        matched_1mr2 = {}
        t_tr = chain.chain_targets(b_tr)
        t_te = chain.chain_targets(b_te)
        for name in ("belief", "rho", "delta", "y_v", "y_perp"):
            matched_1mr2[name] = _probe_1mr2(h_tr, t_tr[name], h_te, t_te[name])

        angular_1mr2 = {}
        for deg, vec in unit_dirs.items():
            angular_1mr2[str(deg)] = _probe_1mr2(
                h_tr, (b_tr @ vec)[:, None], h_te, (b_te @ vec)[:, None]
            )
        best_deg = min(ANGLES_DEG, key=lambda d: angular_1mr2[str(d)])
        worst_deg = max(ANGLES_DEG, key=lambda d: angular_1mr2[str(d)])

        vh_metrics = None
        if value_head is not None:
            v_pred = (h_te @ value_head["w"] + value_head["b"])[:, 0]
            v_true_99 = b_te @ chain.ALPHA_GAMMA99
            v_true_0 = b_te[:, 2]
            vh_metrics = {
                "corr_with_V_gamma99": float(np.corrcoef(v_pred, v_true_99)[0, 1]),
                "corr_with_rho": float(np.corrcoef(v_pred, v_true_0)[0, 1]),
            }

        per_run.append({
            "arm": run["arm"],
            "seed": run["seed"],
            "run_id": run["run_id"],
            "train_init_1mr2": {k: float(1.0 - v["r_squared"]) for k, v in init_rep["targets"].items()},
            "train_final_1mr2": {k: float(1.0 - v["r_squared"]) for k, v in final_rep["targets"].items()},
            "matched_final_1mr2": matched_1mr2,
            "best_angle_deg": best_deg,
            "best_angle_1mr2": angular_1mr2[str(best_deg)],
            "worst_angle_deg": worst_deg,
            "worst_angle_1mr2": angular_1mr2[str(worst_deg)],
            "angular_1mr2": angular_1mr2,
            "value_head": vh_metrics,
        })

    by_arm = {}
    for arm in ARMS:
        sub = [r for r in per_run if r["arm"] == arm]
        by_arm[arm] = {
            "n_seeds": len(sub),
            "matched_final_1mr2_mean": {
                k: float(np.mean([r["matched_final_1mr2"][k] for r in sub]))
                for k in ("belief", "y_v", "y_perp", "rho", "delta")
            },
            "matched_final_1mr2_std": {
                k: float(np.std([r["matched_final_1mr2"][k] for r in sub]))
                for k in ("belief", "y_v", "y_perp", "rho", "delta")
            },
            "best_angle_deg_seeds": [r["best_angle_deg"] for r in sub],
            "best_angle_deg_mean": float(np.mean([r["best_angle_deg"] for r in sub])),
        }

    u_v_angle_deg = float(np.degrees(np.arctan2(chain.U_V @ chain.DELTA_DIR, chain.U_V @ chain.RHO_DIR)))
    summary = {
        "study": chain.STUDY,
        "transition_matrix": chain.TRANSITION_MATRIX.tolist(),
        "alpha_gamma99": chain.ALPHA_GAMMA99.tolist(),
        "u_v_angle_deg": u_v_angle_deg,
        "by_arm": by_arm,
        "per_run": per_run,
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Asymmetric-Future Uncontrolled Chain (`1 -> 2 -> 3 -> 1`): D_V vs D_R",
        "",
        f"True discounted-value direction `y_v` sits at `{u_v_angle_deg:.1f}°` from immediate reward `rho` (`0.0°`).",
        "",
        "## Matched-History Held-Out `1 - R²` (Mean ± SD over seeds 42–44)",
        "",
        "| Arm | `y_v` (`D_V`, -29.9°) | `y_perp` (+60.1°) | `rho` (`D_R`, 0°) | `delta` (90°) | `belief` | Min-error angle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = by_arm[arm]["matched_final_1mr2_mean"]
        s = by_arm[arm]["matched_final_1mr2_std"]
        ang = by_arm[arm]["best_angle_deg_seeds"]
        lines.append(
            f"| `{arm}` | {m['y_v']:.4f} ± {s['y_v']:.4f} | {m['y_perp']:.4f} ± {s['y_perp']:.4f} | "
            f"{m['rho']:.4f} ± {s['rho']:.4f} | {m['delta']:.4f} ± {s['delta']:.4f} | "
            f"{m['belief']:.4f} ± {s['belief']:.4f} | {ang} |"
        )
    lines.append("")
    lines.append("## Per-Seed Breakdown")
    lines.append("")
    lines.append("| Arm | Seed | `y_v` | `y_perp` | `rho` | `delta` | Best angle | Value head `r(V_99)` / `r(rho)` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in per_run:
        m = r["matched_final_1mr2"]
        vh = (
            f"{r['value_head']['corr_with_V_gamma99']:.4f} / {r['value_head']['corr_with_rho']:.4f}"
            if r["value_head"] else "—"
        )
        lines.append(
            f"| `{r['arm']}` | {r['seed']} | {m['y_v']:.4f} | {m['y_perp']:.4f} | "
            f"{m['rho']:.4f} | {m['delta']:.4f} | {r['best_angle_deg']}° | {vh} |"
        )
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
