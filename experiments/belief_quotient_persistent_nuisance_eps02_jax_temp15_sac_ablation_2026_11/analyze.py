"""Post-hoc analysis for the SAC Reward-State delta ablations (alpha=0 and tau=0.05)."""
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
from experiments.belief_quotient_history_jax_2026_11 import models, reinforce
from experiments.belief_quotient_history_jax_2026_11.probe import PROBE_RIDGE
from experiments.belief_quotient_history_jax_2026_11.shared import model_spec
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from harness.storage.b2 import B2StorageConfig

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
FINAL_STEP = 2012217344
GAMMA = 0.99
TEMPERATURE = 1.5

ARM_SPECS = [
    (
        "sac_autoalpha_baseline",
        REPO / "experiments" / "belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11" / "variant_2_transformer" / "results",
        (42, 44, 46, 49, 52),
    ),
    (
        "sac_alpha0",
        ROOT / "variant_2_alpha0" / "results",
        (42, 43, 44),
    ),
    (
        "sac_tau05",
        ROOT / "variant_2_tau05" / "results",
        (42, 43, 44),
    ),
]


def _discover() -> list[dict]:
    runs = []
    for arm_name, results_root, seeds in ARM_SPECS:
        arm_runs = []
        for manifest_path in sorted(results_root.glob("*/run_manifest.json")):
            m = json.loads(manifest_path.read_text())
            if m["runtime"]["smoke"] or m["status"] != "completed":
                continue
            seed = int(m["runtime"]["seed"])
            if seed not in seeds:
                continue
            arm_runs.append({
                "arm": arm_name,
                "seed": seed,
                "run_id": m["run_id"],
                "results_dir": manifest_path.parent,
                "manifest": m,
            })
        arm_runs.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in arm_runs] != list(seeds):
            raise ValueError(f"Expected seeds {seeds} for {arm_name}, got {[r['seed'] for r in arm_runs]}")
        runs.extend(arm_runs)
    return runs


def _download_file(run: dict, rel: str, client) -> Path:
    remote = run["manifest"]["remote_artifacts"]
    durability = json.loads(
        client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read()
    )
    entry = next(f for f in durability["files"] if f["relative_path"] == rel)
    local_path = ROOT / "artifacts" / "analysis_checkpoints" / run["run_id"] / Path(rel).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        client.download_file(remote["bucket"], entry["key"], str(local_path))
    digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"SHA mismatch for {local_path}")
    return local_path


def _unflatten_from_npz(saved, template, prefix: str = ""):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    values = [jnp.asarray(saved[f"{prefix}{jax.tree_util.keystr(k)}"]) for k, _ in leaves]
    return jax.tree_util.tree_unflatten(treedef, values)


def _load_actor_and_critics(run: dict, spec, client):
    actor_path = _download_file(run, f"checkpoints/step_{FINAL_STEP:010d}.npz", client)
    critic_path = _download_file(run, f"checkpoints/critic_step_{FINAL_STEP:010d}.npz", client)
    template = models.init_params(spec, jax.random.PRNGKey(0))
    with np.load(actor_path) as saved:
        actor = _unflatten_from_npz(saved, template)
    with np.load(critic_path) as saved:
        q1 = _unflatten_from_npz(saved, template, prefix="['critics'][0]")
        q2 = _unflatten_from_npz(saved, template, prefix="['critics'][1]")
        alpha = float(saved["['alpha']"]) if "['alpha']" in saved else 0.0
    return actor, q1, q2, alpha


def _rollout_records(spec, process, actor, *, seed: int, n_episodes: int, greedy: bool):
    key = jax.random.PRNGKey(seed)
    rec = reinforce.rollout(
        spec, process, actor, key, n_episodes, 1024,
        temperature=TEMPERATURE, epsilon=0.0, greedy=greedy, track_belief=True,
    )
    return {k: np.asarray(v) for k, v in rec.items()}


def _eval_trunk(spec, params, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chunks = np.array_split(obs, max(1, len(obs) // 32))
    logits_list, emb_list = [], []
    for chunk in chunks:
        logits, _, emb = models.sequence(spec, params, jnp.asarray(chunk))
        logits_list.append(np.asarray(logits, dtype=np.float64))
        emb_list.append(np.asarray(emb, dtype=np.float64))
    return np.concatenate(logits_list, axis=0), np.concatenate(emb_list, axis=0)


def _mc_returns(rewards: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float64)
    fut = np.zeros(rewards.shape[0], dtype=np.float64)
    for t in range(rewards.shape[1] - 1, -1, -1):
        fut = rewards[:, t] + gamma * fut
        out[:, t] = fut
    return out


def _probe_targets(h_tr: np.ndarray, b_tr: np.ndarray, h_te: np.ndarray, b_te: np.ndarray) -> dict[str, float]:
    targets_tr = {"belief": b_tr, "rho": b_tr[:, 2:3], "delta": b_tr[:, 0:1] - b_tr[:, 1:2]}
    targets_te = {"belief": b_te, "rho": b_te[:, 2:3], "delta": b_te[:, 0:1] - b_te[:, 1:2]}
    out = {}
    for name in ("belief", "rho", "delta"):
        w, bias = fit_affine_probe(h_tr, targets_tr[name], ridge=PROBE_RIDGE)
        pred = probe_predict(w, bias, h_te)
        out[name] = float(1.0 - r2_score(pred, targets_te[name]))
    return out


def main():
    jax.config.update("jax_default_matmul_precision", "highest")
    runs = _discover()
    process = environment.ladder_process(2)
    spec = model_spec("transformer", process)
    client = B2StorageConfig.from_env().s3_client()
    warmup = 64

    # Build a shared donor history stream from sac_autoalpha seed 42 (stochastic temp=1.5).
    donor_run = next(r for r in runs if r["arm"] == "sac_autoalpha_baseline" and r["seed"] == 42)
    donor_actor, _, _, _ = _load_actor_and_critics(donor_run, spec, client)
    donor_tr = _rollout_records(spec, process, donor_actor, seed=3001, n_episodes=64, greedy=False)
    donor_te = _rollout_records(spec, process, donor_actor, seed=4002, n_episodes=84, greedy=False)
    donor_b_tr = donor_tr["belief"][:, warmup:].reshape(-1, 3)[:60_000]
    donor_b_te = donor_te["belief"][:, warmup:].reshape(-1, 3)[:80_000]

    per_run = []
    for run in runs:
        actor, q1, q2, alpha = _load_actor_and_critics(run, spec, client)
        # Own-policy greedy evaluation (matches training-time probes) + stochastic evaluation (for MC Q calibration).
        own_tr = _rollout_records(spec, process, actor, seed=5001 + run["seed"], n_episodes=64, greedy=True)
        own_te = _rollout_records(spec, process, actor, seed=6001 + run["seed"], n_episodes=84, greedy=True)
        own_b_tr = own_tr["belief"][:, warmup:].reshape(-1, 3)[:60_000]
        own_b_te = own_te["belief"][:, warmup:].reshape(-1, 3)[:80_000]

        # Stochastic rollout for MC Q calibration (interior window [64:512] to avoid terminal horizon drop).
        stoc = _rollout_records(spec, process, actor, seed=7001 + run["seed"], n_episodes=64, greedy=False)
        gt = _mc_returns(stoc["reward"])[:, warmup:512]
        q1_stoc_l, _ = _eval_trunk(spec, q1, stoc["obs"])
        q2_stoc_l, _ = _eval_trunk(spec, q2, stoc["obs"])
        min_q = np.minimum(q1_stoc_l, q2_stoc_l) / (1.0 - GAMMA)
        act_stoc = stoc["action"]
        q_taken = np.take_along_axis(min_q, act_stoc[..., None], axis=-1)[..., 0][:, warmup:512]
        q_scale_ratio = float(q_taken.mean() / gt.mean())
        q_mc_r2 = float(r2_score(q_taken.reshape(-1, 1), gt.reshape(-1, 1)))
        q_mc_corr = float(np.corrcoef(q_taken.ravel(), gt.ravel())[0, 1])

        # Probe actor, Q1, Q2 on own greedy histories AND on shared donor histories.
        own_probes = {}
        matched_probes = {}
        for label, params in (("actor", actor), ("q1", q1), ("q2", q2)):
            _, h_tr = _eval_trunk(spec, params, own_tr["obs"])
            _, h_te = _eval_trunk(spec, params, own_te["obs"])
            own_probes[label] = _probe_targets(
                h_tr[:, warmup:].reshape(-1, spec.width)[:60_000],
                own_b_tr,
                h_te[:, warmup:].reshape(-1, spec.width)[:80_000],
                own_b_te,
            )
            _, dh_tr = _eval_trunk(spec, params, donor_tr["obs"])
            _, dh_te = _eval_trunk(spec, params, donor_te["obs"])
            matched_probes[label] = _probe_targets(
                dh_tr[:, warmup:].reshape(-1, spec.width)[:60_000],
                donor_b_tr,
                dh_te[:, warmup:].reshape(-1, spec.width)[:80_000],
                donor_b_te,
            )

        per_run.append({
            "arm": run["arm"],
            "seed": run["seed"],
            "run_id": run["run_id"],
            "alpha_final": alpha,
            "greedy_reward_per_step": float(own_te["reward"][:, warmup:].mean()),
            "stochastic_reward_per_step": float(stoc["reward"][:, warmup:].mean()),
            "stochastic_entropy": float(stoc["entropy"][:, warmup:].mean()),
            "q_scale_ratio": q_scale_ratio,
            "q_mc_corr": q_mc_corr,
            "q_mc_r2": q_mc_r2,
            "own_greedy_1mr2": own_probes,
            "matched_donor_1mr2": matched_probes,
        })

    by_arm = {}
    for arm_name, _, _ in ARM_SPECS:
        sub = [r for r in per_run if r["arm"] == arm_name]
        by_arm[arm_name] = {
            "n_seeds": len(sub),
            "greedy_reward_mean": float(np.mean([r["greedy_reward_per_step"] for r in sub])),
            "alpha_final_mean": float(np.mean([r["alpha_final"] for r in sub])),
            "q_scale_ratio_mean": float(np.mean([r["q_scale_ratio"] for r in sub])),
            "q_mc_corr_mean": float(np.mean([r["q_mc_corr"] for r in sub])),
            "own_greedy_1mr2_mean": {
                trunk: {
                    k: float(np.mean([r["own_greedy_1mr2"][trunk][k] for r in sub]))
                    for k in ("belief", "rho", "delta")
                }
                for trunk in ("actor", "q1", "q2")
            },
            "matched_donor_1mr2_mean": {
                trunk: {
                    k: float(np.mean([r["matched_donor_1mr2"][trunk][k] for r in sub]))
                    for k in ("belief", "rho", "delta")
                }
                for trunk in ("actor", "q1", "q2")
            },
        }

    summary = {"by_arm": by_arm, "per_run": per_run}
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# SAC Reward-State Quotient `delta` Ablations (`alpha=0` and `tau=0.05`)",
        "",
        "## Matched-History Held-Out `1 - R²` (Mean across seeds; donor = `sac_autoalpha` seed 42)",
        "",
        "| Arm | Greedy r/step | Final `alpha` | Q/MC scale | Actor `rho` / `delta` | Q1 `rho` / `delta` | Q2 `rho` / `delta` |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, _, _ in ARM_SPECS:
        b = by_arm[arm_name]
        mp = b["matched_donor_1mr2_mean"]
        lines.append(
            f"| `{arm_name}` | {b['greedy_reward_mean']:.4f} | {b['alpha_final_mean']:.4f} | "
            f"{b['q_scale_ratio_mean']:.3f} | "
            f"{mp['actor']['rho']:.4f} / {mp['actor']['delta']:.4f} | "
            f"{mp['q1']['rho']:.4f} / {mp['q1']['delta']:.4f} | "
            f"{mp['q2']['rho']:.4f} / {mp['q2']['delta']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-Seed Breakdown (Matched Donor `1 - R²`)")
    lines.append("")
    lines.append("| Arm | Seed | Greedy r/step | Final `alpha` | Q/MC | Actor `delta` | Q1 `delta` | Q2 `delta` | Actor `rho` | Q1 `rho` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in per_run:
        mp = r["matched_donor_1mr2"]
        lines.append(
            f"| `{r['arm']}` | {r['seed']} | {r['greedy_reward_per_step']:.4f} | {r['alpha_final']:.4f} | "
            f"{r['q_scale_ratio']:.3f} | {mp['actor']['delta']:.4f} | {mp['q1']['delta']:.4f} | "
            f"{mp['q2']['delta']:.4f} | {mp['actor']['rho']:.4f} | {mp['q1']['rho']:.4f} |"
        )
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
