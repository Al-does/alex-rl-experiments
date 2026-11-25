"""Critic-aware probes for Belief-Quotient Ladder (eps02): REINFORCE, PPO, and SAC (actor & critics)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import numpy as np

from .targets import load_solution, linear_residual_from_belief, alpha_geometry_summary

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SEEDS = (42, 44, 46)
FINAL = 2012217344
GAMMA = 0.99

STUDIES = {
    "reinforce": "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11",
    "ppo": "belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11",
    "sac_autoalpha": "belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def discover(seeds=SEEDS):
    runs = []
    for algorithm, study in STUDIES.items():
        for variant in (1, 2, 3):
            found = []
            results_root = REPO / "experiments" / study / f"variant_{variant}_transformer" / "results"
            for manifest in results_root.glob("*/run_manifest.json"):
                m = read_json(manifest)
                seed = m["runtime"]["seed"]
                if seed not in seeds or m["runtime"]["smoke"]:
                    continue
                if m["status"] != "completed" or m["remote_artifacts"]["status"] != "completed":
                    continue
                found.append(dict(
                    algorithm=algorithm,
                    variant=variant,
                    seed=seed,
                    run_id=m["run_id"],
                    results_path=str(manifest.parent.relative_to(REPO)),
                ))
            if sorted(r["seed"] for r in found) != sorted(seeds):
                raise ValueError(f"missing seeds for {algorithm} variant {variant}: {[r['seed'] for r in found]}")
            runs.extend(sorted(found, key=lambda r: r["seed"]))
    return runs


def _download_from_b2(run, relative_path):
    from harness.storage.b2 import B2StorageConfig
    remote = read_json(REPO / run["results_path"] / "run_manifest.json")["remote_artifacts"]
    cache_path = ROOT / "artifacts" / "checkpoints" / run["run_id"] / Path(relative_path).name
    # Reuse existing algorithm_generality_probes checkpoint cache if present
    alt_path = REPO / "experiments" / "algorithm_generality_probes_eps02_2026_11" / "artifacts" / "checkpoints" / run["run_id"] / Path(relative_path).name
    if alt_path.exists() and not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.symlink_to(alt_path)
    if cache_path.exists():
        return cache_path
    client = B2StorageConfig.from_env().s3_client()
    dur = json.loads(client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read())
    entries = [f for f in dur["files"] if f["relative_path"] == relative_path]
    if len(entries) != 1:
        raise ValueError(f"missing durable file {relative_path} in {run['run_id']}")
    entry = entries[0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(remote["bucket"], entry["key"], str(cache_path))
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if digest != entry["sha256"] or cache_path.stat().st_size != entry["size_bytes"]:
        cache_path.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {cache_path}")
    return cache_path


def _unflatten_params(saved, template, prefix=""):
    import jax
    leaves, tree = jax.tree_util.tree_flatten_with_path(template)
    values = []
    for key, leaf in leaves:
        k = prefix + jax.tree_util.keystr(key)
        val = saved[k]
        if val.shape != leaf.shape or val.dtype != leaf.dtype or not np.isfinite(val).all():
            raise ValueError(f"invalid param leaf {k}")
        values.append(jax.numpy.asarray(val))
    return jax.tree_util.tree_unflatten(tree, values)


def restore_actor(run, step, spec):
    import jax
    from experiments.belief_quotient_history_jax_2026_11 import models
    path = _download_from_b2(run, f"checkpoints/step_{step:010d}.npz")
    template = models.init_params(spec, jax.random.PRNGKey(0))
    with np.load(path) as saved:
        params = _unflatten_params(saved, template)
        value_head = None
        if run["algorithm"] == "ppo" and "['value']['w']" in saved:
            value_head = {
                "w": np.asarray(saved["['value']['w']"], dtype=np.float64),
                "b": np.asarray(saved["['value']['b']"], dtype=np.float64),
            }
    return params, value_head


def restore_sac_critics(run, step, spec):
    import jax
    from experiments.belief_quotient_history_jax_2026_11 import models
    path = _download_from_b2(run, f"checkpoints/critic_step_{step:010d}.npz")
    template = models.init_params(spec, jax.random.PRNGKey(0))
    with np.load(path) as saved:
        critics = [
            _unflatten_params(saved, template, prefix=f"['critics'][{i}]")
            for i in (0, 1)
        ]
        alpha = float(saved["['alpha']"]) if "['alpha']" in saved else None
    return critics, alpha


def fit_and_score_probe(x_fit, y_fit, x_test, y_test, ridge=1e-6):
    """Affine ridge probe returning {r2, error, mse, target_var}."""
    from analysis.probes.linear import fit_affine_probe, global_mse_metrics, r2_score
    yf = np.asarray(y_fit, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)
    if yf.ndim == 1:
        yf = yf[:, None]
        yt = yt[:, None]
    target_var = float(np.mean(np.sum((yt - yt.mean(axis=0, keepdims=True)) ** 2, axis=1)))
    if target_var < 1e-12:
        return {"r2": None, "error": None, "mse": 0.0, "target_var": target_var, "degenerate_target": True}
    weight, bias = fit_affine_probe(np.asarray(x_fit, dtype=np.float64), yf, ridge=ridge)
    pred = np.asarray(x_test, dtype=np.float64) @ weight + bias
    metrics = global_mse_metrics(pred, yt)
    r2 = float(r2_score(pred, yt))
    return {
        "r2": r2,
        "error": float(1.0 - r2),
        "mse": float(metrics["mse"]),
        "target_var": target_var,
        "degenerate_target": False,
    }


def direct_r2(pred, target):
    """Direct (no refitting) R^2 and 1 - R^2 of a head prediction against target."""
    from analysis.probes.linear import r2_score
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if t.ndim == 1:
        t = t[:, None]
    tvar = float(np.mean(np.sum((t - t.mean(axis=0, keepdims=True)) ** 2, axis=1)))
    if tvar < 1e-12:
        return {"r2": None, "error": None}
    r2 = float(r2_score(p, t))
    return {"r2": r2, "error": float(1.0 - r2)}


def slice_flat(arr, start, end):
    a = np.asarray(arr)[:, start:end]
    return a.reshape(-1, *a.shape[2:])


def build_targets(sol, b_fit, b_test):
    """Construct all linear belief and critic-aware value/Q/advantage targets."""
    v_fit, v_test = sol.v_values(b_fit), sol.v_values(b_test)
    q_fit, q_test = sol.q_values(b_fit), sol.q_values(b_test)
    a_fit, a_test = q_fit - v_fit[:, None], q_test - v_test[:, None]

    dv_nl_fit, dv_nl_test, v_nl_frac = linear_residual_from_belief(b_fit, v_fit, b_test, v_test)
    dq_nl_fit, dq_nl_test, q_nl_frac = linear_residual_from_belief(b_fit, q_fit, b_test, q_test)
    da_nl_fit, da_nl_test, a_nl_frac = linear_residual_from_belief(b_fit, a_fit, b_test, a_test)

    targets_fit = {
        "b": b_fit,
        "rho": b_fit[:, 2:3],
        "delta": b_fit[:, 0:1] - b_fit[:, 1:2],
        "V_star": v_fit[:, None],
        "dV_nl": dv_nl_fit[:, None],
        "Q_star": q_fit,
        "dQ_nl": dq_nl_fit,
        "A_star": a_fit,
        "dA_nl": da_nl_fit,
    }
    targets_test = {
        "b": b_test,
        "rho": b_test[:, 2:3],
        "delta": b_test[:, 0:1] - b_test[:, 1:2],
        "V_star": v_test[:, None],
        "dV_nl": dv_nl_test[:, None],
        "Q_star": q_test,
        "dQ_nl": dq_nl_test,
        "A_star": a_test,
        "dA_nl": da_nl_test,
    }
    meta = {
        "v_nl_var_frac": float(v_nl_frac),
        "q_nl_var_frac": float(q_nl_frac),
        "a_nl_var_frac": float(a_nl_frac),
    }
    return targets_fit, targets_test, meta


def probe_representation(h_fit, h_test, targets_fit, targets_test):
    return {
        name: fit_and_score_probe(h_fit, targets_fit[name], h_test, targets_test[name])
        for name in targets_fit
    }


def run_one(task):
    run, smoke, cpu = task
    os.sched_setaffinity(0, {cpu})
    t0 = time.monotonic()

    import jax
    from experiments.belief_quotient_history_jax_2026_11 import models
    from experiments.belief_quotient_history_jax_2026_11.env import make_process
    from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout

    out_dir = ROOT / "artifacts" / ("ladder_smoke" if smoke else "ladder")
    out_path = out_dir / f"{run['algorithm']}_variant{run['variant']}_seed{run['seed']}.json"
    if out_path.exists():
        return str(out_path)

    recipe = read_json(REPO / run["results_path"] / "resolved_recipe.json")
    spec = models.ModelSpec(**recipe["model"])
    process = make_process(**recipe["process"])
    sol = load_solution(ROOT / "artifacts" / f"pbvi_variant_{run['variant']}.npz")

    actor_params, value_head = restore_actor(run, FINAL, spec)
    init_params, _ = restore_actor(run, 0, spec)

    n_fit_ep, n_test_ep = (4, 6) if smoke else (32, 40)
    t_start, t_end = 64, 600  # Stationary interior: gamma^(1024-600) < 0.014

    @jax.jit
    def replay_emb_and_logits(params, obs):
        logits, _, emb = models.sequence(spec, params, obs)
        return logits, emb

    with jax.default_matmul_precision("highest"):
        k_fit, k_test = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(run["seed"]), 2))
        raw_fit = _probe_rollout(spec, process, actor_params, k_fit, n_fit_ep, 1024)
        raw_test = _probe_rollout(spec, process, actor_params, k_test, n_test_ep, 1024)

        h_fit = slice_flat(raw_fit["embedding"], t_start, t_end)
        h_test = slice_flat(raw_test["embedding"], t_start, t_end)
        b_fit = slice_flat(raw_fit["belief"], t_start, t_end)
        b_test = slice_flat(raw_test["belief"], t_start, t_end)

        targets_fit, targets_test, nl_meta = build_targets(sol, b_fit, b_test)
        geom = alpha_geometry_summary(sol, b_test)

        # Probe trained actor (or shared trunk for PPO)
        actor_probes = probe_representation(h_fit, h_test, targets_fit, targets_test)

        # Replay matched step-0 init on exact same histories
        _, init_emb_fit = replay_emb_and_logits(init_params, raw_fit["obs"])
        _, init_emb_test = replay_emb_and_logits(init_params, raw_test["obs"])
        init_probes = probe_representation(
            slice_flat(init_emb_fit, t_start, t_end),
            slice_flat(init_emb_test, t_start, t_end),
            targets_fit,
            targets_test,
        )

        # Direct PPO value head check
        head_fidelity = {}
        if value_head is not None:
            v_hat_fit = (h_fit @ value_head["w"] + value_head["b"])[:, 0] / (1.0 - GAMMA)
            v_hat_test = (h_test @ value_head["w"] + value_head["b"])[:, 0] / (1.0 - GAMMA)
            _, vhat_nl_test, _ = linear_residual_from_belief(b_fit, v_hat_fit, b_test, v_hat_test)
            head_fidelity["ppo_v_head_vs_V_star"] = direct_r2(v_hat_test, targets_test["V_star"])
            head_fidelity["ppo_v_head_nl_vs_dV_nl"] = direct_r2(vhat_nl_test, targets_test["dV_nl"])
            head_fidelity["ppo_v_head_corr_with_dV_nl"] = (
                float(np.corrcoef(vhat_nl_test, targets_test["dV_nl"][:, 0])[0, 1])
                if np.std(targets_test["dV_nl"]) > 1e-12 and np.std(vhat_nl_test) > 1e-12
                else None
            )

        # For SAC: also probe both independent Q-critic transformers on the same histories!
        critic_probes = {}
        sac_alpha = None
        if run["algorithm"] == "sac_autoalpha":
            critics, sac_alpha = restore_sac_critics(run, FINAL, spec)
            for idx, c_params in enumerate(critics):
                q_fit_raw, c_emb_fit = replay_emb_and_logits(c_params, raw_fit["obs"])
                q_test_raw, c_emb_test = replay_emb_and_logits(c_params, raw_test["obs"])
                ch_fit = slice_flat(c_emb_fit, t_start, t_end)
                ch_test = slice_flat(c_emb_test, t_start, t_end)
                critic_probes[f"critic_{idx}"] = probe_representation(ch_fit, ch_test, targets_fit, targets_test)
                q_hat_fit = slice_flat(q_fit_raw, t_start, t_end) / (1.0 - GAMMA)
                q_hat_test = slice_flat(q_test_raw, t_start, t_end) / (1.0 - GAMMA)
                _, qhat_nl_test, _ = linear_residual_from_belief(b_fit, q_hat_fit, b_test, q_hat_test)
                head_fidelity[f"sac_q{idx}_vs_Q_star"] = direct_r2(q_hat_test, targets_test["Q_star"])
                head_fidelity[f"sac_q{idx}_nl_vs_dQ_nl"] = direct_r2(qhat_nl_test, targets_test["dQ_nl"])

        # Reward gating metrics (full post-warmup [64, 1024) and interior [64, 600))
        greedy_reward_full = float(slice_flat(raw_test["reward"], 64, 1024).mean())
        greedy_reward_interior = float(slice_flat(raw_test["reward"], t_start, t_end).mean())

    result = {
        **run,
        "smoke": smoke,
        "window": [t_start, t_end],
        "n_fit_rows": int(h_fit.shape[0]),
        "n_test_rows": int(h_test.shape[0]),
        "greedy_reward_full": greedy_reward_full,
        "greedy_reward_interior": greedy_reward_interior,
        "sac_final_alpha": sac_alpha,
        "target_nonlinear_var_frac": nl_meta,
        "alpha_geometry_on_rollout": geom,
        "actor_probes": actor_probes,
        "matched_init_probes": init_probes,
        "critic_probes": critic_probes,
        "head_fidelity": head_fidelity,
        "wall_seconds": time.monotonic() - t0,
    }
    write_json(out_path, result)
    print(f"DONE ladder {run['algorithm']} v{run['variant']} s{run['seed']} "
          f"rew={greedy_reward_full:.4f} rho={actor_probes['rho']['error']:.4f} "
          f"V*={actor_probes['V_star']['error']:.4f} "
          f"dV_nl={actor_probes['dV_nl']['error']} ({result['wall_seconds']:.1f}s)", flush=True)
    jax.clear_caches()
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cpus = sorted(os.sched_getaffinity(0))
    workers = min(args.workers, len(cpus))
    runs = discover()
    if args.smoke:
        run_one((next(r for r in runs if r["algorithm"] == "ppo" and r["variant"] == 3), True, cpus[0]))
        return
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        for i in range(0, len(runs), workers):
            batch = runs[i:i + workers]
            list(pool.map(run_one, [(r, False, cpus[j]) for j, r in enumerate(batch)], chunksize=1))


if __name__ == "__main__":
    main()
