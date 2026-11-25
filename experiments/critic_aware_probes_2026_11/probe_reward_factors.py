"""Critic-aware probes for Reward Relevance (eps02): REINFORCE, PPO, and SAC (actor & critics)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

from .probe_ladder import (
    FINAL, GAMMA, REPO, ROOT, SEEDS,
    direct_r2, fit_and_score_probe, read_json, restore_actor, restore_sac_critics, slice_flat, write_json,
)
from .targets import linear_residual_from_belief, load_solution

RF_STUDIES = {
    "reinforce": "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11",
    "ppo": "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_ppo_2026_11",
    "sac_autoalpha": "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_sac_autoalpha_2026_11",
}
CONDITIONS = ("reward_both", "reward_factor_1")


def discover(seeds=SEEDS):
    runs = []
    for algorithm, study in RF_STUDIES.items():
        for condition in CONDITIONS:
            found = []
            results_root = REPO / "experiments" / study / f"{condition}_transformer" / "results"
            for manifest in results_root.glob("*/run_manifest.json"):
                m = read_json(manifest)
                seed = m["runtime"]["seed"]
                if seed not in seeds or m["runtime"]["smoke"]:
                    continue
                if m["status"] != "completed" or m["remote_artifacts"]["status"] != "completed":
                    continue
                found.append(dict(
                    algorithm=algorithm,
                    condition=condition,
                    seed=seed,
                    run_id=m["run_id"],
                    results_path=str(manifest.parent.relative_to(REPO)),
                ))
            if sorted(r["seed"] for r in found) != sorted(seeds):
                raise ValueError(f"missing seeds for {algorithm} {condition}: {[r['seed'] for r in found]}")
            runs.extend(sorted(found, key=lambda r: r["seed"]))
    return runs


def factor_marginals(b_joint):
    g = np.asarray(b_joint, dtype=np.float64).reshape(-1, 3, 3)
    return g.sum(axis=2), g.sum(axis=1)


def build_rf_targets(sol_fsq, b_joint_fit, b_joint_test, condition):
    b1_f, b2_f = factor_marginals(b_joint_fit)
    b1_t, b2_t = factor_marginals(b_joint_test)
    b12_f = np.concatenate([b1_f, b2_f], axis=1)
    b12_t = np.concatenate([b1_t, b2_t], axis=1)

    v1_f, v1_t = sol_fsq.v_values(b1_f), sol_fsq.v_values(b1_t)
    v2_f, v2_t = sol_fsq.v_values(b2_f), sol_fsq.v_values(b2_t)
    q1_f, q1_t = sol_fsq.q_values(b1_f), sol_fsq.q_values(b1_t)
    q2_f, q2_t = sol_fsq.q_values(b2_f), sol_fsq.q_values(b2_t)

    dv1_f, dv1_t, v1_nl_frac = linear_residual_from_belief(b1_f, v1_f, b1_t, v1_t)
    dv2_f, dv2_t, v2_nl_frac = linear_residual_from_belief(b2_f, v2_f, b2_t, v2_t)
    dq1_f, dq1_t, _ = linear_residual_from_belief(b1_f, q1_f, b1_t, q1_t)
    dq2_f, dq2_t, _ = linear_residual_from_belief(b2_f, q2_f, b2_t, q2_t)

    if condition == "reward_both":
        v_tot_f, v_tot_t = v1_f + v2_f, v1_t + v2_t
        q_tot_f = (q1_f[:, :, None] + q2_f[:, None, :]).reshape(-1, 9)
        q_tot_t = (q1_t[:, :, None] + q2_t[:, None, :]).reshape(-1, 9)
    else:
        v_tot_f, v_tot_t = v1_f, v1_t
        q_tot_f = np.broadcast_to(q1_f[:, :, None], (len(b1_f), 3, 3)).reshape(-1, 9).copy()
        q_tot_t = np.broadcast_to(q1_t[:, :, None], (len(b1_t), 3, 3)).reshape(-1, 9).copy()

    dv_tot_f, dv_tot_t, v_tot_nl_frac = linear_residual_from_belief(b12_f, v_tot_f, b12_t, v_tot_t)
    dq_tot_f, dq_tot_t, q_tot_nl_frac = linear_residual_from_belief(b12_f, q_tot_f, b12_t, q_tot_t)

    targets_fit = {
        "b1": b1_f, "rho1": b1_f[:, 2:3], "delta1": b1_f[:, 0:1] - b1_f[:, 1:2],
        "b2": b2_f, "rho2": b2_f[:, 2:3], "delta2": b2_f[:, 0:1] - b2_f[:, 1:2],
        "V1": v1_f[:, None], "dV1_nl": dv1_f[:, None],
        "V2": v2_f[:, None], "dV2_nl": dv2_f[:, None],
        "V_star": v_tot_f[:, None], "dV_nl": dv_tot_f[:, None],
        "Q1": q1_f, "dQ1_nl": dq1_f,
        "Q2": q2_f, "dQ2_nl": dq2_f,
        "Q_star": q_tot_f, "dQ_nl": dq_tot_f,
    }
    targets_test = {
        "b1": b1_t, "rho1": b1_t[:, 2:3], "delta1": b1_t[:, 0:1] - b1_t[:, 1:2],
        "b2": b2_t, "rho2": b2_t[:, 2:3], "delta2": b2_t[:, 0:1] - b2_t[:, 1:2],
        "V1": v1_t[:, None], "dV1_nl": dv1_t[:, None],
        "V2": v2_t[:, None], "dV2_nl": dv2_t[:, None],
        "V_star": v_tot_t[:, None], "dV_nl": dv_tot_t[:, None],
        "Q1": q1_t, "dQ1_nl": dq1_t,
        "Q2": q2_t, "dQ2_nl": dq2_t,
        "Q_star": q_tot_t, "dQ_nl": dq_tot_t,
    }
    meta = {
        "v1_nl_var_frac": float(v1_nl_frac),
        "v2_nl_var_frac": float(v2_nl_frac),
        "v_tot_nl_var_frac": float(v_tot_nl_frac),
        "q_tot_nl_var_frac": float(q_tot_nl_frac),
    }
    return targets_fit, targets_test, meta, b12_f, b12_t


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

    out_dir = ROOT / "artifacts" / ("reward_factors_smoke" if smoke else "reward_factors")
    out_path = out_dir / f"{run['algorithm']}_{run['condition']}_seed{run['seed']}.json"
    if out_path.exists():
        return str(out_path)

    recipe = read_json(REPO / run["results_path"] / "resolved_recipe.json")
    spec = models.ModelSpec(**recipe["model"])
    process = make_process(**recipe["process"])
    sol_fsq = load_solution(ROOT / "artifacts" / "pbvi_variant_3.npz")

    actor_params, value_head = restore_actor(run, FINAL, spec)
    init_params, _ = restore_actor(run, 0, spec)

    n_fit_ep, n_test_ep = (4, 6) if smoke else (32, 40)
    t_start, t_end = 64, 600

    @jax.jit
    def replay_all(params, obs):
        logits, residual, emb = models.sequence(spec, params, obs)
        return logits, residual, emb

    with jax.default_matmul_precision("highest"):
        k_fit, k_test = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(run["seed"]), 2))
        raw_fit = _probe_rollout(spec, process, actor_params, k_fit, n_fit_ep, 1024)
        raw_test = _probe_rollout(spec, process, actor_params, k_test, n_test_ep, 1024)

        # Probe post-final-LN embedding (where policy & critic linear heads read)
        h_fit = slice_flat(raw_fit["embedding"], t_start, t_end)
        h_test = slice_flat(raw_test["embedding"], t_start, t_end)
        b_fit = slice_flat(raw_fit["belief"], t_start, t_end)
        b_test = slice_flat(raw_test["belief"], t_start, t_end)

        targets_fit, targets_test, nl_meta, b12_f, b12_t = build_rf_targets(sol_fsq, b_fit, b_test, run["condition"])
        actor_probes = probe_representation(h_fit, h_test, targets_fit, targets_test)

        # Matched step-0 init on exact same histories
        _, _, init_emb_fit = replay_all(init_params, raw_fit["obs"])
        _, _, init_emb_test = replay_all(init_params, raw_test["obs"])
        init_probes = probe_representation(
            slice_flat(init_emb_fit, t_start, t_end),
            slice_flat(init_emb_test, t_start, t_end),
            targets_fit,
            targets_test,
        )

        head_fidelity = {}
        if value_head is not None:
            v_hat_fit = (h_fit @ value_head["w"] + value_head["b"])[:, 0] / (1.0 - GAMMA)
            v_hat_test = (h_test @ value_head["w"] + value_head["b"])[:, 0] / (1.0 - GAMMA)
            _, vhat_nl_test, _ = linear_residual_from_belief(b12_f, v_hat_fit, b12_t, v_hat_test)
            head_fidelity["ppo_v_head_vs_V_star"] = direct_r2(v_hat_test, targets_test["V_star"])
            head_fidelity["ppo_v_head_nl_vs_dV_nl"] = direct_r2(vhat_nl_test, targets_test["dV_nl"])
            head_fidelity["ppo_v_head_corr_with_dV_nl"] = (
                float(np.corrcoef(vhat_nl_test, targets_test["dV_nl"][:, 0])[0, 1])
                if np.std(targets_test["dV_nl"]) > 1e-12 and np.std(vhat_nl_test) > 1e-12
                else None
            )

        critic_probes = {}
        sac_alpha = None
        if run["algorithm"] == "sac_autoalpha":
            critics, sac_alpha = restore_sac_critics(run, FINAL, spec)
            for idx, c_params in enumerate(critics):
                q_fit_raw, _, c_emb_fit = replay_all(c_params, raw_fit["obs"])
                q_test_raw, _, c_emb_test = replay_all(c_params, raw_test["obs"])
                ch_fit = slice_flat(c_emb_fit, t_start, t_end)
                ch_test = slice_flat(c_emb_test, t_start, t_end)
                critic_probes[f"critic_{idx}"] = probe_representation(ch_fit, ch_test, targets_fit, targets_test)
                q_hat_fit = slice_flat(q_fit_raw, t_start, t_end) / (1.0 - GAMMA)
                q_hat_test = slice_flat(q_test_raw, t_start, t_end) / (1.0 - GAMMA)
                _, qhat_nl_test, _ = linear_residual_from_belief(b12_f, q_hat_fit, b12_t, q_hat_test)
                head_fidelity[f"sac_q{idx}_vs_Q_star"] = direct_r2(q_hat_test, targets_test["Q_star"])
                head_fidelity[f"sac_q{idx}_nl_vs_dQ_nl"] = direct_r2(qhat_nl_test, targets_test["dQ_nl"])

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
        "actor_probes": actor_probes,
        "matched_init_probes": init_probes,
        "critic_probes": critic_probes,
        "head_fidelity": head_fidelity,
        "wall_seconds": time.monotonic() - t0,
    }
    write_json(out_path, result)
    print(f"DONE rf {run['algorithm']} {run['condition']} s{run['seed']} "
          f"rew={greedy_reward_full:.4f} b1={actor_probes['b1']['error']:.4f} "
          f"b2={actor_probes['b2']['error']:.4f} dV_nl={actor_probes['dV_nl']['error']:.4f} "
          f"({result['wall_seconds']:.1f}s)", flush=True)
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
        run_one((next(r for r in runs if r["algorithm"] == "ppo" and r["condition"] == "reward_factor_1"), True, cpus[0]))
        return
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        for i in range(0, len(runs), workers):
            batch = runs[i:i + workers]
            list(pool.map(run_one, [(r, False, cpus[j]) for j, r in enumerate(batch)], chunksize=1))


if __name__ == "__main__":
    main()
