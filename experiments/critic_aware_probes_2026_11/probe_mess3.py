"""Critic-aware probes on Mess3 Passive Token Prediction (PPO vs Supervised CE).

Tests Prediction 1 of the critic-aware belief construct:
- In gamma=0 passive Mess3 token-guessing, Q*(b, a) = (b @ E)_a is affine in b,
  while V*(b) = max_a (b @ E)_a is strictly convex and ~100% nonlinear by S_3 symmetry.
- Both PPO and Supervised CE should linearly decode the 2D belief b (and Q*).
- PPO's shared post-LayerNorm residual stream (trained with a linear value head)
  must linearly encode Delta V_nl(b), whereas Supervised CE (no value critic)
  has no critic pressure to linearize Delta V_nl(b).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

from analysis.probes.linear import fit_affine_probe, r2_score
from envs.hmm import HMMEnv
from envs.mess3.model import passive_model
from experiments.critic_aware_probes_2026_11.targets import (
    linear_residual_from_belief,
    mess3_passive_targets,
)
from harness.storage.b2 import B2StorageConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = STUDY_DIR / "artifacts" / "mess3_passive"
CACHE_DIR = STUDY_DIR / "artifacts" / "b2_cache_mess3"

ARMS = (
    ("ppo", "mess3_token_guess_cycle_2-ppo-seed45-2p5m"),
    ("supervised_ce", "mess3_token_guess_cycle_2-supervised_ce-seed45-2p5m-ppomatch"),
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_rl_module(arm: str, run_id: str, ckpt_kind: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = (
        REPO_ROOT / "experiments" / "mess3_token_guess_cycle_2" / arm / "results" / run_id / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    remote = manifest["remote_artifacts"]
    bucket = remote["bucket"]
    client = B2StorageConfig.from_env().s3_client()
    dur = json.loads(
        client.get_object(Bucket=bucket, Key=remote["canonical_manifest_key"])["Body"].read()
    )
    if ckpt_kind == "initial":
        prefix = "initial_checkpoint/learner_group/learner/rl_module/default_policy/"
    else:
        ckpts = sorted(
            {
                f["relative_path"].split("/learner_group/")[0]
                for f in dur["files"]
                if "/learner_group/learner/rl_module/default_policy/" in f["relative_path"]
                and not f["relative_path"].startswith("initial_checkpoint/")
            }
        )
        prefix = f"{ckpts[-1]}/learner_group/learner/rl_module/default_policy/"

    local_dir = CACHE_DIR / run_id / ckpt_kind
    local_dir.mkdir(parents=True, exist_ok=True)
    matched = [f for f in dur["files"] if f["relative_path"].startswith(prefix)]
    if not matched:
        raise FileNotFoundError(f"No RLModule files under {prefix} for {run_id}")
    hashes = {}
    for entry in matched:
        name = Path(entry["relative_path"]).name
        dst = local_dir / name
        if not dst.exists() or _sha256_file(dst) != entry["sha256"]:
            dst.write_bytes(client.get_object(Bucket=bucket, Key=entry["key"])["Body"].read())
            actual = _sha256_file(dst)
            if actual != entry["sha256"]:
                raise RuntimeError(f"SHA256 mismatch for {dst}: {actual} != {entry['sha256']}")
        hashes[name] = entry["sha256"]
    return local_dir, {"prefix": prefix, "hashes": hashes}


def _collect_passive_episodes(
    recipe_env: dict[str, Any],
    *,
    n_episodes: int,
    episode_length: int,
    warmup: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect passive Mess3 observation sequences and exact decision-time beliefs."""
    env_cfg = dict(recipe_env)
    env_cfg["episode_length"] = episode_length
    env_cfg["randomize_first_episode_length"] = False
    env_cfg["diagnostics"] = {"belief": True}
    env = HMMEnv(env_cfg)

    obs_seq = np.zeros((n_episodes, episode_length, 3), dtype=np.float32)
    belief_seq = np.zeros((n_episodes, episode_length, 3), dtype=np.float64)
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        for t in range(episode_length):
            obs_seq[ep, t] = np.asarray(obs, dtype=np.float32)
            belief_seq[ep, t] = np.asarray(info["belief_current"], dtype=np.float64)
            obs, _r, term, trunc, info = env.step(0)
            if term or trunc:
                break
    return obs_seq, belief_seq


@torch.no_grad()
def _encode_and_predict(
    module: RLModule,
    obs_seq: np.ndarray,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run PaperActorCriticModel encoder, policy head, and value head on (E, T, 3) sequences."""
    module.eval()
    n_ep, seq_len, obs_dim = obs_seq.shape
    lookback = module.sequence_lookback
    ctx = torch.zeros((n_ep, lookback, obs_dim), dtype=torch.float32)
    ctx_len = torch.zeros((n_ep, 1), dtype=torch.float32)
    obs_t = torch.from_numpy(obs_seq)

    encoded = module.encoder(ctx, ctx_len, obs_t)  # (E, T, d_model)
    logits = module.heads.policy(encoded)  # (E, T, 3)
    values = module.heads.value(encoded).squeeze(-1)  # (E, T)

    feats = encoded[:, warmup:, :].reshape(-1, encoded.shape[-1]).cpu().numpy().astype(np.float64)
    val_np = values[:, warmup:].reshape(-1).cpu().numpy().astype(np.float64)
    logits_np = logits[:, warmup:, :].reshape(-1, logits.shape[-1]).cpu().numpy().astype(np.float64)
    return feats, val_np, logits_np


def _fit_and_score(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ridge: float = 1e-6,
) -> dict[str, float]:
    yf = np.asarray(y_fit, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)
    if yf.ndim == 1:
        yf = yf[:, None]
        yt = yt[:, None]
    w, b = fit_affine_probe(x_fit, yf, ridge=ridge)
    pred = x_test @ w + b
    r2 = float(r2_score(pred, yt))
    return {"error": 1.0 - r2, "r2": r2}


def _direct_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    var_t = float(np.mean((t - t.mean()) ** 2))
    raw_r2 = float(1.0 - np.mean((p - t) ** 2) / var_t) if var_t > 1e-12 else float("nan")
    corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-12 and np.std(t) > 1e-12 else 0.0
    # Best affine recalibration (1 scalar slope + 1 intercept)
    slope = float(np.cov(p, t, ddof=0)[0, 1] / np.var(p)) if np.var(p) > 1e-12 else 0.0
    intercept = float(t.mean() - slope * p.mean())
    cal_r2 = float(1.0 - np.mean((slope * p + intercept - t) ** 2) / var_t) if var_t > 1e-12 else float("nan")
    return {"raw_r2": raw_r2, "corr": corr, "calibrated_r2": cal_r2, "slope": slope, "intercept": intercept}


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    recipe_path = (
        REPO_ROOT
        / "experiments"
        / "mess3_token_guess_cycle_2"
        / "ppo"
        / "results"
        / "mess3_token_guess_cycle_2-ppo-seed45-2p5m"
        / "resolved_recipe.json"
    )
    recipe = json.loads(recipe_path.read_text())
    model = passive_model(alpha=0.85)
    E = np.asarray(model.emission_matrix, dtype=np.float64)

    warmup = 16
    ep_len = 256
    n_ep = 80  # 80 * (256 - 16) = 19,200 samples per split

    obs_fit, bel_fit = _collect_passive_episodes(
        recipe["environment"], n_episodes=n_ep, episode_length=ep_len, warmup=warmup, seed_offset=10_000
    )
    obs_test, bel_test = _collect_passive_episodes(
        recipe["environment"], n_episodes=n_ep, episode_length=ep_len, warmup=warmup, seed_offset=20_000
    )
    b_fit = bel_fit[:, warmup:, :].reshape(-1, 3)
    b_test = bel_test[:, warmup:, :].reshape(-1, 3)

    t_fit = mess3_passive_targets(b_fit, E)
    t_test = mess3_passive_targets(b_test, E)

    # Nonlinear residual of V*(b) after best affine function of full 2D belief b
    dV_fit, dV_test, nl_frac = linear_residual_from_belief(b_fit, t_fit["V_star"], b_test, t_test["V_star"])
    var_V = float(np.var(t_test["V_star"]))
    var_dV_nl = float(np.var(dV_test))

    results: dict[str, Any] = {
        "environment": "mess3_passive_alpha085_gamma0",
        "n_fit": int(b_fit.shape[0]),
        "n_test": int(b_test.shape[0]),
        "var_V_star": var_V,
        "var_dV_nl": var_dV_nl,
        "nonlinear_variance_fraction": nl_frac,
        "runs": {},
    }

    for arm, run_id in ARMS:
        results["runs"][arm] = {}
        for ckpt_kind in ("initial", "final"):
            local_dir, ckpt_meta = _download_rl_module(arm, run_id, ckpt_kind)
            module = RLModule.from_checkpoint(str(local_dir))
            x_fit, v_fit, _ = _encode_and_predict(module, obs_fit, warmup)
            x_test, v_test, logits_test = _encode_and_predict(module, obs_test, warmup)

            # Policy greedy accuracy on test split
            greedy_act = np.argmax(logits_test, axis=1)
            expected_acc = float(np.mean(t_test["Q_star"][np.arange(len(greedy_act)), greedy_act]))
            bayes_acc = float(np.mean(t_test["V_star"]))

            probes = {
                "belief": _fit_and_score(x_fit, b_fit, x_test, b_test),
                "Q_star": _fit_and_score(x_fit, t_fit["Q_star"], x_test, t_test["Q_star"]),
                "V_star": _fit_and_score(x_fit, t_fit["V_star"], x_test, t_test["V_star"]),
                "dV_nl": _fit_and_score(x_fit, dV_fit, x_test, dV_test),
                "A_star": _fit_and_score(x_fit, t_fit["A_star"], x_test, t_test["A_star"]),
            }

            # Direct value-head check vs V* and vs pure nonlinear residual dV_nl
            vhead_vs_V = _direct_metrics(v_test, t_test["V_star"])
            vhead_res_fit, vhead_res_test, _ = linear_residual_from_belief(b_fit, v_fit, b_test, v_test)
            vhead_vs_dV_nl = _direct_metrics(vhead_res_test, dV_test)

            results["runs"][arm][ckpt_kind] = {
                "run_id": run_id,
                "checkpoint": ckpt_meta,
                "expected_greedy_accuracy": expected_acc,
                "bayes_optimal_accuracy": bayes_acc,
                "probes": probes,
                "direct_value_head": {
                    "vs_V_star": vhead_vs_V,
                    "nonlinear_residual_vs_dV_nl": vhead_vs_dV_nl,
                },
            }

    out_path = ARTIFACTS_DIR / "mess3_passive_seed45.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    print(f"Nonlinear fraction of Var(V*): {nl_frac:.6f}")
    print()
    header = f"{'Arm':<15} {'Ckpt':<8} {'Acc':>6} {'belief':>7} {'Q*':>7} {'V*':>7} {'dV_nl':>7} {'A*':>7} {'Vhead_r2':>9} {'Vhead_nl_r':>10}"
    print(header)
    print("-" * len(header))
    for arm, _ in ARMS:
        for ckpt_kind in ("initial", "final"):
            r = results["runs"][arm][ckpt_kind]
            p = r["probes"]
            vh = r["direct_value_head"]
            print(
                f"{arm:<15} {ckpt_kind:<8} {r['expected_greedy_accuracy']:6.4f} "
                f"{p['belief']['error']:7.4f} {p['Q_star']['error']:7.4f} "
                f"{p['V_star']['error']:7.4f} {p['dV_nl']['error']:7.4f} {p['A_star']['error']:7.4f} "
                f"{vh['vs_V_star']['raw_r2']:9.4f} {vh['nonlinear_residual_vs_dV_nl']['corr']:10.4f}"
            )


if __name__ == "__main__":
    main()
