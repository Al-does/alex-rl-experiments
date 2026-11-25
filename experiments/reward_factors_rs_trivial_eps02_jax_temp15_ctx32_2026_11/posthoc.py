"""Post-hoc geometry of trained checkpoints (CPU; no training).

For each (arm, seed) checkpoint this measures, on fresh rollouts of the
trained policy (temperature 1.5 sampling, the training behaviour, and greedy):

- held-out affine probes of rho_f, delta_f and V* from the pre-final-LayerNorm
  residual, plus finite-window references: the same probes fit to the exact
  filter restarted from the uniform prior on the last 33 decisions (one
  attention layer's band, keys i-32..i) and on the last 129 decisions (the
  4-layer stack's full receptive field), and to the one-hot window of the last
  33 observations (linear-in-input reference). These are affine readouts of
  uniform-prior window filters, not certified ceilings;
- value readouts: the critic head (shared/separate) against V* and its affine
  dependence on (rho_1, rho_2);
- control geometry: how much of the tempered log-policy is affinely explained
  by rho_1, rho_2 and both; and residual steering along each target's
  encoding direction (the partial regression of the residual on
  [rho_1, rho_2, delta_1, delta_2], scaled to one visited standard deviation),
  reported as the fraction of greedy factor-1 / factor-2 actions that change
  and the mean total-variation change of the tempered policy (a random
  direction of the same norm is the control, and each direction is also
  applied rescaled to the rho_1 displacement norm).

All are descriptive linear-accessibility / local-sensitivity measurements, not
proof of causal use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from analysis.probes.linear import fit_affine_probe, probe_predict, r2_score
from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.reinforce import rollout
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo
from experiments.reward_factors_history_jax_2026_11 import shared as source
from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11 import algorithms, design
from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.process import (
    ACTION_PAIRS, make_targets, rs_trivial_process,
)
from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11.shared import (
    ARCHITECTURE, CONTEXT_LEN, STUDY,
)

TEMPERATURE = 1.5
RIDGE = 1e-6
WARMUP = 64
LAYER_WINDOW = CONTEXT_LEN + 1  # banded attention: query i sees keys i-32..i
RECEPTIVE_FIELD = 4 * CONTEXT_LEN + 1  # four stacked layers
PROBE_TARGETS = ("rho_1", "rho_2", "delta_1", "delta_2", "value_star")
FACTOR_ACTION = np.asarray(ACTION_PAIRS)  # (9, 2): joint action -> per-factor action


def spec_and_process():
    process = rs_trivial_process()
    from dataclasses import replace
    return replace(source.model_spec(ARCHITECTURE, process), context_len=CONTEXT_LEN), process


def template(arm, spec):
    key = jax.random.PRNGKey(0)
    if arm == "actor_only":
        return models.init_params(spec, key)
    if arm == "shared_critic":
        return ppo.init_params(spec, key)
    return algorithms.separate_init(spec, key)


def load_params(arm, spec, path):
    saved = np.load(path)
    leaves = jax.tree_util.tree_leaves_with_path(template(arm, spec))
    values = [jnp.asarray(saved[jax.tree_util.keystr(k)]) for k, _ in leaves]
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template(arm, spec)), values)


def window_beliefs(process, tokens, prev_actions, context_len=CONTEXT_LEN):
    """Exact joint filter restarted from the uniform prior on the last ``context_len`` steps."""
    transitions, emission = np.asarray(process.transitions), np.asarray(process.emission)
    n_ep, length = tokens.shape
    t = np.arange(length)
    start = np.maximum(0, t - context_len + 1)
    belief = emission[:, tokens[:, start]].transpose(1, 2, 0) / 9
    belief /= belief.sum(-1, keepdims=True)
    for j in range(1, context_len):
        k = np.minimum(start + j, length - 1)
        live = (start + j <= t)[None, :, None]
        moved = np.einsum("ets,etsu->etu", belief, transitions[prev_actions[:, k]])
        moved = moved * emission[:, tokens[:, k]].transpose(1, 2, 0)
        moved /= moved.sum(-1, keepdims=True)
        belief = np.where(live, moved, belief)
    return belief


def obs_windows(obs, context_len=CONTEXT_LEN):
    n_ep, length, dim = obs.shape
    padded = np.concatenate([np.zeros((n_ep, context_len - 1, dim), obs.dtype), obs], 1)
    return np.stack([padded[:, j:j + length] for j in range(context_len)], 2).reshape(n_ep, length, -1)


def _affine_r2(train_x, train_y, test_x, test_y):
    weight, bias = fit_affine_probe(train_x, train_y, ridge=RIDGE)
    return float(r2_score(probe_predict(weight, bias, test_x), test_y))


def collect(spec, process, params, key, episodes, greedy, value_fn):
    records = jax.jit(lambda p, k: rollout(spec, process, p, k, episodes, design.EPISODE_LENGTH,
                                           greedy=greedy, track_belief=True,
                                           temperature=TEMPERATURE))(params, key)
    records = jax.device_get(records)
    obs = np.asarray(records["obs"])
    tokens = obs[..., :9].argmax(-1)
    prev = obs[..., 9:]
    prev_actions = np.where(prev.sum(-1) > 0, prev.argmax(-1), 0)
    out = {"residual": records["residual"], "belief": records["belief"], "action": records["action"],
           "reward": records["reward"], "components": records["components"],
           "window_33": window_beliefs(process, tokens, prev_actions, LAYER_WINDOW),
           "window_129": window_beliefs(process, tokens, prev_actions, RECEPTIVE_FIELD)}
    if value_fn is not None:
        out["value_head"] = np.asarray(value_fn(params, records["obs"]))
    sl = slice(WARMUP, None)
    out = {k: np.asarray(v)[:, sl].reshape(-1, *np.asarray(v).shape[2:]) for k, v in out.items()}
    out["obs_window"] = obs_windows(obs, LAYER_WINDOW)[:, sl].reshape(-1, obs.shape[-1] * LAYER_WINDOW)
    return out


def value_function(arm, spec):
    if arm == "shared_critic":
        def fn(params, obs):
            _, _, embedding = models.sequence(spec, params, obs)
            return models._dense(params["value"], embedding)[..., 0]
        return jax.jit(fn)
    if arm == "separate_critic":
        return jax.jit(lambda params, obs: algorithms.critic_values(spec, params, obs))
    return None


def logits_from_residual(params, residual):
    return np.asarray(models._head(params, jnp.asarray(residual))[0])


def tempered(logits):
    z = logits / TEMPERATURE
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def analyse(arm, spec, process, params, *, episodes, seed, greedy):
    targets_fn = make_targets(design.value_star)
    key = jax.random.PRNGKey(10_000 + seed)
    vf = value_function(arm, spec)
    train = collect(spec, process, params, jax.random.fold_in(key, 0), episodes, greedy, vf)
    test = collect(spec, process, params, jax.random.fold_in(key, 1), episodes, greedy, vf)
    t_tr, t_te = targets_fn(train["belief"]), targets_fn(test["belief"])
    factor_actions = FACTOR_ACTION[test["action"]]
    report = {
        "policy_mode": "greedy" if greedy else f"sampled_temperature_{TEMPERATURE}",
        "rows_per_split": int(len(test["action"])),
        "reward_per_step": float(test["reward"].mean()),
        "reward_components_per_step": test["components"].mean(0).tolist(),
        "factor_1_action_fractions": np.bincount(factor_actions[:, 0], minlength=3).__truediv__(len(factor_actions)).tolist(),
        "factor_2_action_fractions": np.bincount(factor_actions[:, 1], minlength=3).__truediv__(len(factor_actions)).tolist(),
        "probes": {},
    }
    feats = {"residual": "residual", "window_33_filter": "window_33",
             "window_129_filter": "window_129", "window_33_obs_linear": "obs_window"}
    for name in PROBE_TARGETS:
        report["probes"][name] = {
            label: _affine_r2(train[f], t_tr[name], test[f], t_te[name]) for label, f in feats.items()}
    if vf is not None:
        rho = lambda t: np.concatenate([t["rho_1"], t["rho_2"]], 1)
        v_tr, v_te = train["value_head"][:, None], test["value_head"][:, None]
        w, b = fit_affine_probe(rho(t_tr), v_tr, ridge=RIDGE)
        w_star, _ = fit_affine_probe(rho(t_tr), t_tr["value_star"], ridge=RIDGE)
        report["value_head"] = {
            "r_squared_vs_value_star": float(r2_score(v_te, t_te["value_star"])),
            "affine_r_squared_vs_value_star": _affine_r2(t_tr["value_star"], v_tr, t_te["value_star"], v_te),
            "affine_r_squared_from_rho_1": _affine_r2(t_tr["rho_1"], v_tr, t_te["rho_1"], v_te),
            "affine_r_squared_from_rho_1_rho_2": _affine_r2(rho(t_tr), v_tr, rho(t_te), v_te),
            "coefficients_rho_1_rho_2": np.asarray(w)[:, 0].tolist(),
            "value_star_coefficients_rho_1_rho_2": np.asarray(w_star)[:, 0].tolist(),
        }
    logp = np.log(tempered(logits_from_residual(params, test["residual"])) + 1e-12)
    logp_tr = np.log(tempered(logits_from_residual(params, train["residual"])) + 1e-12)
    controls = {"rho_1": ["rho_1"], "rho_2": ["rho_2"], "rho_1_rho_2": ["rho_1", "rho_2"],
                "rho_delta_both_factors": ["rho_1", "rho_2", "delta_1", "delta_2"]}
    report["log_policy_affine_r_squared"] = {
        label: _affine_r2(np.concatenate([t_tr[n] for n in names], 1), logp_tr,
                          np.concatenate([t_te[n] for n in names], 1), logp)
        for label, names in controls.items()}
    base_logits = logits_from_residual(params, test["residual"])
    base_p, base_a = tempered(base_logits), FACTOR_ACTION[base_logits.argmax(-1)]
    rng = np.random.default_rng(seed)
    steering = {}
    names = ("rho_1", "rho_2", "delta_1", "delta_2")
    design_matrix = np.concatenate([t_tr[n] for n in names] + [np.ones((len(t_tr["rho_1"]), 1))], 1)
    encoding = np.linalg.lstsq(design_matrix, train["residual"], rcond=None)[0]
    for index, name in enumerate(names):
        displacement = encoding[index] * float(t_tr[name].std())
        w = displacement
        random = rng.standard_normal(w.shape)
        random *= np.linalg.norm(displacement) / np.linalg.norm(random)
        entry = {"displacement_norm_over_mean_residual_norm":
                 float(np.linalg.norm(displacement) / np.linalg.norm(test["residual"], axis=1).mean())}
        matched = displacement * np.linalg.norm(encoding[0] * float(t_tr["rho_1"].std())) / np.linalg.norm(displacement)
        for label, d in (("probe", displacement), ("random", random), ("probe_norm_matched_to_rho_1", matched)):
            for sign in (1, -1):
                logits = logits_from_residual(params, test["residual"] + sign * d)
                acts = FACTOR_ACTION[logits.argmax(-1)]
                entry[f"{label}_{'plus' if sign > 0 else 'minus'}"] = {
                    "factor_1_greedy_change": float((acts[:, 0] != base_a[:, 0]).mean()),
                    "factor_2_greedy_change": float((acts[:, 1] != base_a[:, 1]).mean()),
                    "mean_tv_tempered_policy": float(0.5 * np.abs(tempered(logits) - base_p).sum(-1).mean()),
                }
        steering[name] = entry
    report["steering_one_visited_std"] = steering
    return report


def fetch_checkpoint(arm, seed, step, cache):
    import boto3
    run = f"{arm}-seed{seed}"
    key = f"experiments/{STUDY}/{arm}_transformer/{run}/checkpoints/step_{step:010d}.npz"
    path = Path(cache) / run / f"step_{step:010d}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        from harness.storage.b2 import normalize_b2_endpoint
        boto3.client("s3", endpoint_url=normalize_b2_endpoint(os.environ["B2_ENDPOINT"]),
                     aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
                     aws_secret_access_key=os.environ["B2_APPLICATION_KEY"]).download_file(
            os.environ["B2_BUCKET"], key, str(path))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(algorithms.ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--cache", default="artifacts/posthoc_checkpoints")
    parser.add_argument("--checkpoint", help="local checkpoint (single arm/seed/step)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.checkpoint and len(args.arms) * len(args.seeds) * len(args.steps) != 1:
        parser.error("--checkpoint requires exactly one --arms, --seeds and --steps value")
    spec, process = spec_and_process()
    out = Path(args.output)
    reports = json.loads(out.read_text()) if out.exists() else {}
    for arm in args.arms:
        for seed in args.seeds:
            for step in args.steps:
                name = f"{arm}/seed{seed}/step{step}"
                if name in reports:
                    continue
                path = args.checkpoint or fetch_checkpoint(arm, seed, step, args.cache)
                params = load_params(arm, spec, path)
                reports[name] = {mode: analyse(arm, spec, process, params, episodes=args.episodes,
                                               seed=seed, greedy=mode == "greedy")
                                 for mode in ("sampled", "greedy")}
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(reports, indent=1))
                print(name, json.dumps({m: {"reward": r["reward_per_step"],
                                            "rho_2": r["probes"]["rho_2"]["residual"]}
                                        for m, r in reports[name].items()}), flush=True)


if __name__ == "__main__":
    main()
