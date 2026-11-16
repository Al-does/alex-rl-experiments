"""Factor-2 suppression on the fifteen PR #234 Joint Reward agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import harness
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11 import steering as base
from experiments.two_factor_coupled_moving_reward_steering_eps02_jax_2026_11 import steering as source

STUDY_DIR = Path(__file__).resolve().parent
TARGETS = ("factored", "joint")


def suppression_target(decoded, target):
    factors = base.decoded_factor_beliefs(decoded, "marginal" if target == "factored" else target)
    point = jax.nn.one_hot(jnp.argmin(factors[:, 1, :2], -1), 3)
    if target == "factored":
        return decoded.at[:, 2:].set(point[:, :2])
    if target == "joint":
        return (factors[:, 0, :, None] * point[:, None, :]).reshape(-1, 9)[:, :8]
    raise ValueError(target)


def steering_shift(x, steer, target):
    decoded = x @ steer["decoder"] + steer["bias"]
    delta = (suppression_target(decoded, target) - decoded) @ steer["embedding"]
    null = jnp.linalg.norm(delta, axis=-1, keepdims=True) * steer["null_direction"]
    return steer["strength"] * jnp.where(steer["use_null"], null, delta)


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
def rollout(spec, params, process, episodes, policy, site, target, steer, seed):
    reset_key, step_key, action_key = jax.random.split(base._keys(seed)[0], 3)
    env = jax.vmap(base.hmm.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    zero = jnp.zeros((episodes, 2))

    def body(state, t):
        env, carry, reward, changed = state
        obs = jax.vmap(base.hmm.observe, (None, 0))(process, env)
        logits, _, intact_carry = base.forward(spec, params, carry, obs, t)
        intact = jnp.argmax(logits, -1)
        if policy == "steered":
            logits, _, carry = base.forward(spec, params, carry, obs, t, site,
                                            partial(steering_shift, steer=steer, target=target))
            action = jnp.argmax(logits, -1)
        else:
            carry = intact_carry
            action = (jax.random.randint(jax.random.fold_in(action_key, t), (episodes,), 0, 9)
                      if policy == "random" else intact)
        changed += jnp.asarray(base.ACTION_TABLE)[action] != jnp.asarray(base.ACTION_TABLE)[intact]
        env, _, components = jax.vmap(base.hmm.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), env, action)
        return (env, carry, reward + components, changed), None

    (_, _, reward, changed), _ = jax.lax.scan(
        body, (env, base.models.init_carry(spec, episodes), zero, zero), jnp.arange(base.HORIZON))
    return reward / base.HORIZON, changed / base.HORIZON


def evaluate(spec, params, process, *, episodes, seed, policy="intact", site=3,
             target="joint", steer=None):
    if episodes < 2:
        raise ValueError("at least two episodes are required for episode SEs")
    reward, changed = jax.device_get(rollout(spec, params, process, episodes, policy, site,
                                          target, steer, jnp.int32(seed)))
    reward = np.asarray(reward, dtype=np.float64)
    total = reward.sum(1)
    return {"factor_reward": reward.mean(0).tolist(),
            "factor_reward_se": (reward.std(0, ddof=1) / math.sqrt(episodes)).tolist(),
            "total_reward": float(total.mean()),
            "total_reward_se": float(total.std(ddof=1) / math.sqrt(episodes)),
            "factor_action_change_fraction": np.asarray(changed).mean(0).tolist()}


def select_strength(sweep, random_reward):
    eligible = [s for s in base.STRENGTHS if sweep[str(s)]["factor_reward"][1] <= random_reward]
    selected = min(eligible) if eligible else min(
        base.STRENGTHS, key=lambda s: sweep[str(s)]["factor_reward"][1])
    return {"strength": selected, "reached_random_baseline": bool(eligible)}


def analyze(run, checkpoint, *, fit_rows=base.FIT_ROWS, test_rows=base.TEST_ROWS,
            tune_episodes=64, confirm_episodes=256):
    recipe = json.loads((base.REPO / run["results"] / "resolved_recipe.json").read_text())
    spec = base.models.ModelSpec(**recipe["model"])
    process = source.source_process()
    for name, value in process._asdict().items():
        if not np.allclose(value, recipe["process"][name], atol=1e-6):
            raise ValueError(f"recorded process differs: {name}")
    params = base.load_params(checkpoint, spec)
    fit = base.collect_rows(spec, params, process, rows=fit_rows, seed=42)
    test = base.collect_rows(spec, params, process, rows=test_rows, seed=43)
    fit_y, test_y = base.map_targets(fit["belief"]), base.map_targets(test["belief"])
    maps, probes = {}, {}
    sites = [f"block_{i + 1}" for i in range(spec.n_layers)] + ["post_final_norm"]
    for site, name in enumerate(sites):
        probes[name] = {}
        for target in TARGETS:
            key = "marginal" if target == "factored" else target
            features = fit["sites"][:, site].astype(np.float64)
            decoder, bias = base.affine_fit(features, fit_y[key].astype(np.float64))
            embedding, _ = base.affine_fit(fit_y[key].astype(np.float64), features)
            maps[site, target] = dict(decoder=jnp.asarray(decoder, jnp.float32),
                bias=jnp.asarray(bias, jnp.float32), embedding=jnp.asarray(embedding, jnp.float32),
                null_direction=jnp.asarray(base.null_direction(decoder, 10041), jnp.float32))
            probes[name][target] = base.r_squared(test["sites"][:, site] @ decoder + bias, test_y[key])
    del fit, test

    def evaluate_at(site, target, strength, episodes, seed, use_null=False):
        return evaluate(spec, params, process, episodes=episodes, seed=seed, policy="steered",
                        site=site, target=target, steer={**maps[site, target],
                        "strength": jnp.float32(strength), "use_null": jnp.bool_(use_null)})

    final = spec.n_layers - 1
    tuning = {policy: evaluate(spec, params, process, episodes=tune_episodes, seed=1042,
                              policy=policy) for policy in ("intact", "random")}
    sweeps, selected = {}, {}
    for target in TARGETS:
        sweeps[target] = {str(s): evaluate_at(final, target, s, tune_episodes, 1042)
                          for s in base.STRENGTHS}
        selected[target] = select_strength(sweeps[target], tuning["random"]["factor_reward"][1])
    confirmation = {policy: evaluate(spec, params, process, episodes=confirm_episodes, seed=2042,
                                   policy=policy) for policy in ("intact", "random")}
    for target in TARGETS:
        strength = selected[target]["strength"]
        for site, name in enumerate(sites):
            confirmation[f"{target}_{name}"] = evaluate_at(site, target, strength, confirm_episodes, 2042)
        confirmation[f"{target}_matched_null_block_{final + 1}"] = evaluate_at(
            final, target, strength, confirm_episodes, 2042, True)
    return {"source": {k: str(v) if isinstance(v, Path) else v for k, v in run.items()},
            "checkpoint": checkpoint.name, "checkpoint_step": source.FINAL_STEP,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "environment": source.environment.environment_metadata(), "model": recipe["model"],
            "policy_mode": "greedy", "episode_horizon": base.HORIZON,
            "reward_states_code": [2, 2], "fit_seed": 42, "test_seed": 43,
            "fit_rows": fit_rows, "test_rows": test_rows, "map_warmup_dropped": base.WARMUP,
            "evaluation_warmup_dropped": 0, "tune_seed": 1042, "confirm_seed": 2042,
            "tune_episodes": tune_episodes, "confirm_episodes": confirm_episodes,
            "matmul_precision": "highest", "map_solver": "numpy.linalg.lstsq float64 rcond=1e-10",
            "action_change_reference": "intact forward on the same steered history and carry",
            "probe_r2": probes, "strength_tuning": {**tuning, "grid": sweeps,
                "selected": selected}, "confirmation": confirmation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-seeds", type=int, nargs="+", default=list(range(42, 57)))
    parser.add_argument("--fit-rows", type=int, default=base.FIT_ROWS)
    parser.add_argument("--test-rows", type=int, default=base.TEST_ROWS)
    parser.add_argument("--tune-episodes", type=int, default=64)
    parser.add_argument("--confirm-episodes", type=int, default=256)
    parser.add_argument("--checkpoints", type=Path, default=STUDY_DIR / "artifacts/checkpoints")
    parser.add_argument("--output-dir", type=Path, default=STUDY_DIR / "results")
    args = parser.parse_args()
    if not set(args.agent_seeds) <= set(range(42, 57)) or min(args.fit_rows, args.test_rows) < 2:
        parser.error("agent seeds must be 42–56 and row budgets at least two")
    if min(args.tune_episodes, args.confirm_episodes) < 2:
        parser.error("episode budgets must be at least two")
    revisions = {"experiment": base._git(STUDY_DIR),
                 "harness": base._git(Path(harness.__file__).resolve().parent)}
    with jax.default_matmul_precision("highest"):
        for run in source.source_runs():
            if run["seed"] not in args.agent_seeds:
                continue
            output = args.output_dir / f"eps02_factor2_suppression__seed{run['seed']}.json"
            if output.exists():
                raise FileExistsError(output)
            start = time.monotonic()
            report = analyze(run, source.fetch_checkpoint(run, args.checkpoints),
                             fit_rows=args.fit_rows, test_rows=args.test_rows,
                             tune_episodes=args.tune_episodes, confirm_episodes=args.confirm_episodes)
            report.update(revisions=revisions, wall_seconds=time.monotonic() - start,
                          jax_backend=jax.default_backend())
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"seed {run['seed']} done in {report['wall_seconds']:.1f}s: {output}", flush=True)


if __name__ == "__main__":
    main()
