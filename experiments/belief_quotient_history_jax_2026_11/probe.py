"""Held-out affine belief probes on JAX greedy rollouts.

Fit and test rows come from independent streams of fresh greedy episodes. The
first ``warmup`` decisions of every episode are excluded. Targets are exact
filtered beliefs over the current hidden state given tokens and executed
actions; fitting reuses the harness's NumPy ridge probe.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np

from analysis.probes.linear import (
    fit_affine_probe,
    global_mse_metrics,
    probe_predict,
    r2_score,
)
from experiments.belief_quotient_history_jax_2026_11.reinforce import rollout

PROBE_RIDGE = 1e-6


def probe_budget(smoke: bool) -> dict:
    if smoke:
        return {"train_rows": 4_096, "test_rows": 4_096, "warmup": 4}
    return {"train_rows": 60_000, "test_rows": 80_000, "warmup": 64}


@partial(jax.jit, static_argnums=(0, 4, 5))
def _probe_rollout(spec, process, params, key, num_envs, episode_length):
    return rollout(spec, process, params, key, num_envs, episode_length,
                   greedy=True, track_belief=True)


def collect(spec, process, params, key, *, rows, warmup, episode_length):
    usable = episode_length - warmup
    n_episodes = -(-rows // usable)
    records = _probe_rollout(spec, process, params, key, n_episodes, episode_length)
    names = ("belief", "residual", "embedding", "action", "reward", "components", "state")
    out = {}
    for name in names:
        value = np.asarray(records[name])[:, warmup:]
        out[name] = value.reshape(-1, *value.shape[2:])[:rows]
    return out


def probe_checkpoint(spec, process, params, key, *, smoke, episode_length,
                     representation, targets):
    """``targets(belief) -> {name: target_rows}`` maps joint beliefs to probes."""

    budget = probe_budget(smoke)
    train_key, test_key = jax.random.split(key)
    kwargs = {"warmup": budget["warmup"], "episode_length": episode_length}
    train = collect(spec, process, params, train_key, rows=budget["train_rows"], **kwargs)
    test = collect(spec, process, params, test_key, rows=budget["test_rows"], **kwargs)
    metrics = {
        "probe": "held_out_affine_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "representation": representation,
        "policy_mode": "greedy",
        **budget,
        "policy": {
            "reward_per_step": float(test["reward"].mean()),
            "reward_components_per_step": test["components"].mean(0).tolist(),
            "action_fractions": np.bincount(
                test["action"], minlength=spec.n_actions
            ).astype(float).__truediv__(len(test["action"])).tolist(),
        },
        "targets": {},
    }
    train_targets, test_targets = targets(train["belief"]), targets(test["belief"])
    for name, train_target in train_targets.items():
        weight, bias = fit_affine_probe(train[representation], train_target, ridge=PROBE_RIDGE)
        predicted = probe_predict(weight, bias, test[representation])
        metrics["targets"][name] = {
            **global_mse_metrics(predicted, test_targets[name]),
            "r_squared": r2_score(predicted, test_targets[name]),
        }
    return metrics
