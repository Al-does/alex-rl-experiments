"""Moved factor-2 reward steering on the PR #225 persistent-nuisance v2 agents.

The frozen ctx32 transformers were trained with both factors paying for code
state 2. Here factor 2 pays for code state 0 instead; transitions, emissions
and actions are unchanged. The decoded factor-2 belief selects a factor action
from an action-value table (``myopic``: expected next-step reward;
``discounted``: QMDP with the training discount). The steering target is a
factor-2 point mass on the state where the frozen agent already takes that
action; factor 1 is kept as decoded. The update ``h' = h + lambda (y* - D(h)) E``
and the factored/joint maps follow the cycle-4 ``belief_causal_followup.py``
(revision 9ecb94542a2d0977e1e369040c956ca8d6353cbe) and the RLlib follow-up in
``two_factor_moved_reward_steering_2026_11``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import harness
from experiments.belief_quotient_history_jax_2026_11 import env as hmm
from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import (
    environment as v2,
)
from experiments.reward_factors_history_jax_2026_11.process import ACTION_PAIRS
from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process

STUDY_DIR = Path(__file__).resolve().parent
REPO = STUDY_DIR.parents[1]
SOURCE_STUDIES = (
    "reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11",
    "reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11",
)
SOURCE_REWARD_STATES = (2, 2)
# Factor 2 now pays for code state 0 (manuscript state 1) instead of 2.
MOVED_REWARD_STATES = (2, 0)
HORIZON = 1024
WARMUP = 64
FIT_ROWS, TEST_ROWS = 60_000, 80_000
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5, 2.0, 3.0)
TARGETS = ("marginal", "joint")
ACTION_MAPS = ("myopic", "discounted")
GAMMA = 0.99
FACTOR_TRANSITIONS = v2.action_transitions(3)  # (action, from, to)
ACTION_TABLE = np.asarray(ACTION_PAIRS, dtype=np.int32)
PAIR_INDEX = np.zeros((3, 3), dtype=np.int32)
for _index, (_first, _second) in enumerate(ACTION_PAIRS):
    PAIR_INDEX[_first, _second] = _index


# --- action maps -------------------------------------------------------------

def myopic_table(reward_state: int) -> np.ndarray:
    """``[state, action]`` probability that the next state is the reward state."""
    return FACTOR_TRANSITIONS[:, :, reward_state].T.copy()


def discounted_table(reward_state: int, gamma: float = GAMMA) -> np.ndarray:
    """``[state, action]`` optimal discounted Q for pre-transition occupancy reward."""
    reward = (np.arange(3) == reward_state).astype(np.float64)
    value = np.zeros(3)
    for _ in range(1_000_000):
        q = reward[:, None] + gamma * np.einsum("ast,t->sa", FACTOR_TRANSITIONS, value)
        if np.abs(q.max(1) - value).max() < 1e-13:
            return q
        value = q.max(1)
    raise RuntimeError("value iteration did not converge")


def action_table(kind: str, reward_state: int) -> np.ndarray:
    if kind == "myopic":
        return myopic_table(reward_state)
    if kind == "discounted":
        return discounted_table(reward_state)
    raise ValueError(f"unknown action map {kind!r}")


def stationary_occupancy(policy: tuple[int, int, int] | None = None) -> np.ndarray:
    """Factor occupancy under uniform random actions or a state-known policy."""
    if policy is None:
        matrix = FACTOR_TRANSITIONS.mean(0)
    else:
        matrix = FACTOR_TRANSITIONS[np.asarray(policy), np.arange(3)]
    values, vectors = np.linalg.eig(matrix.T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    return vector / vector.sum()


def best_factor_action(belief, table):
    return jnp.argmax(jnp.asarray(belief) @ jnp.asarray(table, dtype=jnp.float32), -1)


def factor_beliefs(joint_belief):
    joint = jnp.asarray(joint_belief).reshape(-1, 3, 3)
    return jnp.stack((joint.sum(2), joint.sum(1)), 1)


# --- processes and checkpoints ----------------------------------------------

def source_process(condition: str):
    return factor_process(condition, v2)


def moved_process():
    source = source_process("reward_both")
    first, second = np.divmod(np.arange(9), 3)
    components = np.stack([first == MOVED_REWARD_STATES[0],
                           second == MOVED_REWARD_STATES[1]]).astype(float)
    return hmm.make_process(*(np.asarray(a) for a in (
        source.initial, source.transitions, source.emission)), components.sum(0), components)


def load_params(path: Path, spec: models.ModelSpec):
    template = models.init_params(spec, jax.random.PRNGKey(0))
    with np.load(path) as data:
        stored = {name: data[name] for name in data.files}
    expected = {jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_leaves_with_path(template)}
    if expected != set(stored):
        raise ValueError(f"{path}: parameter names do not match the model spec")

    def restore(key, leaf):
        value = stored[jax.tree_util.keystr(key)]
        if value.shape != leaf.shape:
            raise ValueError(f"{path}: shape mismatch at {jax.tree_util.keystr(key)}")
        return jnp.asarray(value, dtype=leaf.dtype)

    return jax.tree_util.tree_map_with_path(restore, template)


# --- transformer forward with residual capture and steering ------------------

def forward(spec, params, carry, obs, t, site=None, shift=None):
    """``models.step`` for the transformer, returning all five sites.

    Sites 0-3 are the residual after each block (site 3 is the pre-final-norm
    residual probed in PR #225); site 4 is the post-final-norm embedding. When
    ``site`` is given, ``shift(x)`` is added there, so later blocks' keys and
    values at this step see the steered residual.
    """
    if spec.kind != "transformer":
        raise ValueError("only the transformer is supported")
    batch = obs.shape[0]
    cache_len = spec.context_len + 1
    slot = t % cache_len
    valid = jnp.arange(cache_len) <= t
    x = models._dense(params["input"], obs)
    sites, keys, values = [], [], []
    for layer, block in enumerate(params["blocks"]):
        q, k, v = jnp.split(models._dense(block["qkv"], models._layer_norm(block["ln1"], x)), 3, -1)
        heads = (batch, spec.n_heads, spec.head_dim)
        q, k, v = models._rope(q.reshape(heads), t), models._rope(k.reshape(heads), t), v.reshape(heads)
        k_cache = carry["k"][layer].at[:, slot].set(k)
        v_cache = carry["v"][layer].at[:, slot].set(v)
        scores = jnp.einsum("bhd,bshd->bhs", q, k_cache) / math.sqrt(spec.head_dim)
        weights = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), -1)
        attended = jnp.einsum("bhs,bshd->bhd", weights, v_cache).reshape(batch, spec.width)
        x = x + models._dense(block["proj"], attended)
        x = x + models._mlp(block["mlp"], models._layer_norm(block["ln2"], x))
        if site == layer:
            x = x + shift(x)
        sites.append(x)
        keys.append(k_cache)
        values.append(v_cache)
    embedding = models._layer_norm(params["final_norm"], x)
    if site == len(params["blocks"]):
        embedding = embedding + shift(embedding)
    sites.append(embedding)
    logits = models._dense(params["policy"], embedding)
    return logits, jnp.stack(sites, 1), {"k": jnp.stack(keys), "v": jnp.stack(values)}


def decoded_factor_beliefs(decoded, target: str):
    if target == "marginal":
        pairs = decoded.reshape(-1, 2, 2)
        return jnp.concatenate((pairs, 1 - pairs.sum(2, keepdims=True)), 2)
    if target == "joint":
        return factor_beliefs(jnp.concatenate((decoded, 1 - decoded.sum(1, keepdims=True)), 1))
    raise ValueError(f"unknown target {target!r}")


def belief_target(decoded, target: str, table, anchor):
    """Steering target: decoded factor 1, factor-2 point mass on the anchor."""
    factors = decoded_factor_beliefs(decoded, target)
    point = jax.nn.one_hot(jnp.asarray(anchor)[best_factor_action(factors[:, 1], table)], 3)
    if target == "marginal":
        return decoded.at[:, 2:].set(point[:, :2])
    first = factors[:, 0]
    return (first[:, :, None] * point[:, None, :]).reshape(-1, 9)[:, :8]


def steering_shift(x, steer, target: str):
    decoded = x @ steer["decoder"] + steer["bias"]
    shift = (belief_target(decoded, target, steer["table"], steer["anchor"]) - decoded) @ steer["embedding"]
    null = jnp.linalg.norm(shift, axis=1, keepdims=True) * steer["null_direction"]
    return steer["strength"] * jnp.where(steer["use_null"], null, shift)


# --- rollouts ----------------------------------------------------------------

def _keys(seed: int):
    return jax.random.split(jax.random.PRNGKey(seed), 3)


@partial(jax.jit, static_argnums=(0, 3, 4))
def collect(spec, params, process, episodes: int, horizon: int, key):
    """Unsteered greedy rollouts recording all sites and exact joint beliefs."""
    reset_key, step_key, _ = jax.random.split(key, 3)
    env_state = jax.vmap(hmm.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    belief = jax.vmap(hmm.belief_reset, (None, 0))(process, env_state.token)

    def body(state, t):
        env_state, carry, belief = state
        obs = jax.vmap(hmm.observe, (None, 0))(process, env_state)
        logits, sites, carry = forward(spec, params, carry, obs, t)
        action = jnp.argmax(logits, -1)
        next_env, _, components = jax.vmap(hmm.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), env_state, action)
        record = {"sites": sites, "belief": belief, "action": action, "components": components}
        belief = jax.vmap(hmm.belief_update, (None, 0, 0, 0))(process, belief, action, next_env.token)
        return (next_env, carry, belief), record

    _, records = jax.lax.scan(body, (env_state, models.init_carry(spec, episodes), belief),
                              jnp.arange(horizon))
    return jax.tree_util.tree_map(lambda a: jnp.moveaxis(a, 0, 1), records)


def collect_rows(spec, params, process, *, rows: int, seed: int):
    episodes = -(-rows // (HORIZON - WARMUP))
    records = jax.device_get(collect(spec, params, process, episodes, HORIZON, _keys(seed)[0]))
    out = {}
    for name, value in records.items():
        value = np.asarray(value)[:, WARMUP:]
        out[name] = value.reshape(-1, *value.shape[2:])[:rows]
    return out


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7))
def _evaluate(spec, params, process, episodes, policy, site, target, steered, steer,
              policy_tables, myopic_tables, key):
    reset_key, step_key, action_key = jax.random.split(key, 3)
    env_state = jax.vmap(hmm.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    belief = jax.vmap(hmm.belief_reset, (None, 0))(process, env_state.token)
    pair_index = jnp.asarray(PAIR_INDEX)
    table = jnp.asarray(ACTION_TABLE)
    moved = jnp.asarray(MOVED_REWARD_STATES)
    zeros = jnp.zeros((episodes, 2))
    acc = {"reward": zeros, "source_reward": zeros, "expected_next": zeros,
           "agree": zeros, "changed": zeros}

    def body(state, t):
        env_state, carry, belief, acc = state
        obs = jax.vmap(hmm.observe, (None, 0))(process, env_state)
        base_logits, _, base_carry = forward(spec, params, carry, obs, t)
        base = jnp.argmax(base_logits, -1)
        factors = factor_beliefs(belief)
        mapped = pair_index[best_factor_action(factors[:, 0], policy_tables[0]),
                            best_factor_action(factors[:, 1], policy_tables[1])]
        if policy == "agent" and steered:
            logits, _, carry = forward(spec, params, carry, obs, t, site,
                                       partial(steering_shift, steer=steer, target=target))
            action = jnp.argmax(logits, -1)
        elif policy == "agent":
            action, carry = base, base_carry
        elif policy == "random":
            action = jax.random.randint(jax.random.fold_in(action_key, t), (episodes,), 0, 9)
            carry = base_carry
        elif policy == "map":
            action, carry = mapped, base_carry
        else:
            raise ValueError(policy)
        executed = table[action]
        latent = jnp.stack(jnp.divmod(env_state.state, 3), 1)
        next_value = jnp.stack([
            jnp.take_along_axis(factors[:, f] @ myopic_tables[f], executed[:, f:f + 1], 1)[:, 0]
            for f in range(2)], 1)
        acc = {
            "reward": acc["reward"] + (latent == moved),
            "source_reward": acc["source_reward"] + (latent == 2),
            "expected_next": acc["expected_next"] + next_value,
            "agree": acc["agree"] + (executed == table[mapped]),
            "changed": acc["changed"] + (executed != table[base]),
        }
        next_env, _, _ = jax.vmap(hmm.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), env_state, action)
        belief = jax.vmap(hmm.belief_update, (None, 0, 0, 0))(process, belief, action, next_env.token)
        return (next_env, carry, belief, acc), None

    init = (env_state, models.init_carry(spec, episodes), belief, acc)
    (_, _, _, acc), _ = jax.lax.scan(body, init, jnp.arange(HORIZON))
    return acc


def evaluate(spec, params, *, episodes: int, seed: int, policy: str = "agent",
             action_map: str = "myopic", site: int = 3, target: str = "joint",
             steer: dict | None = None):
    """Complete-episode per-step rewards under the moved reward.

    ``policy``: ``agent`` (steered when ``steer`` is given), ``random`` (uniform
    joint actions) or ``map`` (``action_map`` applied to the exact belief).
    Environment keys depend only on ``seed``, so conditions share streams.
    """
    if episodes < 2:
        raise ValueError("at least two episodes are required")
    policy_tables = jnp.asarray(np.stack([action_table(action_map, r) for r in MOVED_REWARD_STATES]),
                                dtype=jnp.float32)
    myopic_tables = jnp.asarray(np.stack([myopic_table(r) for r in MOVED_REWARD_STATES]),
                                dtype=jnp.float32)
    dummy = {"decoder": jnp.zeros((spec.width, 8)), "bias": jnp.zeros(8),
             "embedding": jnp.zeros((8, spec.width)), "table": jnp.zeros((3, 3)),
             "anchor": jnp.zeros(3, jnp.int32), "null_direction": jnp.zeros(spec.width),
             "strength": jnp.float32(0), "use_null": jnp.bool_(False)}
    acc = jax.device_get(_evaluate(
        spec, params, moved_process(), episodes, policy, site, target, steer is not None,
        steer if steer is not None else dummy, policy_tables, myopic_tables, _keys(seed)[0]))

    def stats(name):
        values = np.asarray(acc[name], dtype=np.float64) / HORIZON
        return values.mean(0).tolist(), (values.std(0, ddof=1) / math.sqrt(episodes)).tolist()

    out = {}
    for name, label in (("reward", "factor_reward"), ("source_reward", "source_reward_state_occupancy"),
                        ("expected_next", "expected_next_step_reward")):
        out[label], out[f"{label}_se"] = stats(name)
    out["factor_action_agrees_with_action_map"] = stats("agree")[0]
    out["factor_action_change_fraction"] = stats("changed")[0]
    return out


# --- maps --------------------------------------------------------------------

def affine_fit(features, targets):
    x = np.column_stack((features, np.ones(len(features))))
    fitted = np.linalg.lstsq(x, targets, rcond=1e-10)[0]
    return fitted[:-1], fitted[-1]


def r_squared(prediction, truth) -> float:
    return float(1 - np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean(0)) ** 2))


def map_targets(belief: np.ndarray) -> dict[str, np.ndarray]:
    factors = np.asarray(factor_beliefs(belief))
    return {"marginal": factors[:, :, :2].reshape(-1, 4), "joint": belief[:, :8]}


def learned_action_map(rows) -> dict[str, list]:
    """``[argmax exact-belief state, factor action]`` row-normalized fractions."""
    factors = np.asarray(factor_beliefs(rows["belief"]))
    actions = ACTION_TABLE[rows["action"]]
    out = {}
    for factor in range(2):
        counts = np.zeros((3, 3))
        np.add.at(counts, (factors[:, factor].argmax(1), actions[:, factor]), 1)
        out[f"factor_{factor + 1}"] = (counts / np.maximum(counts.sum(1, keepdims=True), 1)).tolist()
    return out


def anchor_states(factor_map: np.ndarray) -> np.ndarray:
    """For each factor action, the state where the agent takes it most often."""
    return np.asarray(factor_map).argmax(0).astype(np.int32)


def null_direction(decoder: np.ndarray, seed: int) -> np.ndarray:
    direction = np.random.default_rng(seed).standard_normal(decoder.shape[0])
    direction -= decoder @ np.linalg.lstsq(decoder, direction, rcond=1e-10)[0]
    return direction / np.linalg.norm(direction)


# --- driver ------------------------------------------------------------------

def _git(path: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def source_runs() -> list[dict]:
    runs = []
    for study in SOURCE_STUDIES:
        for manifest in sorted((REPO / "experiments" / study).glob("*/results/*/run_manifest.json")):
            data = json.loads(manifest.read_text())
            leaf = manifest.parents[2].name
            runs.append({"study": study, "leaf": leaf, "condition": leaf.removesuffix("_transformer"),
                         "seed": data["runtime"]["seed"], "run_id": data["run_id"],
                         "results": manifest.parent,
                         "remote_prefix": data["remote_artifacts"]["prefix"]})
    return runs


def checkpoint_path(run: dict, checkpoints: Path) -> Path:
    tag = run["study"].removeprefix("reward_factors_persistent_nuisance_v2_jax_temp15_")
    matches = sorted(checkpoints.glob(f"{tag}__{run['leaf']}__seed{run['seed']}__{run['run_id']}__step_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one final checkpoint for {run['run_id']}")
    return matches[0]


def analyze(run: dict, checkpoint: Path, *, tune_episodes: int, confirm_episodes: int,
            seed: int, arrays: Path | None = None) -> dict:
    recipe = json.loads((run["results"] / "resolved_recipe.json").read_text())
    spec = models.ModelSpec(**recipe["model"])
    process = source_process(run["condition"])
    for name, value in process._asdict().items():
        if not np.allclose(np.asarray(value), np.asarray(recipe["process"][name]), atol=1e-6):
            raise ValueError(f"{run['run_id']}: recorded process {name} differs")
    params = load_params(checkpoint, spec)
    fit = collect_rows(spec, params, process, rows=FIT_ROWS, seed=seed)
    test = collect_rows(spec, params, process, rows=TEST_ROWS, seed=seed + 1)
    learned = learned_action_map(fit)
    anchor = anchor_states(np.asarray(learned["factor_2"]))
    fit_targets, test_targets = map_targets(fit["belief"]), map_targets(test["belief"])
    n_sites = spec.n_layers + 1
    maps, probes = {}, {}
    for site in range(n_sites):
        name = f"block_{site + 1}" if site < spec.n_layers else "post_final_norm"
        probes[name] = {}
        for target in TARGETS:
            features = fit["sites"][:, site].astype(np.float64)
            decoder, bias = affine_fit(features, fit_targets[target].astype(np.float64))
            embedding, _ = affine_fit(fit_targets[target].astype(np.float64), features)
            maps[site, target] = (decoder, bias, embedding)
            predicted = test["sites"][:, site] @ decoder + bias
            probes[name][target] = r_squared(predicted, test_targets[target])
    if arrays is not None:
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays, factor_belief=np.asarray(factor_beliefs(test["belief"])),
                            actions=test["action"])

    def steer(site, target, action_map, strength, use_null=False):
        decoder, bias, embedding = maps[site, target]
        return {"decoder": jnp.asarray(decoder, jnp.float32), "bias": jnp.asarray(bias, jnp.float32),
                "embedding": jnp.asarray(embedding, jnp.float32),
                "table": jnp.asarray(action_table(action_map, MOVED_REWARD_STATES[1]), jnp.float32),
                "anchor": jnp.asarray(anchor), "strength": jnp.float32(strength),
                "null_direction": jnp.asarray(null_direction(decoder, seed + 9999), jnp.float32),
                "use_null": jnp.bool_(use_null)}

    final_site = spec.n_layers - 1
    tune_seed, confirm_seed = seed + 1000, seed + 2000
    tuning = {"intact": evaluate(spec, params, episodes=tune_episodes, seed=tune_seed)}
    selected, sweeps = {}, {}
    for action_map in ACTION_MAPS:
        for target in TARGETS:
            sweep = {str(s): evaluate(spec, params, episodes=tune_episodes, seed=tune_seed,
                                      action_map=action_map, site=final_site, target=target,
                                      steer=steer(final_site, target, action_map, s))
                     for s in STRENGTHS}
            best = max(sweep[str(s)]["factor_reward"][1] for s in STRENGTHS)
            key = f"{action_map}_{target}"
            selected[key] = min(s for s in STRENGTHS if sweep[str(s)]["factor_reward"][1] == best)
            sweeps[key] = sweep
    confirm = partial(evaluate, spec, params, episodes=confirm_episodes, seed=confirm_seed)
    confirmation = {
        "intact": confirm(),
        "random": confirm(policy="random"),
        **{f"exact_belief_{m}_map": confirm(policy="map", action_map=m) for m in ACTION_MAPS},
        "comparisons": {}, "matched_null": {},
    }
    for action_map in ACTION_MAPS:
        for target in TARGETS:
            key = f"{action_map}_{target}"
            strength = selected[key]
            for site in range(n_sites):
                name = f"block_{site + 1}" if site < spec.n_layers else "post_final_norm"
                confirmation["comparisons"][f"{key}_{name}"] = {"strength": strength, **confirm(
                    action_map=action_map, site=site, target=target,
                    steer=steer(site, target, action_map, strength))}
            confirmation["matched_null"][key] = {"strength": strength, **confirm(
                action_map=action_map, site=final_site, target=target,
                steer=steer(final_site, target, action_map, strength, use_null=True))}
    return {
        "source": {key: str(value) if isinstance(value, Path) else value for key, value in run.items()},
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model": recipe["model"],
        "source_reward_states_code": list(SOURCE_REWARD_STATES),
        "moved_reward_states_code": list(MOVED_REWARD_STATES),
        "policy_mode": "greedy", "episode_horizon": HORIZON,
        "fit_seed": seed, "test_seed": seed + 1, "fit_rows": FIT_ROWS, "test_rows": TEST_ROWS,
        "warmup": WARMUP, "tune_seed": tune_seed, "tune_episodes": tune_episodes,
        "confirm_seed": confirm_seed, "confirm_episodes": confirm_episodes,
        "probe_r2": probes,
        "learned_action_map_rows_argmax_state_cols_noop_x_y": learned,
        "factor_2_anchor_state_by_action_noop_x_y": anchor.tolist(),
        "analytic_factor_occupancy": {
            "uniform_random": stationary_occupancy().tolist(),
            **{f"state_known_{m}_policy_reward_state_{r}": stationary_occupancy(
                tuple(action_table(m, r).argmax(1).tolist())).tolist()
               for m in ACTION_MAPS for r in (0, 2)}},
        "action_tables_factor_2": {m: action_table(m, MOVED_REWARD_STATES[1]).tolist() for m in ACTION_MAPS},
        "strength_tuning": {**tuning, "grid": sweeps, "selected_strength": selected,
                            "selection": "smallest grid strength maximizing tuning factor-2 reward"},
        "confirmation": confirmation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=STUDY_DIR / "artifacts" / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=STUDY_DIR / "results")
    parser.add_argument("--tune-episodes", type=int, default=64)
    parser.add_argument("--confirm-episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="substring filter on '<study>/<leaf>/seed<seed>'")
    args = parser.parse_args()
    revisions = {"experiment": _git(STUDY_DIR), "harness": _git(Path(harness.__file__).resolve().parent)}
    for run in source_runs():
        tag = f"{run['study']}/{run['leaf']}/seed{run['seed']}"
        if args.only and args.only not in tag:
            continue
        stem = f"{run['study'].removeprefix('reward_factors_persistent_nuisance_v2_jax_temp15_')}__{run['condition']}__seed{run['seed']}"
        output = args.output_dir / f"{stem}.json"
        if output.exists():
            raise FileExistsError(output)
        report = analyze(run, checkpoint_path(run, args.checkpoints),
                         tune_episodes=args.tune_episodes, confirm_episodes=args.confirm_episodes,
                         seed=args.seed, arrays=STUDY_DIR / "artifacts" / f"{stem}_heldout.npz")
        report["revisions"] = revisions
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1) + "\n")
        print(tag, "done", flush=True)


if __name__ == "__main__":
    main()
