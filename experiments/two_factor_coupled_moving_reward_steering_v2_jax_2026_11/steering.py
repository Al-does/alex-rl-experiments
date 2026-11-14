"""XOR and moving-reward belief steering on the PR #225 Joint Reward agents.

Two reward mappings on the unchanged v2 dynamics, chosen so that the best
factor-2 action is not a function of the factor-2 belief alone:

* ``xor``: reward 1 when exactly one factor is in code state 2 (manuscript
  state 3). The best factor-2 action depends on factor 1.
* ``moving``: factor 1 pays for state 3 as in training; factor 2 pays for
  state 3 during even ``PERIOD``-step phases and for states 1/2 during odd
  phases. The best factor-2 action depends on time.

Action maps are time-indexed tables ``Q_t[joint state, joint action]`` applied
to a joint belief: ``myopic`` (expected next-step reward) or ``discounted``
(finite-horizon QMDP, gamma=0.99). Steering decodes the belief, picks the joint
action, and moves each steered factor to a point mass on the state where the
frozen agent most often takes that factor action. Unsteered factors keep their
decoded belief. Model, maps and update rule are those of
``two_factor_moved_reward_steering_v2_jax_2026_11``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import harness
from experiments.belief_quotient_history_jax_2026_11 import env as hmm
from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11 import steering as base

STUDY_DIR = Path(__file__).resolve().parent
REWARDS = ("xor", "moving")
STEERED_FACTORS = {"xor": (True, True), "moving": (False, True)}
PERIOD = 256
HORIZON = base.HORIZON
GAMMA = base.GAMMA
ACTION_MAPS = base.ACTION_MAPS
TARGETS = base.TARGETS
STRENGTHS = base.STRENGTHS
_FIRST, _SECOND = np.divmod(np.arange(9), 3)


def joint_transitions() -> np.ndarray:
    """``[joint action, from, to]`` in float64, rows renormalized."""
    transitions = np.asarray(base.source_process("reward_both").transitions, np.float64)
    return transitions / transitions.sum(2, keepdims=True)


def reward_components(reward: str, t: int) -> np.ndarray:
    """``[component, joint state]`` reward at step ``t``."""
    first, second = (_FIRST == 2).astype(float), (_SECOND == 2).astype(float)
    if reward == "xor":
        return (first != second).astype(float)[None]
    if reward == "moving":
        return np.stack((first, second if (t // PERIOD) % 2 == 0 else 1 - second))
    raise ValueError(f"unknown reward {reward!r}")


def reward_schedule(reward: str) -> np.ndarray:
    return np.stack([reward_components(reward, t) for t in range(HORIZON)])


def q_tables(reward: str, kind: str, gamma: float = GAMMA) -> np.ndarray:
    """``[t, joint state, joint action]`` action values over one episode.

    ``myopic``: expected reward of the next state. ``discounted``: optimal
    finite-horizon values for the pre-transition occupancy reward.
    """
    transitions = joint_transitions()
    total = reward_schedule(reward).sum(1)
    if kind == "myopic":
        upcoming = np.concatenate((total[1:], total[-1:]))
        return np.einsum("ast,nt->nsa", transitions, upcoming)
    if kind != "discounted":
        raise ValueError(f"unknown action map {kind!r}")
    q = np.zeros((HORIZON, 9, 9))
    value = np.zeros(9)
    for t in range(HORIZON - 1, -1, -1):
        q[t] = total[t][:, None] + gamma * np.einsum("ast,t->sa", transitions, value)
        value = q[t].max(1)
    return q


def decoded_joint(decoded, target: str):
    """Joint belief (9 entries) implied by a factored or joint decode."""
    if target == "joint":
        return jnp.concatenate((decoded, 1 - decoded.sum(1, keepdims=True)), 1)
    factors = base.decoded_factor_beliefs(decoded, "marginal")
    return (factors[:, 0, :, None] * factors[:, 1, None, :]).reshape(-1, 9)


def belief_target(decoded, target: str, q, anchors, steered: tuple[bool, bool]):
    """Steering target: steered factors on the anchor of the map's action."""
    factors = base.decoded_factor_beliefs(decoded, target)
    desired = jnp.asarray(base.ACTION_TABLE)[jnp.argmax(decoded_joint(decoded, target) @ q, -1)]
    points = [jax.nn.one_hot(anchors[f][desired[:, f]], 3) if steered[f] else factors[:, f]
              for f in range(2)]
    if target == "marginal":
        return jnp.concatenate((points[0][:, :2], points[1][:, :2]), 1)
    return (points[0][:, :, None] * points[1][:, None, :]).reshape(-1, 9)[:, :8]


def steering_shift(x, steer, target: str, steered, t):
    decoded = x @ steer["decoder"] + steer["bias"]
    goal = belief_target(decoded, target, steer["q"][t], steer["anchors"], steered)
    shift = (goal - decoded) @ steer["embedding"]
    null = jnp.linalg.norm(shift, axis=1, keepdims=True) * steer["null_direction"]
    return steer["strength"] * jnp.where(steer["use_null"], null, shift)


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8))
def _evaluate(spec, params, process, episodes, policy, site, target, steered, steer_flag,
              steer, q, schedule, key):
    reset_key, step_key, action_key = jax.random.split(key, 3)
    env_state = jax.vmap(hmm.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    belief = jax.vmap(hmm.belief_reset, (None, 0))(process, env_state.token)
    table = jnp.asarray(base.ACTION_TABLE)
    n_components = schedule.shape[1]
    acc = {"reward": jnp.zeros((episodes, 2, n_components)), "agree": jnp.zeros((episodes, 2)),
           "changed": jnp.zeros((episodes, 2))}

    def body(state, t):
        env_state, carry, belief, acc = state
        obs = jax.vmap(hmm.observe, (None, 0))(process, env_state)
        base_logits, _, base_carry = base.forward(spec, params, carry, obs, t)
        agent = jnp.argmax(base_logits, -1)
        mapped = jnp.argmax(belief @ q[t], -1)
        if policy == "agent" and steer_flag:
            logits, _, carry = base.forward(
                spec, params, carry, obs, t, site,
                partial(steering_shift, steer=steer, target=target, steered=steered, t=t))
            action = jnp.argmax(logits, -1)
        elif policy == "agent":
            action, carry = agent, base_carry
        elif policy == "random":
            action = jax.random.randint(jax.random.fold_in(action_key, t), (episodes,), 0, 9)
            carry = base_carry
        elif policy == "map":
            action, carry = mapped, base_carry
        else:
            raise ValueError(policy)
        executed = table[action]
        phase = (t // PERIOD) % 2
        earned = schedule[t][:, env_state.state].T
        acc = {"reward": acc["reward"].at[:, phase].add(earned),
               "agree": acc["agree"] + (executed == table[mapped]),
               "changed": acc["changed"] + (executed != table[agent])}
        next_env, _, _ = jax.vmap(hmm.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), env_state, action)
        belief = jax.vmap(hmm.belief_update, (None, 0, 0, 0))(process, belief, action, next_env.token)
        return (next_env, carry, belief, acc), None

    init = (env_state, models.init_carry(spec, episodes), belief, acc)
    (_, _, _, acc), _ = jax.lax.scan(body, init, jnp.arange(HORIZON))
    return acc


def evaluate(spec, params, *, reward: str, episodes: int, seed: int, policy: str = "agent",
             action_map: str = "discounted", site: int = 3, target: str = "joint",
             steer: dict | None = None):
    """Per-step reward (total, per component, per phase) over complete episodes."""
    if episodes < 2:
        raise ValueError("at least two episodes are required")
    q = jnp.asarray(q_tables(reward, action_map), jnp.float32)
    dummy = {"decoder": jnp.zeros((spec.width, 8)), "bias": jnp.zeros(8),
             "embedding": jnp.zeros((8, spec.width)), "q": q,
             "anchors": jnp.zeros((2, 3), jnp.int32), "null_direction": jnp.zeros(spec.width),
             "strength": jnp.float32(0), "use_null": jnp.bool_(False)}
    acc = jax.device_get(_evaluate(
        spec, params, base.source_process("reward_both"), episodes, policy, site, target,
        STEERED_FACTORS[reward], steer is not None, steer if steer is not None else dummy, q,
        jnp.asarray(reward_schedule(reward), jnp.float32), base._keys(seed)[0]))
    per_phase = np.asarray(acc["reward"], np.float64)  # (episodes, phase, component)

    def summary(values):
        return values.mean(0).tolist(), (values.std(0, ddof=1) / math.sqrt(episodes)).tolist()

    total = per_phase.sum((1, 2)) / HORIZON
    out = {}
    out["total_reward"], out["total_reward_se"] = (float(v) for v in summary(total))
    out["component_reward"], out["component_reward_se"] = summary(per_phase.sum(1) / HORIZON)
    out["phase_component_reward"], out["phase_component_reward_se"] = summary(
        per_phase / (HORIZON / 2))
    out["factor_action_agrees_with_exact_belief_map"] = (
        np.asarray(acc["agree"]).mean(0) / HORIZON).tolist()
    out["factor_action_change_fraction"] = (np.asarray(acc["changed"]).mean(0) / HORIZON).tolist()
    return out


def analyze(run: dict, checkpoint: Path, *, tune_episodes: int, confirm_episodes: int,
            seed: int) -> dict:
    recipe = json.loads((run["results"] / "resolved_recipe.json").read_text())
    spec = models.ModelSpec(**recipe["model"])
    process = base.source_process(run["condition"])
    params = base.load_params(checkpoint, spec)
    fit = base.collect_rows(spec, params, process, rows=base.FIT_ROWS, seed=seed)
    learned = base.learned_action_map(fit)
    anchors = np.stack([base.anchor_states(np.asarray(learned[f"factor_{f}"])) for f in (1, 2)])
    fit_targets = base.map_targets(fit["belief"])
    n_sites = spec.n_layers + 1
    maps = {}
    for site in range(n_sites):
        features = fit["sites"][:, site].astype(np.float64)
        for target in TARGETS:
            decoder, bias = base.affine_fit(features, fit_targets[target].astype(np.float64))
            embedding, _ = base.affine_fit(fit_targets[target].astype(np.float64), features)
            maps[site, target] = (decoder, bias, embedding)
    site_names = [f"block_{i + 1}" for i in range(spec.n_layers)] + ["post_final_norm"]
    final_site = spec.n_layers - 1
    tune_seed, confirm_seed = seed + 1000, seed + 2000
    rewards = {}
    for reward in REWARDS:
        def steer(site, target, action_map, strength, use_null=False):
            decoder, bias, embedding = maps[site, target]
            return {"decoder": jnp.asarray(decoder, jnp.float32),
                    "bias": jnp.asarray(bias, jnp.float32),
                    "embedding": jnp.asarray(embedding, jnp.float32),
                    "q": jnp.asarray(q_tables(reward, action_map), jnp.float32),
                    "anchors": jnp.asarray(anchors), "strength": jnp.float32(strength),
                    "null_direction": jnp.asarray(base.null_direction(decoder, seed + 9999), jnp.float32),
                    "use_null": jnp.bool_(use_null)}

        tune = partial(evaluate, spec, params, reward=reward, episodes=tune_episodes, seed=tune_seed)
        confirm = partial(evaluate, spec, params, reward=reward, episodes=confirm_episodes,
                          seed=confirm_seed)
        selected, sweeps = {}, {}
        for action_map in ACTION_MAPS:
            for target in TARGETS:
                sweep = {str(s): tune(action_map=action_map, site=final_site, target=target,
                                      steer=steer(final_site, target, action_map, s))
                         for s in STRENGTHS}
                best = max(sweep[str(s)]["total_reward"] for s in STRENGTHS)
                key = f"{action_map}_{target}"
                selected[key] = min(s for s in STRENGTHS if sweep[str(s)]["total_reward"] == best)
                sweeps[key] = sweep
        confirmation = {"intact": confirm(), "random": confirm(policy="random"),
                        **{f"exact_belief_{m}_map": confirm(policy="map", action_map=m)
                           for m in ACTION_MAPS},
                        "comparisons": {}, "matched_null": {}}
        for action_map in ACTION_MAPS:
            for target in TARGETS:
                key = f"{action_map}_{target}"
                strength = selected[key]
                for site in range(n_sites):
                    confirmation["comparisons"][f"{key}_{site_names[site]}"] = {
                        "strength": strength,
                        **confirm(action_map=action_map, site=site, target=target,
                                  steer=steer(site, target, action_map, strength))}
                confirmation["matched_null"][key] = {"strength": strength, **confirm(
                    action_map=action_map, site=final_site, target=target,
                    steer=steer(final_site, target, action_map, strength, use_null=True))}
        rewards[reward] = {"steered_factors": list(STEERED_FACTORS[reward]),
                           "strength_tuning": {"grid": sweeps, "selected_strength": selected},
                           "confirmation": confirmation}
        print(run["run_id"], reward, "done", flush=True)
    return {
        "source": {k: str(v) if isinstance(v, Path) else v for k, v in run.items()},
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model": recipe["model"], "policy_mode": "greedy", "episode_horizon": HORIZON,
        "moving_reward_period": PERIOD, "gamma": GAMMA,
        "fit_seed": seed, "fit_rows": base.FIT_ROWS, "warmup": base.WARMUP,
        "tune_seed": tune_seed, "tune_episodes": tune_episodes,
        "confirm_seed": confirm_seed, "confirm_episodes": confirm_episodes,
        "anchor_state_by_factor_action_noop_x_y": anchors.tolist(),
        "selection": "smallest grid strength maximizing tuning total reward at block 4",
        "rewards": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path,
                        default=base.STUDY_DIR / "artifacts" / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=STUDY_DIR / "results")
    parser.add_argument("--tune-episodes", type=int, default=64)
    parser.add_argument("--confirm-episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="substring filter on '<study>/<leaf>/seed<seed>'")
    args = parser.parse_args()
    revisions = {"experiment": base._git(STUDY_DIR),
                 "harness": base._git(Path(harness.__file__).resolve().parent)}
    for run in base.source_runs():
        if run["condition"] != "reward_both":
            continue
        tag = f"{run['study']}/{run['leaf']}/seed{run['seed']}"
        if args.only and args.only not in tag:
            continue
        stem = (f"{run['study'].removeprefix('reward_factors_persistent_nuisance_v2_jax_temp15_')}"
                f"__{run['condition']}__seed{run['seed']}")
        output = args.output_dir / f"{stem}.json"
        if output.exists():
            raise FileExistsError(output)
        report = analyze(run, base.checkpoint_path(run, args.checkpoints),
                         tune_episodes=args.tune_episodes, confirm_episodes=args.confirm_episodes,
                         seed=args.seed)
        report["revisions"] = revisions
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
