"""Point and empirical targets with archived readouts and action maps."""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving as old
from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
from experiments.belief_quotient_history_jax_2026_11 import env, models
from . import replay

METHODS = {k: i for i, k in enumerate(("intact", "random", "map", "eq4", "encoder", "patch", "mirror_input"))}
SITE = 3


def prototypes(b, a):
    a = replay.ACTION[a]
    out = np.zeros((2, 3, 3, 3))
    valid = np.zeros((2, 3), bool)
    for f in range(2):
        for action in range(3):
            seen = b[a[:, f] == action, f]
            if len(seen):
                seen = seen[np.argsort(seen[:, 2], kind="stable")]
                out[f, action] = seen[np.rint((len(seen) - 1) * np.array([.25, .5, .75])).astype(int)]
                valid[f, action] = True
    return out, valid


def empirical_target(b, request, training_map, proto, available, steered, keep_agreed=False):
    current = jnp.take_along_axis(training_map[None], b.argmax(-1)[..., None], -1)[..., 0]
    change = (request != current) & steered
    opposite = change & (request != 0) & (current != 0)
    p = proto[jnp.arange(2), request]
    dist = jnp.square(p - b[..., None, :]).sum(-1)
    nearest = jnp.take_along_axis(p, dist.argmin(-1)[..., None, None], -2)[..., 0, :]
    match = available[jnp.arange(2), request]
    replace = steered & match & (change | ~jnp.asarray(keep_agreed))
    goal = jnp.where(replace[..., None], nearest, b)
    goal = jnp.where(opposite[..., None], b[..., jnp.asarray(replay.TOKEN_SWAP)], goal)
    frame = opposite[:, 0].astype(jnp.int32) + 2 * opposite[:, 1].astype(jnp.int32)
    covered = (frame != 0) & jnp.all(~replace | opposite, -1)
    return goal, frame, covered


def coordinates(b, target):
    if target == "marginal":
        return b[..., :2].reshape(-1, 4)
    return (b[:, 0, :, None] * b[:, 1, None, :]).reshape(-1, 9)[:, :8]


def prepare(rows):
    learned = base.learned_action_map(rows)
    training_map = np.stack([learned[f"factor_{f}"] for f in (1, 2)])
    proto, valid = prototypes(np.asarray(base.factor_beliefs(rows["belief"])), rows["action"])
    shared = dict(anchors=np.stack([base.anchor_states(m) for m in training_map]), training_map=training_map.argmax(-1),
                  prototype=proto, valid=valid)
    maps = {}
    for name, b in base.map_targets(rows["belief"]).items():
        h = rows["sites"][:, SITE].astype(float)
        b = b.astype(float)
        decoder, bias = base.affine_fit(h, b)
        enc, _ = base.affine_fit(b, h)
        maps[name] = {k: jnp.asarray(v, jnp.bool_ if k == "valid" else jnp.int32 if k in ("anchors", "training_map")
                                    else jnp.float32) for k, v in dict(shared, decoder=decoder, bias=bias, embedding=enc).items()}
    return maps


def logits(params, h):
    return models._dense(params["policy"], models._layer_norm(params["final_norm"], h))


def task_tables(task, process):
    transitions = np.asarray(process.transitions, dtype=np.float64)
    transitions /= transitions.sum(2, keepdims=True)
    np.testing.assert_array_equal(transitions, old.joint_transitions())
    if task == "moving":
        return old.q_tables(task, "discounted"), old.reward_schedule(task)
    if task != "xor":
        raise ValueError(task)
    first, second = np.divmod(np.arange(9), 3)
    schedule = np.broadcast_to(((first == 2) != (second == 2)).astype(float), (base.HORIZON, 1, 9))
    q = np.zeros((base.HORIZON, 9, 9))
    value = np.zeros(9)
    for t in range(base.HORIZON - 1, -1, -1):
        q[t] = schedule[t, 0, :, None] + base.GAMMA * np.einsum("ast,t->sa", transitions, value)
        value = q[t].max(1)
    return q, schedule


def marginal_tv(p, q):
    a, b = (x[..., jnp.asarray(replay.INVERSE)] for x in (p, q))
    return .5 * jnp.stack((jnp.abs(a.sum(-1) - b.sum(-1)).sum(-1),
                           jnp.abs(a.sum(-2) - b.sum(-2)).sum(-1)), -1)


@partial(jax.jit, static_argnames=("spec", "episodes", "target", "families", "steered", "length", "capture"))
def evaluate(spec, params, process, episodes, target, families, encoders, steered, length, q, schedule,
             key, maps, method, reachable, strength, capture=False):
    reset_key, step_key, action_key = jax.random.split(key, 3)
    initial = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    belief = jax.vmap(env.belief_reset, (None, 0))(process, initial.token)
    table = jnp.asarray(replay.ACTION)
    perm = jnp.asarray(replay.PERM)
    positive_frames = (1, 2, 3) if steered[0] else (2,)
    shadow = models.init_carry(spec, episodes)
    initial_frames = models.init_carry(spec, episodes * len(positive_frames))
    support = jnp.asarray(steered)
    acc = dict(
        reward=jnp.zeros((episodes, 2, schedule.shape[1])),
        occupation=jnp.zeros((episodes, 2)),
        agree=jnp.zeros((episodes, 2)),
        request_agree=jnp.zeros((episodes, 2)),
        changed=jnp.zeros((episodes, 2)),
        phase_agree=jnp.zeros((episodes, 2, 2)),
        fidelity=jnp.zeros((3, 2, 2)),
        fidelity_n=jnp.zeros(3), drop_sum=jnp.zeros(3), belief_error=jnp.zeros((3, 2)),
        state3_agree=jnp.zeros((episodes, 2, 2)), state3_n=jnp.zeros((episodes, 2, 2)))

    def body(carry, t):
        state, cache, frame_cache, last_base, last_refs, drop_history, b, acc = carry
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        z, sites, cache = base.forward(spec, params, cache, obs, t)
        h = sites[:, SITE]
        agent = z.argmax(-1)
        exact_map = (b @ q[t]).argmax(-1)
        raw = h @ maps["decoder"] + maps["bias"]
        decoded = base.decoded_factor_beliefs(raw, target)
        bf = jnp.clip(decoded, 0., 1.)
        bf /= jnp.maximum(bf.sum(-1, keepdims=True), 1e-7)
        requested = table[(old.decoded_joint(raw, target) @ q[t]).argmax(-1)]
        rgoal, frame, covered = empirical_target(bf, requested, maps["training_map"], maps["prototype"],
                                                 maps["valid"], support, keep_agreed=steered[0])
        phase = (t // old.PERIOD) % 2
        need_refs = capture | (method == METHODS["patch"]) | (method == METHODS["mirror_input"])

        def ref_outputs():
            obs_frames = jnp.concatenate([replay.mirror_inputs(obs, f, jnp) for f in positive_frames])
            zr, hr, cache = base.forward(spec, params, frame_cache, obs_frames, t)
            zr = zr.reshape(len(positive_frames), episodes, 9)
            hr = hr[:, SITE].reshape(len(positive_frames), episodes, spec.width)
            zp = jnp.broadcast_to(z, (4,) + z.shape).at[jnp.asarray(positive_frames)].set(zr)
            hp = jnp.broadcast_to(h, (4,) + h.shape).at[jnp.asarray(positive_frames)].set(hr)
            return zp, hp, cache

        z_all, h_all, frame_cache = jax.lax.cond(
            need_refs, ref_outputs, lambda: (jnp.broadcast_to(z, (4,) + z.shape),
                                              jnp.broadcast_to(h, (4,) + h.shape), frame_cache))
        idx = jnp.arange(episodes)
        zref = z_all[frame, idx]
        h_ref = h_all[frame, idx]
        mframe = frame if steered[0] else jnp.full_like(frame, phase * 2)
        base_prev = jnp.take_along_axis(last_base, state.prev_action[:, None].clip(0), -1)[:, 0]
        aligned_prev = perm[:, state.prev_action.clip(0)]
        ref_prev = jnp.take_along_axis(last_refs, aligned_prev[..., None], -1)[..., 0]
        instantaneous_drop = jnp.where(t > 0, base_prev - ref_prev, 0.)
        drop_history = drop_history.at[t % 8].set(instantaneous_drop)
        drop = drop_history.sum(0)[frame, idx] / jnp.maximum(jnp.minimum(t, 8), 1)

        def eq4():
            point = old.belief_target(raw, target, q[t], maps["anchors"], steered)
            # Keep the historical joint-product target when the requested action differs.
            goal = jnp.where(reachable, coordinates(rgoal, target), point)
            return (goal - raw) @ maps["embedding"]

        def encoder():
            anchor = maps["anchors"][jnp.arange(2), requested]
            point = jnp.where(support[None, :, None], jax.nn.one_hot(anchor, 3), bf)
            goal = jnp.where(reachable, rgoal, point)
            nuisance = jnp.zeros((episodes, 12))

            def step(current, activation, f):
                center, pc, coef = encoders[f]
                next_b = current.at[:, f].set(goal[:, f])
                shift = replay.design(current, next_b, activation, nuisance, f, families[f], center, pc, jnp) @ coef
                return next_b, activation + shift

            if steered[0]:
                before, first = step(bf, h, 0)
                _, first = step(before, first, 1)
                before, second = step(bf, h, 1)
                _, second = step(before, second, 0)
                return (first + second) * .5 - h
            return step(bf, h, 1)[1] - h

        zero = lambda: jnp.zeros_like(h)
        delta = jax.lax.switch(method, (zero, zero, zero, eq4, encoder,
                                       lambda: jnp.where(covered[:, None], h_ref - h, 0.), zero))
        written = logits(params, h + strength * delta)
        action = written.argmax(-1)
        action = jnp.where(method == METHODS["intact"], agent, action)
        action = jnp.where(method == METHODS["map"], exact_map, action)
        action = jnp.where(method == METHODS["random"],
                           jax.random.randint(jax.random.fold_in(action_key, t), (episodes,), 0, 9), action)
        action = jnp.where(method == METHODS["mirror_input"],
                           perm[mframe, z_all[mframe, idx].argmax(-1)], action)
        taken = table[action]
        agreed = taken == table[exact_map]
        exact = base.factor_beliefs(b)
        state3 = exact.argmax(-1) == 2
        reward = schedule[t][:, state.state].T
        acc = dict(acc,
            reward=acc["reward"].at[:, phase].add(reward),
            occupation=acc["occupation"] + (jnp.stack((state.state // 3, state.state % 3), -1) == 2),
            agree=acc["agree"] + agreed,
            request_agree=acc["request_agree"] + (taken == requested),
            changed=acc["changed"] + (taken != table[agent]),
            phase_agree=acc["phase_agree"].at[:, phase].add(agreed),
            state3_agree=acc["state3_agree"].at[:, phase].add(agreed & state3),
            state3_n=acc["state3_n"].at[:, phase].add(state3))
        if capture:
            p, p_ref, p_out = (jax.nn.softmax(x / 1.5) for x in (z, zref, written))
            mirrored_logits = jnp.take_along_axis(z_all[mframe, idx], perm[mframe], -1)
            p_out = jnp.where(method == METHODS["mirror_input"], jax.nn.softmax(mirrored_logits / 1.5), p_out)
            effect = marginal_tv(p, p_ref)
            err = marginal_tv(p_out, p_ref)
            mask = jnp.stack((covered, covered & (drop < .5), covered & (drop >= .5)), 0)
            acc["fidelity"] += jnp.einsum("kb,bij->kij", mask.astype(jnp.float32), jnp.stack((effect, err), -1))
            acc["fidelity_n"] += mask.sum(-1)
            acc["drop_sum"] += (mask * drop).sum(-1)
            flipped = ((frame[:, None] >> jnp.arange(2)) & 1).astype(bool)
            exact_ref = jnp.where(flipped[..., None], exact[..., jnp.asarray(replay.TOKEN_SWAP)], exact)
            acc["belief_error"] += mask @ (.5 * jnp.abs(exact_ref - rgoal).sum(-1))
        next_state, _, _ = jax.vmap(env.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), state, action)
        b = jax.vmap(env.belief_update, (None, 0, 0, 0))(process, b, action, next_state.token)
        lp = jax.nn.log_softmax(z / 1.5)
        lpr = jax.nn.log_softmax(z_all / 1.5)
        return (next_state, cache, frame_cache, lp, lpr, drop_history, b, acc), None

    carry = (initial, shadow, initial_frames, jnp.zeros((episodes, 9)), jnp.zeros((4, episodes, 9)),
             jnp.zeros((8, 4, episodes)), belief, acc)
    return jax.lax.scan(body, carry, jnp.arange(length))[0][-1]


def metrics(acc, length, capture):
    a = {k: np.asarray(v, float) for k, v in acc.items()}
    ep = a["reward"].sum((1, 2)) / length
    counts = np.bincount(np.arange(length) // old.PERIOD % 2, minlength=2)
    phase = [(a["reward"][:, p] / n).mean(0).tolist() if n else None for p, n in enumerate(counts)]
    agreement = [(a["phase_agree"][:, p] / n).mean(0).tolist() if n else None for p, n in enumerate(counts)]
    state3 = a["state3_n"].sum(0)
    result = dict(reward=float(ep.mean()), reward_se=float(ep.std(ddof=1) / math.sqrt(len(ep))),
                  phase_reward=phase, phase_agree=agreement,
                  state3_agreement=[[float(x / n) if n else None for x, n in zip(row, denom)] for row, denom
                                    in zip(a["state3_agree"].sum(0), state3)],
                  components=(a["reward"].sum(1) / length).mean(0).tolist(),
                  occupancy=(a["occupation"] / length).mean(0).tolist(),
                  change=(a["changed"] / length).mean(0).tolist(),
                  exact_agree=(a["agree"] / length).mean(0).tolist(),
                  request_agree=(a["request_agree"] / length).mean(0).tolist())
    if capture:
        result["support"] = {label: dict(n=int(n), coverage=float(n / (len(ep) * length)),
                                        fidelity=[float(1 - err / effect) if n and effect > n * 1e-6 else None
                                                  for effect, err in rows],
                                        replay_tv=[float(effect / n) if n else None for effect, _ in rows],
                                        mean_drop=float(drop / n) if n else None,
                                        exact_reference_target_tv=(bv / n).tolist() if n else None)
                             for label, n, rows, drop, bv in zip(("all", "low", "high"), a["fidelity_n"],
                                                                 a["fidelity"], a["drop_sum"], a["belief_error"])}
    return result


def compare(spec, params, process, rows, encoder_maps, archived, task, smoke):
    maps = prepare(rows)
    enc = tuple(encoder_maps[i][SITE] for i in range(2))
    families, encoders = zip(*enc)
    steered = (task == "xor", True)
    q, schedule = (jnp.asarray(x, jnp.float32) for x in task_tables(task, process))
    nt, nc = (2, 2) if smoke else (archived["tune_episodes"], archived["confirm_episodes"])
    length = 512 if smoke else base.HORIZON
    grid = (0., 1.) if smoke else base.STRENGTHS
    tune_seed, confirmation = archived["tune_seed"], archived["confirm_seed"]

    def run(method, target, goal, strength, tune=False, capture=True):
        episodes, seed = (nt, tune_seed) if tune else (nc, confirmation)
        acc = evaluate(spec, params, process, episodes, target, families, encoders, steered, length, q, schedule,
                       base._keys(seed)[0], maps[target], METHODS[method], goal == "reachable", jnp.float32(strength),
                       capture=capture)
        return metrics(acc, length, capture)

    old = archived["rewards"][task]["confirmation"]
    reports = {method: run(method, "joint", "reachable", 0., capture=method == "mirror_input")
               for method in ("intact", "random", "map", "mirror_input")}
    reports["archive"] = {k: v for k, v in old["comparisons"].items() if k.startswith("discounted") and k.endswith("block_4")}
    reports["writes"] = {}
    for target in base.TARGETS:
        for method in ("eq4", "encoder"):
            for goal in ("point", "reachable"):
                # The published comparator already has a tuning sweep on these episodes.
                if not smoke and method == "eq4" and goal == "point":
                    strength = reports["archive"][f"discounted_{target}_block_4"]["strength"]
                    curve = archived["rewards"][task]["strength_tuning"]["grid"][f"discounted_{target}"]
                    curve = {k: v["total_reward"] for k, v in curve.items()}
                else:
                    curve = {str(s): run(method, target, goal, s, tune=True, capture=False)["reward"] for s in grid}
                    strength = min(float(s) for s, r in curve.items() if r == max(curve.values()))
                report = run(method, target, goal, strength)
                reports["writes"][f"{target}_{method}_{goal}"] = dict(report, strength=strength, tuning=curve)
        reports["writes"][f"{target}_patch_reachable"] = dict(run("patch", target, "reachable", 1.), strength=1.)
    delta = reports["map"]["reward"] - reports["intact"]["reward"]
    for value in list(reports["writes"].values()) + [reports["mirror_input"]]:
        value["gap_closed"] = (value["reward"] - reports["intact"]["reward"]) / delta if delta else None
    return reports
