"""Block-4 XOR twin-write evaluation for the v2 and paper eps02 agents."""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving as old
from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_twin_steering_2026_11 import twins
from experiments.belief_quotient_twin_steering_2026_11.ladder import logits_at_block4


@jax.jit
def observations(process, actions, key):
    """Recover the pinned collector's exact observation stream from its actions."""
    episodes, horizon = actions.shape
    reset_key, step_key, _ = jax.random.split(key, 3)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))

    def step(state, data):
        t, action = data
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        state, _, _ = jax.vmap(env.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), state, action)
        return state, obs

    return jax.lax.scan(step, state, (jnp.arange(horizon), actions.T))[1].swapaxes(0, 1)


@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def rollout(spec, params, process, episodes, policy, target, writer, q, schedule, key,
            strength, use_null=False):
    reset_key, step_key, action_key = jax.random.split(key, 3)
    initial = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    initial_b = jax.vmap(env.belief_reset, (None, 0))(process, initial.token)
    table = jnp.asarray(base.ACTION_TABLE)
    acc = dict(reward=jnp.zeros((episodes, 2, schedule.shape[1])), agree=jnp.zeros((episodes, 2)),
               changed=jnp.zeros((episodes, 2)), norm=jnp.zeros(episodes))

    def body(state, t):
        env_state, carry, belief, acc = state
        obs = jax.vmap(env.observe, (None, 0))(process, env_state)
        logits, sites, carry = base.forward(spec, params, carry, obs, t)
        intact = logits.argmax(-1)
        mapped = (belief @ q[t]).argmax(-1)
        norm = jnp.zeros(episodes)
        if policy == "twin":
            h = sites[:, twins.SITE]
            b = h @ writer["decoder"] + writer["bias"]
            goal = old.belief_target(b, target, q[t], writer["anchors"], (True, True))
            shift = twins.scaled_shift(h, goal, writer["write"], strength, use_null, writer["null"])
            action = logits_at_block4(params, h + shift).argmax(-1)
            norm = jnp.linalg.norm(shift, axis=-1)
        elif policy == "random":
            action = jax.random.randint(jax.random.fold_in(action_key, t), (episodes,), 0, 9)
        elif policy == "map":
            action = mapped
        elif policy == "intact":
            action = intact
        else:
            raise ValueError(policy)
        executed = table[action]
        phase = (t // old.PERIOD) % 2
        acc = dict(reward=acc["reward"].at[:, phase].add(schedule[t][:, env_state.state].T),
                   agree=acc["agree"] + (executed == table[mapped]),
                   changed=acc["changed"] + (executed != table[intact]), norm=acc["norm"] + norm)
        next_state, _, _ = jax.vmap(env.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), env_state, action)
        belief = jax.vmap(env.belief_update, (None, 0, 0, 0))(process, belief, action, next_state.token)
        return (next_state, carry, belief, acc), None

    init = (initial, models.init_carry(spec, episodes), initial_b, acc)
    return jax.lax.scan(body, init, jnp.arange(base.HORIZON))[0][-1]


def metrics(acc):
    def summary(values):
        return values.mean(0).tolist(), (values.std(0, ddof=1) / math.sqrt(len(values))).tolist()

    phase = np.asarray(acc["reward"], np.float64)
    total, total_se = summary(phase.sum((1, 2)) / base.HORIZON)
    component, component_se = summary(phase.sum(1) / base.HORIZON)
    phase_mean, phase_se = summary(phase / (base.HORIZON / 2))
    return dict(total_reward=total, total_reward_se=total_se, component_reward=component,
                component_reward_se=component_se, phase_component_reward=phase_mean,
                phase_component_reward_se=phase_se,
                factor_action_agrees_with_exact_belief_map=(np.asarray(acc["agree"]).mean(0) / base.HORIZON).tolist(),
                factor_action_change_fraction=(np.asarray(acc["changed"]).mean(0) / base.HORIZON).tolist(),
                shift_norm=float(np.asarray(acc["norm"]).mean() / base.HORIZON))


def contrasts(b, target):
    factors = np.asarray(base.decoded_factor_beliefs(jnp.asarray(b), target))
    return np.column_stack((factors[:, 0, 2], factors[:, 0, 0] - factors[:, 0, 1],
                            factors[:, 1, 2], factors[:, 1, 0] - factors[:, 1, 1]))


def run(spec, process, params, baseline, emission, smoke=False):
    fit_rows, test_rows = (256, 256) if smoke else (base.FIT_ROWS, 12000)
    tune_episodes, confirm_episodes = (2, 2) if smoke else (64, 256)
    np.testing.assert_array_equal(np.asarray(process.transitions, np.float64) /
                                  np.asarray(process.transitions, np.float64).sum(2, keepdims=True),
                                  old.joint_transitions())
    q = jnp.asarray(old.q_tables("xor", "discounted"), jnp.float32)
    schedule = jnp.asarray(old.reward_schedule("xor"), jnp.float32)

    def collect(seed, count):
        episodes = -(-count // (base.HORIZON - base.WARMUP))
        key = base._keys(seed)[0]
        record = base.collect(spec, params, process, episodes, base.HORIZON, key)
        h = twins.rows(record["sites"][:, :, twins.SITE], base.WARMUP, count)
        b = twins.rows(record["belief"], base.WARMUP, count)
        obs = observations(process, record["action"], key)
        pair = twins.paired_rows(spec, params, process, obs, h, b, emission, 2, seed, base.WARMUP, count)
        a = twins.rows(record["action"], base.WARMUP, count)
        return h, b, a, pair

    h, b, a, train = collect(twins.FIT_SEED, fit_rows)
    test_h, test_b, _, test = collect(twins.TEST_SEED, test_rows)
    learned = base.learned_action_map({"belief": b, "action": a})
    anchors = jnp.asarray(np.stack([base.anchor_states(np.asarray(learned[f"factor_{f}"])) for f in (1, 2)]))
    writers, diagnostics = {}, {}
    for target in base.TARGETS:
        def coordinates(b):
            return base.map_targets(b)[target]

        paired = lambda p: (p[0], coordinates(p[1]), p[2], coordinates(p[3]))
        train_pairs, test_pairs = paired(train), paired(test)
        decoder, bias = base.affine_fit(h.astype(np.float64), coordinates(b).astype(np.float64))
        write = twins.fit_write(*train_pairs)
        sampled = np.arange(0, test_rows, 12)
        b_decoded = (test_h[sampled] @ decoder + bias).astype(np.float32)
        t = sampled % (base.HORIZON - base.WARMUP) + base.WARMUP
        desired = jax.vmap(lambda b, q: old.belief_target(b[None], target, q, anchors, (True, True))[0])(
            jnp.asarray(b_decoded), q[t])
        diagnostics[target] = dict(
            coordinates=["rho_f1", "delta_f1", "rho_f2", "delta_f2"],
            paired_write=twins.specificity(write, train_pairs, test_pairs, partial(contrasts, target=target)),
            goal_support=twins.support(train_pairs[1], train_pairs[3], coordinates(test_b[sampled]),
                                       np.asarray(desired)),
            old_probe_r2=base.r_squared(test_h @ decoder + bias, coordinates(test_b)))
        writers[target] = dict(
            decoder=jnp.asarray(decoder, jnp.float32), bias=jnp.asarray(bias, jnp.float32), anchors=anchors,
            write=jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), write),
            null=jnp.asarray(base.null_direction(decoder, twins.FIT_SEED + 9999), jnp.float32))
    print(f'{baseline["cohort"]} diagnostic seed {baseline["seed"]}: {diagnostics}', flush=True)

    def evaluate(seed, episodes, target="joint", strength=0., policy="twin", null=False):
        return metrics(rollout(spec, params, process, episodes, policy, target, writers[target], q,
                               schedule, base._keys(seed)[0], jnp.float32(strength), null))

    controls = {p: evaluate(twins.CONFIRM_SEED, confirm_episodes, policy=p)
                for p in ("intact", "random", "map")}
    comparisons = {}
    for target in base.TARGETS:
        grid = (0., 1.) if smoke else base.STRENGTHS
        sweep = {l: evaluate(twins.TUNE_SEED, tune_episodes, target, l)["total_reward"] for l in grid}
        strength = min(grid, key=lambda l: (-sweep[l], l))
        comparisons[target] = dict(
            selected_strength=strength, tuning_xor_reward=sweep,
            selected=evaluate(twins.CONFIRM_SEED, confirm_episodes, target, strength),
            matched_null=evaluate(twins.CONFIRM_SEED, confirm_episodes, target, strength, null=True))
    return dict(old=baseline, diagnostics=diagnostics, controls=controls, comparisons=comparisons,
                action_anchors=anchors.tolist(), protocol=dict(
                    fit_rows=fit_rows, heldout_rows=test_rows, warmup=base.WARMUP, horizon=base.HORIZON,
                    tune_episodes=tune_episodes, confirmation_episodes=confirm_episodes, gamma=base.GAMMA))
