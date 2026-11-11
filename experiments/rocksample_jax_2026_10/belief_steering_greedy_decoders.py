"""Matched action-mode and nuisance-decoder interventions for r7."""

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_steering_continuation as cont

bs, paper, frozen = cont.bs, cont.paper, cont.frozen
HISTORICAL = bs.bp.ROOT/'results/belief_steering_r7_cont330m_20261111_comparison'
VARIANTS = ('plain', 'nuisance-aware', 'hybrid')


def online_nuisance(env, position, step, previous, obs, sampled):
    cell = position[..., 0] + env.n*position[..., 1]
    return jnp.concatenate((jax.nn.one_hot(cell, env.n**2)[..., :-1],
        jax.nn.one_hot(step, env.episode_length)[..., 1:],
        jax.nn.one_hot(previous, env.num_actions+1)[..., :env.num_actions],
        obs[..., 2+2*env.k:2+2*env.k+bs.rs.NUM_SYMBOLS-1], sampled), axis=-1)


def score(y, prediction):
    sst = ((y-y.mean(0))**2).sum(0)
    return [float(1-a/b) if b > 1e-15 else None
            for a, b in zip(((y-prediction)**2).sum(0), sst)]


def null_directions(d, rng, k):
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    basis = u[:, s > s[0]*1e-6]
    random = rng.normal(size=(k, d.shape[0]))
    random -= (random@basis)@basis.T
    norm = np.linalg.norm(random, axis=-1, keepdims=True)
    if (norm < 1e-10).any():
        raise ValueError('no usable decoder-null direction')
    return random/norm


def fit_decoders(spec, env, params, episodes, greedy):
    streams = []
    for seed in (bs.SEEDS['fit'], bs.SEEDS['test']):
        rows = bs.bp.collect(spec, env, params, params, episodes=episodes,
                            key_seed=seed, greedy=greedy)
        x = np.asarray(bs.replay(spec, params, rows['obs']))[rows['valid']].astype(float)
        streams.append((x, bs.bp.targets(rows, env).marginals, bs.nuisance_features(rows, env)))
    (x, y, z), (xt, yt, zt) = streams
    zg, zb = bs.affine(z, y)
    result = {v: [] for v in VARIANTS}
    diagnostics = dict(fit_rows=len(x), test_rows=len(xt), nuisance_columns=z.shape[1],
                       z_only_r2=score(yt, zt@zg+zb), sites=[])
    rngs = {v: np.random.default_rng(bs.SEEDS['null']) for v in VARIANTS}
    for site in range(4):
        h, ht = x[:, site], xt[:, site]
        d, b = bs.affine(h, y)
        coef, pb = bs.affine(np.column_stack((h, z)), y)
        wp, g = coef[:spec.d_model], coef[spec.d_model:]
        a = bs.nuisance_encoder(y, z, h)
        details = dict(plain_r2=score(yt, ht@d+b),
            nuisance_aware_r2=score(yt, ht@wp+zt@g+pb), calibration={})
        for variant in VARIANTS:
            cal = d if variant == 'plain' else wp
            product = a@cal
            e = np.linalg.solve(product, a)
            est, offset = (wp, pb) if variant == 'nuisance-aware' else (d, b)
            nuisance = g if variant == 'nuisance-aware' else np.zeros_like(g)
            null = null_directions(cal, rngs[variant], env.k)
            details['calibration'][variant] = dict(product_condition=float(np.linalg.cond(product)),
                identity_error=float(np.max(np.abs(e.astype('float32')@cal.astype('float32')-np.eye(env.k)))),
                null_decoder_max_abs=float(np.max(np.abs(null.astype('float32')@cal.astype('float32')))))
            cg, cb = (np.zeros_like(g), b) if variant == 'plain' else (g, pb)
            result[variant].append((est, offset, e, null, cal, nuisance, d, b, cg, cb))
        diagnostics['sites'].append(details)
    return {v: tuple(jnp.asarray(a, dtype=jnp.float32) for a in zip(*sites))
            for v, sites in result.items()}, diagnostics


def decoder_shift(h, z, fitted, rock, target, control, active, strength):
    d, b, e, null, cal, g, plain, pb, cg, cb = fitted
    before = h@d+z@g+b
    delta = strength*(target-before[rock])*e[rock]*active
    delta = jnp.where(control, jnp.linalg.norm(delta)*null[rock], delta)
    changed = h+delta
    after = changed@d+z@g+b
    expected_change = strength*(target-before[rock])*active
    actual_change = delta@cal
    expected = jax.nn.one_hot(rock, d.shape[1])*expected_change
    cal_endpoint = (h@cal+z@cg+cb)[rock]+actual_change[rock]
    diagnostics = jnp.stack((before[rock], after[rock],
        jnp.max(jnp.abs((after-before).at[rock].set(0))), jnp.abs(after[rock]-target),
        jnp.linalg.norm(delta), jnp.abs(cal_endpoint-target),
        jnp.abs((changed@plain+pb)[rock]-target), jnp.max(jnp.abs(actual_change-expected))))
    return changed, diagnostics


@partial(jax.jit, static_argnames=('spec', 'env', 'episodes', 'site', 'greedy'))
def rollout(spec, env, params, fitted, *, episodes, seed, rock, good, site,
            control=False, strength=1., greedy=False):
    obs, state = jax.vmap(partial(bs.rs.reset, env))(
        jax.random.split(jax.random.key(seed), episodes))
    state = state._replace(qualities=state.qualities.at[:, rock].set(jnp.asarray(good, bool)))
    initial_good = state.qualities
    cache = jax.vmap(lambda _: bs.tm.empty_cache(spec))(jnp.arange(episodes))
    def advance(carry, t):
        obs, state, cache, done, sampled, previous, key = carry
        active = ~sampled[:, rock] | jnp.asarray(good, bool)
        z = jax.vmap(partial(online_nuisance, env))(state.rover,
            jnp.full((episodes,), t), previous, obs, sampled)
        def encode(c, o, enabled, nuisance):
            diagnostics = []
            def transform(index, h):
                if index != site:
                    return h
                changed, diagnostic = decoder_shift(h.reshape(-1), nuisance,
                    tuple(a[index] for a in fitted), rock, 1-good, control, enabled, strength)
                diagnostics.append(diagnostic)
                return changed.reshape(h.shape)
            h, c = bs.tm.encode_step(spec, params, c, o, residual_transform=transform)
            return h, c, diagnostics[0]
        h, cache, diagnostic = jax.vmap(encode)(cache, obs, active, z)
        logits, _ = bs.tm.heads(params, h)
        key, ak, ek = jax.random.split(key, 3)
        action = (jnp.argmax(logits, axis=-1).astype(jnp.int32) if greedy else
                  jax.random.categorical(ak, logits).astype(jnp.int32))
        out = jax.vmap(partial(bs.rs.step, env))(jax.random.split(ek, episodes), state, action)
        valid = ~done
        picked = ((action == 4)[:, None] & valid[:, None] &
                  jnp.all(state.rover[:, None] == jnp.asarray(env.rocks), axis=-1))
        row = (picked, (action == 5+rock)&valid, out.reward*valid,
            diagnostic*(valid&active)[:, None], valid&active,
            jnp.all(state.rover == jnp.asarray(env.rocks)[rock], axis=-1)&valid,
            jnp.concatenate((state.rover[:4], action[:4, None], valid[:4, None]), axis=1))
        return (out.obs, out.state, cache, done|out.terminated|out.truncated,
                sampled|picked, action, key), row
    carry = (obs, state, cache, jnp.zeros(episodes, bool), jnp.zeros((episodes, env.k), bool),
             jnp.full((episodes,), env.num_actions), jax.random.key(seed+1))
    _, rows = jax.lax.scan(advance, carry, jnp.arange(env.episode_length))
    picked, checks, rewards, diagnostic, active, visits, traces = rows
    values = (picked.any(0), checks.sum(0), rewards.sum(0), diagnostic[:, :, :5].sum((0, 1)),
        active.sum(), initial_good, visits.any(0), traces,
        jnp.max(jnp.where(active, diagnostic[:, :, 3], 0)),
        jnp.max(jnp.where(active, diagnostic[:, :, 2], 0)))
    return values, jnp.max(jnp.where(active[:, :, None], diagnostic[:, :, 5:], 0), axis=(0, 1))


def evaluate(spec, env, params, fitted, episodes, site, greedy, baselines):
    rows = []
    for rock in range(env.k):
        row = {}
        for good, direction in zip((1, 0), paper.DIRECTIONS):
            cells = dict(intact=baselines[rock][direction]['intact'])
            for condition in ('steered', 'control'):
                values, extra = rollout(spec, env, params, fitted, episodes=episodes,
                    seed=frozen.CONFIRM_SEED, rock=rock, good=good, site=site,
                    control=condition == 'control', greedy=greedy)
                cell = bs.summarize(values, rock)
                cell.update(zip(('calibration_endpoint_error_max', 'plain_endpoint_error_max',
                                 'calibration_change_error_max'), map(float, extra)))
                cells[condition] = cell
            row[direction] = cells
        rows.append(row)
    return rows


def run(output, smoke=False, updates=(29, 316)):
    sanity_record = cont.read(output/'sanity.json')
    if len(sanity_record['checkpoints']) != (2 if smoke else 6):
        raise ValueError('complete step-0 sanity before fitting or steering')
    episodes, fit, tune = (8, 8, 8) if smoke else (1024, 256, 128)
    for mapping in cont.mappings()[:1] if smoke else cont.mappings():
        for update in updates:
            params, env, spec, provenance = cont.load(mapping, update)
            path = output/f"{mapping['source_run']}_seed{mapping['seed']}_update{update}.json"
            state = cont.read(path) if path.exists() else dict(mapping=mapping, update=update,
                checkpoint=provenance, modes={})
            old = cont.read(HISTORICAL/path.name)
            for mode in ('sampled', 'greedy'):
                greedy = mode == 'greedy'
                part = state['modes'].setdefault(mode, dict(v4={}, v5={}))
                if any(len(part['v4'].get(v, {})) < 4 for v in VARIANTS):
                    fitted, diagnostics = fit_decoders(spec, env, params, fit, greedy)
                    part['decoder_diagnostics'] = diagnostics
                    if greedy:
                        baseline = paper.evaluate(spec, env, params, fitted['plain'][:4], 'marginal',
                            episodes, frozen.CONFIRM_SEED, 0, dict.fromkeys(paper.DIRECTIONS, 0.),
                            conditions=('intact',), greedy=True)
                    else:
                        baseline = old['v4']['sites']['0']
                    for variant in VARIANTS:
                        cells = part['v4'].setdefault(variant, {})
                        for site in range(4):
                            if str(site) in cells:
                                continue
                            print(f"u{update} seed{mapping['seed']} {mode} v4 {variant} site{site}", flush=True)
                            cells[str(site)] = (old['v4']['sites'][str(site)] if not greedy and
                                variant == 'plain' and not smoke else
                                evaluate(spec, env, params, fitted[variant], episodes, site, greedy, baseline))
                            cont.write(path, state)
                if greedy and any(len(part['v5'].get(rep, {})) < 4 for rep in paper.REPRESENTATIONS):
                    fitted, r2, diagnostics = paper.fit_maps(spec, env, params, fit, greedy=True)
                    part['paper_r2'], part['paper_diagnostics'] = r2, diagnostics
                    if 'tuning' not in part:
                        part['tuning'] = paper.tune(spec, env, params, fitted, tune, greedy=True)
                        cont.write(path, state)
                    baseline = paper.evaluate(spec, env, params, fitted['marginal'], 'marginal',
                        episodes, paper.CONFIRM_SEED, 2, dict.fromkeys(paper.DIRECTIONS, 0.),
                        conditions=('intact', 'random'), greedy=True)
                    for representation in paper.REPRESENTATIONS:
                        cells = part['v5'].setdefault(representation, {})
                        selected = cont.strengths(part['tuning'], representation)
                        for site in range(4):
                            if str(site) in cells:
                                continue
                            print(f"u{update} seed{mapping['seed']} greedy v5 {representation} site{site} λ={selected}", flush=True)
                            cells[str(site)] = paper.evaluate(spec, env, params, fitted[representation],
                                representation, episodes, paper.CONFIRM_SEED, site, selected,
                                baselines=baseline, greedy=True)
                            cont.write(path, state)
                elif not greedy:
                    part['v5'] = old['v5']['representations']
                    part['tuning'], part['paper_r2'] = old['v5']['tuning'], old['v5']['r2']
                    cont.write(path, state)
            jax.clear_caches()


def sanity_metrics(rows, env):
    valid = rows['valid']
    initial = rows['qualities'][:, 0]
    picked = ((rows['action'] == 4)[:, :, None] & valid[:, :, None] &
              np.all(rows['position'][:, :, None] == np.asarray(env.rocks), axis=-1)).any(1)
    cap = (rows['truncated'] & valid).any(1)
    tail = rows['position'][:, -20:]
    stationary = cap & np.all(tail == tail[:, :1], axis=(1, 2))
    traces = np.concatenate((rows['position'], rows['action'][:, :, None]), axis=-1)
    periodic = np.zeros(len(valid), bool)
    for period in range(1, 9):
        periodic |= cap & np.all(traces[:, -20:-period] == traces[:, -20+period:], axis=(1, 2))
    probabilities = {}
    for label, mask in (('good', initial), ('bad', ~initial)):
        probabilities[label] = dict(count=mask.sum(0).tolist(),
            picked=(picked & mask).sum(0).tolist(),
            probability=((picked & mask).sum(0)/mask.sum(0)).tolist())
    return dict(episodes=len(valid), mean_return=float((rows['reward']*valid).sum(1).mean()),
        cap_fraction=float(cap.mean()), stationary_tail_fraction=float(stationary.mean()),
        periodic_tail_fraction=float(periodic.mean()), pickup=probabilities,
        examples=[dict(episode=int(i), tail=traces[i, -20:].tolist())
                  for i in np.flatnonzero(periodic)[:3]])


def sanity(output, smoke=False):
    path = output/'sanity.json'
    if path.exists():
        raise FileExistsError(path)
    result = dict(seed=frozen.CONFIRM_SEED, episodes=8 if smoke else 1024,
                  forced_quality=False, checkpoints=[])
    for mapping in cont.mappings()[:1] if smoke else cont.mappings():
        for update in (29, 316):
            params, env, spec, provenance = cont.load(mapping, update)
            record = dict(mapping=mapping, update=update, checkpoint=provenance, modes={})
            for mode in ('sampled', 'greedy'):
                rows = bs.bp.collect(spec, env, params, params, episodes=result['episodes'],
                    key_seed=result['seed'], greedy=mode == 'greedy')
                record['modes'][mode] = sanity_metrics(rows, env)
                m = record['modes'][mode]
                print(f"{mapping['source_run']} seed{mapping['seed']} u{update} {mode}: "
                      f"return={m['mean_return']:.4f} cap={m['cap_fraction']:.4f} "
                      f"stationary={m['stationary_tail_fraction']:.4f} periodic={m['periodic_tail_fraction']:.4f}", flush=True)
            result['checkpoints'].append(record)
            cont.write(path, result)
            jax.clear_caches()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--phase', choices=('sanity', 'run'), default='sanity')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'sanity':
        sanity(args.output, args.smoke)
    else:
        run(args.output, args.smoke)


if __name__ == '__main__':
    main()
