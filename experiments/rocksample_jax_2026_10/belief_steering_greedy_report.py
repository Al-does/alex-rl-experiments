"""Historical master tables for the r7 greedy/decoder analysis."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_steering_greedy_decoders as gd

bs, cont, paper = gd.bs, gd.cont, gd.paper


def agent_key(mapping):
    return f"{mapping['source_run']}_seed{mapping['seed']}"


def create_manifest(output):
    root = bs.bp.ROOT
    historical = {str(p.relative_to(root)): bs.bp.sha256(p)
        for d in (root/'results').glob('belief_steering*') if d != output.resolve()
        for p in d.rglob('*') if p.is_file()}
    sources = (Path(gd.__file__), Path(bs.bp.__file__), Path(paper.__file__),
        Path(bs.__file__), Path(cont.__file__), Path(gd.frozen.__file__), root/'model.py', root/'env.py')
    cont.write(output/'manifest.json', dict(schema=1, smoke=False, fit_episodes=256,
        test_episodes=256, tune_episodes=128, confirmation_episodes=1024,
        v4_seeds={**bs.SEEDS, 'confirm':gd.frozen.CONFIRM_SEED},
        v5_seeds={**bs.SEEDS, 'tune':paper.TUNE_SEED, 'confirm':paper.CONFIRM_SEED},
        historical=historical, sources={str(p.relative_to(root)):bs.bp.sha256(p) for p in sources},
        jax_version=gd.jax.__version__, experiment_revision=bs.bp.revision(root)))


def recompute_v1(output):
    old = cont.read(bs.bp.ROOT/'results/belief_steering_r7_20261110/results.json')
    path = output/'v1_conditional_metrics.json'
    result = cont.read(path) if path.exists() else {}
    for mapping in cont.mappings():
        key = agent_key(mapping)
        if key in result:
            continue
        params, env, spec, checkpoint = cont.load(mapping, 29)
        fitted, _, _, _ = bs.maps(spec, env, params, 256)
        agent = next(a for a in old['agents'] if (a['run_id'], a['seed']) ==
                     (mapping['source_run'], mapping['seed']))
        sites, baseline = {}, None
        for site in range(4):
            print(f'Complete v1 conditional metrics {key} site{site}', flush=True)
            rows = bs.evaluate(spec, env, params, fitted, 1024, old['seeds']['confirm'],
                               site, old['strength'], intact_rows=baseline)
            if baseline is None:
                baseline = rows
            for rock in range(7):
                for direction in paper.DIRECTIONS:
                    for condition in ('intact', 'steered', 'control'):
                        for metric in ('target_probability', 'mean_return'):
                            if rows[rock][direction][condition][metric] != agent['sites'][str(site)][rock][direction][condition][metric]:
                                raise ValueError(f'v1 replay differs: {key} site{site} {rock} {direction} {condition} {metric}')
            sites[str(site)] = rows
        result[key] = dict(checkpoint=checkpoint, sites=sites,
            target_and_return_match_historical_exactly=True)
        cont.write(path, result)
        gd.jax.clear_caches()


def selective(metrics):
    good, bad = (metrics[d]['steered'] for d in paper.DIRECTIONS)
    return (good['target'] <= .10 and bad['target'] >= .5 and
            good['nontarget_good'] >= .9 and bad['nontarget_good'] >= .9)


def metrics(rows):
    return {d: {c: paper.pooled(rows, d, c) for c in ('intact', 'steered', 'control')}
            for d in paper.DIRECTIONS}


def entries(output):
    mappings = cont.mappings()
    result = []
    def add(method, mode, update, site, agent, rows, seed, status):
        if len(rows) != 7 or any(cell[d][c]['episodes'] != 1024 for cell in rows
                for d in paper.DIRECTIONS for c in ('intact', 'steered', 'control')):
            raise ValueError('incomplete master cell')
        m = metrics(rows)
        result.append(dict(method=method, mode=mode, update=update, site=int(site),
            agent=agent, rows=rows, metrics=m, success=selective(m), confirm_seed=seed,
            site_status=status))
    v1 = cont.read(output/'v1_conditional_metrics.json')
    for i, mapping in enumerate(mappings):
        for site, rows in v1[agent_key(mapping)]['sites'].items():
            add('v1 encoder', 'sampled', 29, site, i, rows, bs.SEEDS['confirm'],
                'site 2 primary; other sites exploratory')
    for suffix, prefix in (('_v2', 'v2'), ('_v3', 'v3')):
        old = cont.read(bs.bp.ROOT/f'results/belief_steering_r7_20261110{suffix}/results.json')
        for i, mapping in enumerate(mappings):
            agent = next(a for a in old['agents'] if (a['run_id'], a['seed']) ==
                         (mapping['source_run'], mapping['seed']))
            for variant, sites in agent['variants'].items():
                for site, rows in sites.items():
                    add(f'{prefix} {variant}', 'sampled', 29, site, i, rows,
                        old['seeds']['confirm'], 'site 0 post hoc on v3 confirmation' if
                        prefix == 'v3' and int(site) == 0 else 'site 2 primary; other sites exploratory')
    for i, mapping in enumerate(mappings):
        for update in (29, 316):
            p = output/f'{agent_key(mapping)}_update{update}.json'
            state = cont.read(p)
            if state['checkpoint']['sha256'] != next(c['checkpoint']['sha256'] for c in
                    cont.read(output/'sanity.json')['checkpoints'] if c['mapping'] == mapping and c['update'] == update):
                raise ValueError('sanity and intervention checkpoint differ')
            for mode in ('sampled', 'greedy'):
                part = state['modes'][mode]
                for variant in gd.VARIANTS:
                    if len(part['v4'][variant]) != 4:
                        raise ValueError('missing v4 sites')
                    for site, rows in part['v4'][variant].items():
                        status = ('v3 post-hoc site choice; held-out v4 confirmation' if
                            mode == 'sampled' and variant == 'plain' and int(site) == 0 else
                            'all sites requested before new confirmation; best-cell selection post hoc')
                        add(f'v4 {variant}', mode, update, site, i, rows,
                            gd.frozen.CONFIRM_SEED, status)
                for representation in paper.REPRESENTATIONS:
                    if len(part['v5'][representation]) != 4:
                        raise ValueError('missing v5 sites')
                    for site, rows in part['v5'][representation].items():
                        add(f'v5 {representation}', mode, update, site, i, rows,
                            paper.CONFIRM_SEED, 'site 2 tuning fixed in advance; all sites requested')
    return result


def paired(m, condition, field, precision=3):
    return '/'.join(f'{m[d][condition][field]:.{precision}f}' for d in paper.DIRECTIONS)


def master_lines(records):
    lines = ['| Method | Mode | Update | Site | Steered G/B | Control G/B | Other Good G/B | Return G/B | Intact G/B | Intact other Good G/B | Intact return G/B | Successes |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    primary = [r for r in records if r['agent'] == 0]
    for r in primary:
        m = r['metrics']
        count = sum(s['success'] for s in records if all(s[k] == r[k] for k in
                    ('method', 'mode', 'update', 'site')))
        lines.append(f"| {r['method']} | {r['mode']} | {r['update']} | {r['site']} | " +
            ' | '.join((paired(m, 'steered', 'target'), paired(m, 'control', 'target'),
            paired(m, 'steered', 'nontarget_good'), paired(m, 'steered', 'mean_return', 2),
            paired(m, 'intact', 'target'), paired(m, 'intact', 'nontarget_good'),
            paired(m, 'intact', 'mean_return', 2), f'{count}/3'))+' |')
    return lines


def best_cells(records):
    selected = []
    for method in dict.fromkeys(r['method'] for r in records):
        candidates = [r for r in records if r['method'] == method and r['agent'] == 0]
        def rank(r):
            g, b = (r['metrics'][d]['steered'] for d in paper.DIRECTIONS)
            return (r['success'], min(1-g['target'], b['target'],
                                     g['nontarget_good'], b['nontarget_good']))
        selected.append(max(candidates, key=rank))
    return selected


def report(output):
    manifest = cont.read(output/'manifest.json')
    for path, digest in manifest['historical'].items():
        if bs.bp.sha256(bs.bp.ROOT/path) != digest:
            raise ValueError(f'historical result changed: {path}')
    for path, digest in manifest['sources'].items():
        if bs.bp.sha256(bs.bp.ROOT/path) != digest:
            raise ValueError(f'execution source changed: {path}')
    records = entries(output)
    sanity = cont.read(output/'sanity.json')
    lines = ['# RockSample greedy action modes and decoder variants', '',
        'All numbers are descriptive. G/B means Good→Bad / Bad→Good, not a quality-conditioned ratio.',
        'Three upper-plateau agents, verified updates 29 and 316; 256 fit / 256 test / 128 tuning / 1024 confirmation episodes.',
        'Agent 0: primary run 2 seed 3; agent 1: run 1 seed 2; agent 2: run 2 seed 2. Update 29 = 30,408,704 lifetime steps; update 316 = 331,350,016.',
        'Sites 0/1/2/3 are after block 1 / after block 2 / after block 3 pre-LN / after final LN.',
        'Environment resets and transition keys are matched within each protocol. Greedy changes action selection only; the environment remains stochastic.',
        'V4 seeds: fit 41101, test 51203, null 81517, confirmation 271829. V5 tuning 371839, confirmation 471841.',
        'V5 excludes decisions 0–7 in fitting/scoring only. V4 uses every valid decision. Greedy probes use greedy own-policy histories.', '',
        '## Step 0: intact-policy sanity, before fitting or interventions', '',
        'Unforced initial qualities, seed 271829, 1024 episodes. These returns mix Good/Bad targets and are not the forced-quality confirmation returns.',
        'Loops are operational flags: a cap-hitting episode with stationary last-20 positions, or a last-20 (position,action) trace periodic with period 1–8.',
        'No greedy checkpoint is degenerate by these checks; all greedy cap and loop fractions are zero.', '',
        '| Agent | Update | Mode | Return | Cap fraction | Stationary tail | Periodic tail | P(pickup given Good), rocks 0–6 | P(pickup given Bad), rocks 0–6 |',
        '|---|---:|---|---:|---:|---:|---:|---|---|']
    for i, record in enumerate(sanity['checkpoints']):
        for mode, m in record['modes'].items():
            vectors = ['; '.join(f'{x:.4f}' for x in m['pickup'][q]['probability']) for q in ('good', 'bad')]
            lines.append(f"| {i//2} | {record['update']} | {mode} | {m['mean_return']:.4f} | {m['cap_fraction']:.4f} | "
                         f"{m['stationary_tail_fraction']:.4f} | {m['periodic_tail_fraction']:.4f} | {' | '.join(vectors)} |")
    lines += ['', '## Verified checkpoint provenance', '',
        'Mappings come from continuation_source.json. Canonical B2 manifest size and SHA-256 are verified before unpickling; parameter architecture matches d128, three blocks, four heads, context 32.', '',
        '| Agent | Update | Run | Seed | Bytes | Lifetime steps | SHA-256 |',
        '|---:|---:|---|---:|---:|---:|---|']
    for i, record in enumerate(sanity['checkpoints']):
        mapping, p = record['mapping'], record['checkpoint']
        run = mapping['continuation_run'] if record['update'] == 316 else mapping['source_run']
        lines.append(f"| {i//2} | {record['update']} | {run} | {mapping['seed']} | {p['size_bytes']} | {p['lifetime_steps']} | {p['sha256']} |")
    lines += ['', '## Primary-agent master table', '',
        'Success requires pooled Good→Bad ≤0.10, Bad→Good ≥0.5, and other-Good ≥0.9 in both directions. Counts are trained agents, not episodes.',
        'Historical v1 lacks conditional nontarget counts: its recipe was replayed and target pickup/returns match every archived condition exactly.',
        'V2/v3 retain their original confirmation seeds; these are not matched-seed comparisons to v4/v5. V4/v5 sampled cells are reused only from the matched continuation comparison.',
        'V4 site 0 was chosen post hoc on v3 confirmation, then tested on held-out v4 seed 271829. New all-site comparisons were requested in advance; choosing a best cell below is post hoc.', '']
    lines += master_lines(records)
    lines += ['', '## Replicates: all methods and cells', '',
        '| Agent | Method | Mode | Update | Site | Steered G/B | Control G/B | Other Good G/B | Return G/B | Intact G/B | Intact other Good G/B | Intact return G/B | Success |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in records:
        if r['agent'] == 0:
            continue
        m = r['metrics']
        lines.append(f"| {r['agent']} | {r['method']} | {r['mode']} | {r['update']} | {r['site']} | " +
            ' | '.join((paired(m, 'steered', 'target'), paired(m, 'control', 'target'),
                paired(m, 'steered', 'nontarget_good'), paired(m, 'steered', 'mean_return', 2),
                paired(m, 'intact', 'target'), paired(m, 'intact', 'nontarget_good'),
                paired(m, 'intact', 'mean_return', 2), str(r['success'])))+' |')
    lines += ['', '## Best cell of each method — post-hoc descriptive selection', '',
        'Select on the primary agent: success first, then maximize min(1−Good→Bad pickup, Bad→Good pickup, other-Good in each direction). Ties retain registry order.',
        'This search spans the available modes, checkpoints, and sites. It is not a held-out test, and methods have different search spaces.', '']
    for r in best_cells(records):
        lines += [f"### {r['method']}: {r['mode']}, update {r['update']}, site {r['site']}", '',
            f"Original site status: {r['site_status']}.", '',
            '| Rock | Intact G/B | Steered Good→Bad [Wilson 95% CI] | Steered Bad→Good [Wilson 95% CI] | Control G/B |',
            '|---:|---:|---:|---:|---:|']
        for rock, row in enumerate(r['rows']):
            cells = []
            for d in paper.DIRECTIONS:
                c = row[d]['steered']; lo, hi = c['sample_ci95'][rock]
                cells.append(f"{c['target_probability']:.4f} [{lo:.4f}, {hi:.4f}]")
            ref = lambda condition: '/'.join(f"{row[d][condition]['target_probability']:.4f}" for d in paper.DIRECTIONS)
            lines.append(f"| {rock} | {ref('intact')} | {' | '.join(cells)} | {ref('control')} |")
    lines += ['', '## Decoder scores, calibration endpoints, and strengths', '',
        'D_p conditions on h and 144 observable nuisance columns. Its R² is NOT comparable to activation-only probe R²: it has additional position/time/action/symbol/sample-history information.',
        'z-only is held-out belief prediction from nuisance alone. Undefined per-rock R² (zero test variance) is null, not zero.',
        'In v5 greedy update-316 streams, rock 2 has exactly zero belief throughout retained fit and test decisions for all three agents. Its activation-only marginal R² is undefined, not evidence of poor or perfect decoding. The manifest records the R²-only serialization amendment needed to represent this as null; fitted maps and rollout algorithms were unchanged.',
        'All three v4 variants use the same nuisance-controlled encoder A. Plain uses W for estimate/calibration; nuisance-aware uses hWp+zG+bp and calibrates through Wp; hybrid estimates with plain D but calibrates through Wp.',
        'For hybrid, calibration-decoder endpoint need not equal the target: Dp(h+δ,z)=Dp(h,z)+target−D(h). We report that residual rather than claiming endpoint exactness. Calibration-change error checks δWp against the requested coordinate change.',
        'For nuisance-aware, the plain-decoder endpoint is also reported: exactness in Dp does not imply exactness in D.', '',
        '| Agent | Update | Mode | Site | Plain R² rocks 0–6 | D_p R² rocks 0–6 | z-only R² rocks 0–6 |',
        '|---:|---:|---|---:|---|---|---|']
    new_states = []
    for i, mapping in enumerate(cont.mappings()):
        for update in (29, 316):
            state = cont.read(output/f'{agent_key(mapping)}_update{update}.json'); new_states.append(state)
            for mode, part in state['modes'].items():
                diag = part['decoder_diagnostics']
                for site, d in enumerate(diag['sites']):
                    fmt = lambda values: '; '.join('null' if v is None else f'{v:.4f}' for v in values)
                    lines.append(f"| {i} | {update} | {mode} | {site} | {fmt(d['plain_r2'])} | {fmt(d['nuisance_aware_r2'])} | {fmt(diag['z_only_r2'])} |")
    lines += ['', '| Agent | Update | Mode | Variant | Site | Calibration endpoint max error | Plain endpoint max error | Calibration change max error |',
              '|---:|---:|---|---|---:|---:|---:|---:|']
    for i, state in enumerate(new_states):
        for mode, part in state['modes'].items():
            for variant, sites in part['v4'].items():
                for site, rows in sites.items():
                    cells = [r[d]['steered'] for r in rows for d in paper.DIRECTIONS]
                    if variant != 'plain' and any(key not in c for c in cells for key in
                            ('calibration_endpoint_error_max', 'plain_endpoint_error_max', 'calibration_change_error_max')):
                        raise ValueError('missing decoder-variant endpoint diagnostics')
                    maxima = [max(c.get(key, c['decoded_endpoint_abs_error_max']) for c in cells)
                        for key in ('calibration_endpoint_error_max', 'plain_endpoint_error_max')]
                    maxima.append(max(c['calibration_change_error_max'] for c in cells)
                                  if all('calibration_change_error_max' in c for c in cells) else None)
                    lines.append(f"| {i//2} | {state['update']} | {mode} | {variant} | {site} | " +
                                 ' | '.join('not archived' if x is None else f'{x:.6g}' for x in maxima)+' |')
    lines += ['', 'V4 λ=1 throughout.', '', 'Historical sampled strengths at update 29:', '',
        '| Method | Selected λ |', '|---|---:|']
    v1_strength = cont.read(bs.bp.ROOT/'results/belief_steering_r7_20261110/results.json')['strength']
    lines.append(f'| v1 encoder | {v1_strength} |')
    for suffix, prefix in (('_v2', 'v2'), ('_v3', 'v3')):
        old = cont.read(bs.bp.ROOT/f'results/belief_steering_r7_20261110{suffix}/results.json')
        lines += [f"| {prefix} {variant} | {value['strength']} |" for variant, value in old['tuning'].items()]
    lines += ['',
        '| Agent | Update | Mode | Representation | Direction | Selected λ | Criterion met |',
        '|---:|---:|---|---|---|---:|---|']
    for i, state in enumerate(new_states):
        for mode, part in state['modes'].items():
            for representation, directions in part['tuning']['representations'].items():
                for direction, value in directions.items():
                    lines.append(f"| {i//2} | {state['update']} | {mode} | {representation} | {direction} | {value['strength']} | {value['selection_succeeded']} |")
    lines += ['', 'Full tuning grids, per-site paper marginal/joint R², null-space diagnostics, and per-rock cells are in the checkpoint JSONs.', '',
        '## Limits', '',
        'Matched-null controls use the calibration decoder for each variant; they can disrupt behavior and do not establish specificity merely by preserving decoded coordinates.',
        'Decoded vectors are not clipped or projected onto a simplex; they may lie outside probability bounds. Pickup rates, unlike decoded coordinates, are empirical probabilities.',
        'R², calibrated endpoints, and behavioral selectivity answer different questions. Nuisance-aware estimates explicitly consume observable history summaries, so their accuracy does not demonstrate representation in activations.',
        'Greedy changes both the probe-history distribution and rollout action selection, as requested; this does not isolate an argmax-only effect. Probe scores are on intact own-policy streams, not intervention-shifted histories.',
        'Wilson intervals quantify episode sampling, not between-agent uncertainty. Best-cell selection is exploratory. No result establishes an internal Bayesian model.', '',
        '## Reproduction', '',
        'Use a fresh output directory. Run step 0 first, then the v1 conditional-metric replay (which creates the provenance manifest), then the new steering cells, then reporting:', '',
        '```bash',
        'uv run --group jax python -m experiments.rocksample_jax_2026_10.belief_steering_greedy_decoders --phase sanity --output OUTPUT',
        'uv run --group jax python -m experiments.rocksample_jax_2026_10.belief_steering_greedy_report --phase v1 --output OUTPUT',
        'uv run --group jax python -m experiments.rocksample_jax_2026_10.belief_steering_greedy_decoders --phase run --output OUTPUT',
        'uv run --group jax python -m experiments.rocksample_jax_2026_10.belief_steering_greedy_report --output OUTPUT',
        '```', '', 'Checkpoint JSONs support resuming incomplete sites with unchanged settings.', '']
    (output/'comparison.md').write_text('\n'.join(lines).rstrip()+'\n')
    compact = [{k: v for k, v in r.items() if k != 'rows'} for r in records]
    cont.write(output/'master_metrics.json', compact)
    cont.write(output/'report_source.json', dict(source=Path(__file__).name, sha256=bs.bp.sha256(Path(__file__))))
    figure(output, records)
    print(f'Master comparison complete: {len(records)} agent cells; {len(records)//3} primary cells; historical/source hashes match.')


def figure(output, records):
    primary = [r for r in records if r['agent'] == 0]
    methods = list(dict.fromkeys(r['method'] for r in primary))
    colors = dict(zip(methods, plt.get_cmap('tab20').colors))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, (update, mode) in zip(axes.flat, ((29, 'sampled'), (316, 'sampled'), (29, 'greedy'), (316, 'greedy'))):
        ax.axvspan(0, .1, ymin=.5, ymax=1, color='#deeedb', alpha=.8)
        intact_seeds = set()
        for r in primary:
            if (r['update'], r['mode']) != (update, mode):
                continue
            g, b = (r['metrics'][d]['steered'] for d in paper.DIRECTIONS)
            selective = min(g['nontarget_good'], b['nontarget_good']) >= .9
            color = colors[r['method']]
            cg, cb = (r['metrics'][d]['control']['target'] for d in paper.DIRECTIONS)
            ax.plot([cg, g['target']], [cb, b['target']], color=color, alpha=.18, lw=.7)
            ax.scatter(cg, cb, marker='x', color=color, s=18, alpha=.4)
            if r['confirm_seed'] not in intact_seeds:
                ig, ib = (r['metrics'][d]['intact']['target'] for d in paper.DIRECTIONS)
                ax.scatter(ig, ib, marker='+', color='black', s=55, linewidths=1)
                intact_seeds.add(r['confirm_seed'])
            ax.scatter(g['target'], b['target'], marker=('o', 's', '^', 'D')[r['site']],
                facecolors=[color] if selective else 'none', edgecolors=[color], s=65, linewidths=1.6)
        ax.set(title=f'Primary agent: update {update}, {mode}', xlim=(-.02, 1.02), ylim=(-.02, 1.02),
               xlabel='Steered Good→Bad target pickup (lower is better)',
               ylabel='Steered Bad→Good target pickup (higher is better)')
        ax.axvline(.1, color='gray', ls=':', lw=1); ax.axhline(.5, color='gray', ls=':', lw=1)
        ax.grid(alpha=.15)
    handles = [plt.Line2D([], [], color=colors[m], marker='o', ls='', label=m) for m in methods]
    handles += [plt.Line2D([], [], color='gray', marker=s, ls='', label=f'site {i}') for i, s in enumerate(('o', 's', '^', 'D'))]
    handles += [plt.Line2D([], [], color='gray', marker='x', ls='', label='control reference'),
                plt.Line2D([], [], color='black', marker='+', ls='', label='intact reference')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8)
    fig.suptitle('R7 master steering comparison — filled markers preserve other-Good ≥0.9 in both directions')
    fig.tight_layout(rect=(0, .17, 1, .96))
    fig.savefig(output/'master_comparison.png', dpi=180)
    fig.savefig(output/'master_comparison.svg')
    svg = output/'master_comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('v1', 'report'), default='report')
    args = parser.parse_args()
    if args.phase == 'v1' and not (args.output/'manifest.json').exists():
        create_manifest(args.output)
    (recompute_v1 if args.phase == 'v1' else report)(args.output)


if __name__ == '__main__':
    main()
