"""Aggregate epsilon-0.02 ladder steering reports over training seeds.

Reads the compact per-seed reports written by ``belief_steering`` and reports
the mean and across-seed standard deviation of each confirmation condition,
together with the exact-filter reference from ``steering_reference``.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.name
CONTINUATION = PARENT.replace('_2026_11', '_cont4b_2026_11')
NAMES = {1: 'Trivial Quotient', 2: 'Reward-State Quotient', 3: 'Full-State Quotient'}
MODES = ('point', 'mirror', 'rho_flip', 'rho_point')
LAGS = ('0', '1', '32', '33', '64')


def load(study, variant):
    folder = HERE.parent/study/f'variant_{variant}_transformer'/'results'
    reports = [json.loads(p.read_text()) for p in sorted(folder.glob('belief_steering_v2_seed*_stable.json'))]
    return sorted(reports, key=lambda r: r['seed'])


def stat(values):
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return None
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else 0.,
                min=float(values.min()), max=float(values.max()), n=int(len(values)))


def rows(report):
    """Confirmation entries keyed by (condition, site); norm-matched entries carry their mode."""
    return {(c['condition'], c['site']): dict(c['metrics'], strength=c['strength']) for c in report['confirmation']}


def summarize(reports):
    keys = sorted({k for r in reports for k in rows(r)})
    confirmation = []
    for condition, site in keys:
        found = [rows(r).get((condition, site)) for r in reports]
        found = [m for m in found if m is not None]
        entry = dict(condition=condition, site=site, seeds=len(found),
                     finite=int(sum(bool(np.isfinite(m['occupancy'])) for m in found)))
        for field in ('occupancy', 'action_change', 'noop_given_reward', 'relative_shift', 'reference_action_agreement', 'strength'):
            entry[field] = stat([m.get(field) for m in found if np.isfinite(m['occupancy'])])
        confirmation.append(entry)
    probes = {target: [stat([r['probes']['exclusion_8'][site][target] for r in reports]) for site in range(5)]
              for target in ('rho', 'delta')}
    pulses = {}
    for name in reports[0]['pulse']['conditions']:
        cells = [r['pulse']['conditions'][name] for r in reports]
        index = 0 if name.startswith('rho') else 1
        reference = 'full_filter' if 'full_filter' in cells[0] else next(k for k in cells[0] if k.endswith('filter'))
        pulses[name] = dict(
            written=('rho', 'delta')[index], reference=reference,
            measured={lag: stat([c['measured']['absolute_at_lag'][lag][index] for c in cells]) for lag in LAGS},
            filter={lag: stat([c[reference]['absolute_at_lag'][lag][index] for c in cells]) for lag in LAGS},
            retained_lag1=stat([c['measured']['absolute_at_lag']['1'][index]/c['measured']['absolute_at_lag']['0'][index]
                                for c in cells if c['measured']['absolute_at_lag']['0'][index] > 0]),
            filter_retained_lag1=stat([c[reference]['absolute_at_lag']['1'][index]/c[reference]['absolute_at_lag']['0'][index]
                                       for c in cells if c[reference]['absolute_at_lag']['0'][index] > 0]),
            action_disagreement={lag: stat([c['action_disagreement'][lag] for c in cells]) for lag in ('0', '1', '32')})
    return dict(seeds=[r['seed'] for r in reports], final_steps=sorted({r['checkpoint']['final_step'] for r in reports}),
                devices=sorted({d for r in reports for d in r['provenance']['devices']}),
                experiment_revisions=sorted({r['provenance']['experiment_revision'] for r in reports}),
                selected={m: stat([r['selected'][m] for r in reports]) for m in MODES},
                probes=probes, confirmation=confirmation, pulse=pulses)


def cell(s, digits=3):
    return 'n/a' if s is None else f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f}"


def span(s, digits=3):
    return 'n/a' if s is None else f"{s['min']:.{digits}f}–{s['max']:.{digits}f}"


def markdown(report):
    out = ['# Belief steering on the epsilon-0.02 ladder transformers', '',
           'Mean ± standard deviation across training seeds. Each seed contributes 32 confirmation episodes of 1,024 steps.',
           'Occupancy is the fraction of steps in reward state 3. "Actions changed" compares with the intact agent on the same inputs.', '']
    reference = report.get('reference', {}).get('variants', {})
    for label, block in report['populations'].items():
        for variant, summary in block.items():
            out += [f"## {label}: {NAMES[int(variant)]} ({len(summary['seeds'])} seeds)", '']
            lookup = {(c['condition'], c['site']): c for c in summary['confirmation']}
            ref = reference.get(variant, {})
            out += ['| Write (λ = 1) | Occupancy, after block 4 | Actions changed | Occupancy, final LayerNorm | Actions changed | Exact-filter reference occupancy | Reference actions changed |',
                    '|---|---|---|---|---|---|---|']
            intact, random = lookup[('intact', -1)], lookup[('random', -1)]
            out.append(f"| none (intact) | {cell(intact['occupancy'])} | 0 | | | {ref.get('intact', {}).get('occupancy', float('nan')):.3f} | 0 |")
            out.append(f"| random actions | {cell(random['occupancy'])} | {cell(random['action_change'])} | | | {ref.get('random', {}).get('occupancy', float('nan')):.3f} | {ref.get('random', {}).get('action_change', float('nan')):.3f} |")
            for mode in MODES:
                a, b = lookup[(f'curve_{mode}', 3)], lookup[(f'curve_{mode}', 4)]
                r = ref.get(mode, {})
                out.append(f"| {mode} | {cell(a['occupancy'])} | {cell(a['action_change'])} | {cell(b['occupancy'])} | {cell(b['action_change'])} | {r.get('occupancy', float('nan')):.3f} | {r.get('action_change', float('nan')):.3f} |")
            out += ['', '| Write after block 4 | λ | Shift / activation norm | Occupancy | Occupancy range over seeds | Actions changed | Agreement with exact-filter actions |',
                    '|---|---|---|---|---|---|---|']
            for name, condition in [(f'{m}, λ = 1', f'curve_{m}') for m in MODES]+[
                    (f'{m}, norm-matched to the rho_flip write', f'normmatched_{m}') for m in ('point', 'mirror')]+[
                    (f'orthogonal control for {m}', f'orthogonal_{m}') for m in MODES]:
                c = lookup.get((condition, 3))
                if c is not None:
                    out.append(f"| {name} | {cell(c['strength'], 2)} | {cell(c['relative_shift'], 2)} | {cell(c['occupancy'])} | {span(c['occupancy'])} | {cell(c['action_change'])} | {cell(c['reference_action_agreement'])} |")
            out += ['', '| Write (tuned λ) | λ | Block 1 | Block 2 | Block 3 | Block 4 | Final LN | Orthogonal control, block 4 | Orthogonal control, final LN |', '|---|---|---|---|---|---|---|---|---|']
            for mode in MODES:
                sites = ' | '.join(cell(lookup[(f'selected_{mode}', s)]['occupancy']) for s in range(5))
                controls = ' | '.join(cell(lookup[(f'orthogonal_{mode}', s)]['occupancy']) for s in (3, 4))
                out.append(f"| {mode} | {cell(summary['selected'][mode], 2)} | {sites} | {controls} |")
            out += ['', '| Probe R² | Block 1 | Block 2 | Block 3 | Block 4 | Final LN |', '|---|---|---|---|---|---|']
            for target in ('rho', 'delta'):
                out.append(f"| {target} | "+' | '.join(cell(s) for s in summary['probes'][target])+' |')
            out += ['', '| Pulse | Series | Lag 0 | Lag 1 | Lag 32 | Lag 33 | Lag 64 | Retained at lag 1 | Actions differing at lag 0 / 1 |', '|---|---|---|---|---|---|---|---|---|']
            for name, pulse in summary['pulse'].items():
                out.append(f"| {name} ({pulse['written']}) | agent | "+' | '.join(cell(pulse['measured'][lag]) for lag in LAGS)
                           + f" | {cell(pulse['retained_lag1'])} | {cell(pulse['action_disagreement']['0'])} / {cell(pulse['action_disagreement']['1'])} |")
                out.append(f"| | {pulse['reference'].replace('_', ' ')} | "+' | '.join(cell(pulse['filter'][lag]) for lag in LAGS)
                           + f" | {cell(pulse['filter_retained_lag1'])} | |")
            out.append('')
    return '\n'.join(out)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'belief_steering_summary.json')
    args = parser.parse_args()
    report = dict(schema=1, populations={})
    for label, study in (('2B-step parents', PARENT), ('4B-step continuations', CONTINUATION)):
        block = {str(v): summarize(load(study, v)) for v in (1, 2, 3) if load(study, v)}
        if block:
            report['populations'][label] = block
    reference = HERE/'results'/'belief_steering_reference.json'
    if reference.exists():
        report['reference'] = json.loads(reference.read_text())
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    args.output.with_suffix('.md').write_text(markdown(report))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
