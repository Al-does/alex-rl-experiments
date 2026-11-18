"""Aggregate epsilon-0.02 reward-factor steering reports over training seeds.

Each write moves one factor's decoded belief and holds the other factor's
decoded marginal. Metrics are reported per factor: occupancy of that factor's
reward state 3 and the fraction of steps on which that factor's action
component differs from the intact agent on the same inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.steering_reference import simulate
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import Policy
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11.steering_factor import KINDS, MODES, SITES, targets

HERE = Path(__file__).resolve().parent
NAMES = dict(reward_both='Joint Reward', reward_factor_1='Unrewarded Factor 2')
SITE_NAMES = ('block 1', 'block 2', 'block 3', 'block 4', 'final LN')
FIELDS = ('occupancy', 'action_change', 'reference_action_agreement')


def load(condition):
    folder = HERE/f'{condition}_transformer'/'results'
    reports = [json.loads(p.read_text()) for p in sorted(folder.glob('belief_steering_factor_seed*.json'))]
    return sorted(reports, key=lambda r: r['seed'])


def stat(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return None
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else 0.,
                min=float(values.min()), max=float(values.max()), n=int(len(values)))


def key(c):
    return c['condition'], c['map'], c['factor'], c['mode'], c['site']


def summarize(reports):
    tables = [{key(c): c for c in r['confirmation']} for r in reports]
    confirmation = []
    for k in sorted({k for t in tables for k in t}, key=str):
        found = [t[k] for t in tables if k in t]
        entry = dict(zip(('condition', 'map', 'factor', 'mode', 'site'), k), seeds=len(found),
                     strength=stat([c['strength'] for c in found]),
                     relative_shift=stat([c['metrics']['relative_shift'] for c in found]))
        for field in FIELDS:
            entry[field] = [stat([c['metrics'][field][f] for c in found]) for f in (0, 1)]
        confirmation.append(entry)
    probes = {kind: {target: [stat([r['probes'][kind][site][target] for r in reports]) for site in range(SITES)]
                     for target in ('rho_1', 'delta_1', 'rho_2', 'delta_2')} for kind in KINDS}
    selected = {k: dict(found=int(sum(r['selected'][k] is not None for r in reports)),
                        strength=stat([r['selected'][k] for r in reports])) for k in reports[0]['selected']}
    return dict(seeds=[r['seed'] for r in reports], final_steps=sorted({r['checkpoint']['final_step'] for r in reports}),
                devices=sorted({d for r in reports for d in r['provenance']['devices']}),
                experiment_revisions=sorted({r['provenance']['experiment_revision'] for r in reports}),
                probes=probes, selected=selected, confirmation=confirmation)


def reference():
    """One rewarded factor in isolation: exact filter and grid policy acting on the written belief."""
    policy = Policy(3, n=100, gamma=.995, environment=environment)
    entry = dict(convergence=policy.convergence, intact=simulate(policy), random=simulate(policy, random=True))
    entry.update({mode: simulate(policy, lambda b, i=i: targets(b, np)[i]) for i, mode in enumerate(MODES)})
    return entry


def cell(s, digits=3):
    return 'n/a' if s is None else f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f}"


def markdown(report):
    out = ['# Single-factor belief steering on the epsilon-0.02 reward-factor transformers', '',
           'Mean ± standard deviation across training seeds; 32 confirmation episodes of 1,024 steps per seed.',
           'Writes use strength λ = 1. The single-factor reference is one rewarded factor under the exact filter.', '']
    ref = report['reference']
    out += ['| Single-factor reference | Occupancy | Actions changed |', '|---|---|---|']
    out += [f"| {k} | {ref[k]['occupancy']:.3f} | {ref[k]['action_change']:.3f} |" for k in ('intact', 'random')+MODES] + ['']
    for condition, summary in report['conditions'].items():
        lookup = {tuple(c[k] for k in ('condition', 'map', 'factor', 'mode', 'site')): c for c in summary['confirmation']}
        out += [f"## {NAMES[condition]} ({len(summary['seeds'])} seeds)", '']
        for kind in KINDS:
            for site in (3, 4):
                out += [f'### {kind} map, after {SITE_NAMES[site]}', '',
                        '| Write | Factor 1 occupancy | Factor 2 occupancy | Factor 1 actions changed | Factor 2 actions changed | Shift / activation norm |',
                        '|---|---|---|---|---|---|']
                for label, k in (('none (intact)', ('intact', None, None, None, -1)), ('random actions', ('random', None, None, None, -1))):
                    c = lookup[k]
                    out.append(f"| {label} | {cell(c['occupancy'][0])} | {cell(c['occupancy'][1])} | {cell(c['action_change'][0])} | {cell(c['action_change'][1])} | |")
                for condition_name in ('write', 'orthogonal'):
                    for factor in (1, 2):
                        for mode in MODES:
                            c = lookup.get((condition_name, kind, factor, mode, site))
                            if c:
                                out.append(f"| {condition_name}: factor {factor} {mode} | {cell(c['occupancy'][0])} | {cell(c['occupancy'][1])} | {cell(c['action_change'][0])} | {cell(c['action_change'][1])} | {cell(c['relative_shift'], 2)} |")
                out.append('')
        out += ['### Steered-factor occupancy by site (marginal map)', '', '| Write | '+' | '.join(SITE_NAMES)+' |', '|---|'+'---|'*SITES]
        for factor in (1, 2):
            for mode in MODES:
                out.append(f'| factor {factor} {mode} | '+' | '.join(cell(lookup[('write', 'marginal', factor, mode, s)]['occupancy'][factor-1]) for s in range(SITES))+' |')
        out += ['', '### Probe R² (marginal map)', '', '| Target | '+' | '.join(SITE_NAMES)+' |', '|---|'+'---|'*SITES]
        for target in ('rho_1', 'delta_1', 'rho_2', 'delta_2'):
            out.append(f'| {target} | '+' | '.join(cell(s) for s in summary['probes']['marginal'][target])+' |')
        out.append('')
    return '\n'.join(out)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'belief_steering_summary.json')
    args = parser.parse_args()
    report = dict(schema=1, reference=reference(), conditions={c: summarize(load(c)) for c in NAMES if load(c)})
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    args.output.with_suffix('.md').write_text(markdown(report))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
