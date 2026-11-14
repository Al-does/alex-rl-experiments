"""Aggregate six explicit per-agent inputs; never mix smoke and scientific reports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering import STUDIES


def aggregate(paths):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    expected = {(s, v) for s in STUDIES for v in (1, 2, 3)}
    keys = [(r['study'], r['variant']) for r in reports]
    if len(keys) != 6 or set(keys) != expected or any(r['smoke'] for r in reports):
        raise ValueError('require exactly one non-smoke report for each of six agents')
    protocol = reports[0]['protocol']
    if any(r['protocol'] != protocol for r in reports):
        raise ValueError('incompatible evaluation protocols')
    table, pulse, comparisons, probes, calibration = [], [], [], [], []
    for r in reports:
        agent = dict(study=r['study'], variant=r['variant'], sha256=r['checkpoint']['sha256'])
        for item in r['confirmation']:
            table.append(dict(**agent, **item))
        for name, condition in r['pulse']['conditions'].items():
            pulse.append(dict(**agent, condition=name, **condition))
        for exclusion, sites in r.get('probes', {}).items():
            for site, scores in enumerate(sites):
                probes.append(dict(**agent, exclusion=exclusion, site=site, **scores))
        for condition, values in r.get('norm_matching', {}).items():
            site = int(condition.split('site')[-1])
            reference = next((x['metrics']['shift_norm'] for x in r['confirmation']
                              if x['condition'] == 'curve_rho_flip' and x['site'] == site and x['strength'] == 1), None)
            ratio = values.get('confirmation_achieved_norm', 0)/reference if reference else None
            calibration.append(dict(**agent, condition=condition, **values,
                confirmation_reference_norm=reference, confirmation_norm_ratio=ratio))
    for v in (1, 2, 3):
        a, b = [next(r for r in reports if r['study'] == s and r['variant'] == v) for s in STUDIES]
        if len(a['confirmation']) != len(b['confirmation']):
            raise ValueError('confirmation condition counts differ')
        for left, right in zip(a['confirmation'], b['confirmation']):
            if (left['condition'], left['site']) != (right['condition'], right['site']):
                raise ValueError('confirmation conditions/order differ')
            differences = np_difference(left['metrics']['episode_occupancy'], right['metrics']['episode_occupancy'])
            comparisons.append(dict(variant=v, condition=left['condition'], site=left['site'],
                no_epsilon_lambda=left['strength'], epsilon_lambda=right['strength'], **differences))
    return dict(schema=2, table=table, pulse=pulse, probes=probes, norm_matching=calibration,
                aggregation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), epsilon_minus_no_epsilon=comparisons,
                inputs=[dict(path=str(p), provenance=r['provenance'], checkpoint=r['checkpoint']) for p, r in zip(paths, reports)],
                caveat='One training seed per condition; uncertainties describe matched evaluation episodes, not training-seed variability.')


def np_difference(left, right):
    delta = np.asarray(right)-left
    return dict(occupancy_difference=float(delta.mean()), paired_episode_se=float(delta.std(ddof=1)/np.sqrt(len(delta))))


def site_label(site):
    return 'post-LN' if site == 4 else f'block {site+1}' if site >= 0 else 'baseline'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs=6)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError(args.output)
    report = aggregate(args.inputs)
    if args.predictions:
        report['numerical_references'] = json.loads(args.predictions.read_text())
    lines = ['# Six-agent v2 steering comparison', '', report['caveat'], '',
             '| Study | Variant | Intervention | Site | λ | Occupancy | Noop given S3 | Action change | Shift | Relative shift |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in report['table']:
        if item['condition'].startswith('curve_') and item['strength'] != 1:
            continue
        m = item['metrics']
        lines.append(f"| {'eps05' if 'eps05' in item['study'] else 'no eps'} | {item['variant']} | {item['condition']} | {site_label(item['site'])} | {item['strength']:.4g} | {m['occupancy']:.6f} ± {m['occupancy_se']:.6f} | {m['noop_given_reward']} | {m['action_change']:.5f} | {m['shift_norm']:.4g} | {m['relative_shift']:.4g} |")
    lines += ['', '## Coordinate probe R²', '', '| Study | Variant | Exclusion | Site | b1 | b2 | ρ | δ |', '|---|---:|---|---|---:|---:|---:|---:|']
    for p in report['probes']:
        scores = ' | '.join('undefined' if p[k] is None else f'{p[k]:.6f}' for k in ('b1', 'b2', 'rho', 'delta'))
        lines.append(f"| {'eps05' if 'eps05' in p['study'] else 'no eps'} | {p['variant']} | {p['exclusion']} | {site_label(p['site'])} | {scores} |")
    lines += ['', '## Norm-calibration diagnostics', '',
        'The `normmatched` condition name denotes attempted tuning-only scalar calibration, NOT a guarantee. Compare achieved confirmation norm with the same-agent/site rho_flip λ=1 confirmation norm; ratios far from one are failed matches. Discontinuous closed-loop policies can also defeat tuning bisection itself. These failures must not be treated as equal-norm evidence.', '',
        '| Study | Variant | Condition | Effective λ | Target tuning norm | Achieved tuning norm | Achieved confirmation norm | Confirmation rho_flip reference | Confirmation norm ratio | Bracketed |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for p in report['norm_matching']:
        lines.append(f"| {'eps05' if 'eps05' in p['study'] else 'no eps'} | {p['variant']} | {p['condition']} | {p['effective_lambda']:.5g} | {p['target_tune_norm']:.5g} | {p['tune_achieved_norm']:.5g} | {p['confirmation_achieved_norm']:.5g} | {p['confirmation_reference_norm']} | {p['confirmation_norm_ratio']} | {p['bracketed']} |")
    lines += ['', '## Pulse decay summary', '', '| Study | Variant | Condition | Measured ρ multiplier | Measured δ multiplier | Full filter ρ / δ |', '|---|---:|---|---:|---:|---|']
    for p in report['pulse']:
        measured, predicted = p['measured']['decay'], p['full_filter']['decay']
        lines.append(f"| {'eps05' if 'eps05' in p['study'] else 'no eps'} | {p['variant']} | {p['condition']} | {measured['rho']['multiplier']} | {measured['delta']['multiplier']} | {predicted['rho']['multiplier']} / {predicted['delta']['multiplier']} |")
    lines += ['', '### Pulse amplitudes and finite-band support', '',
        'The reported amplitude is mean absolute effect on δ for mirror and ρ for rho_flip. Tail rates exclude lag 0 and are not retention fractions. Post-block pulses have at most 96/64/32 downstream cache steps for blocks 1/2/3; 128 is the input receptive field, not post-block-1 pulse support.', '',
        '| Study | Variant | Condition | Lag 0 | Lag 1 | Lag 33 | Lag 64 | Lag 96 | Lag 128 | Last nonzero lag (either coordinate) | TV 0 → 1 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for p in report['pulse']:
        absolute = np.asarray(p['measured']['absolute'])
        coordinate = 0 if p['condition'].startswith('rho_flip') else 1
        values = ' | '.join(f'{absolute[k, coordinate]:.6g}' if k < len(absolute) else 'n/a' for k in (0, 1, 33, 64, 96, 128))
        active = np.flatnonzero(absolute.max(-1) > 0)
        lines.append(f"| {'eps05' if 'eps05' in p['study'] else 'no eps'} | {p['variant']} | {p['condition']} | {values} | {int(active[-1]) if len(active) else 'none'} | {p['tv'][0]:.5g} → {p['tv'][1]:.5g} |")
    lines += ['', '### Quotient-filter ρ decay (variant 2)', '', '| Study | Condition | Multiplier | Fit R² |', '|---|---|---:|---:|']
    for p in report['pulse']:
        if 'quotient_filter' in p:
            fit = p['quotient_filter']['decay']['rho']
            lines.append(f"| {'eps05' if 'eps05' in p['study'] else 'no eps'} | {p['condition']} | {fit['multiplier']} | {fit['r2']} |")
    lines += ['', '## Epsilon minus no-epsilon paired occupancy differences', '', '| Variant | Condition | Site | Difference | Paired episode SE |', '|---:|---|---:|---:|---:|']
    for c in report['epsilon_minus_no_epsilon']:
        if c['condition'].startswith('selected_') or c['condition'].startswith('normmatched_') or c['condition'] == 'intact':
            lines.append(f"| {c['variant']} | {c['condition']} | {c['site']} | {c['occupancy_difference']:.6f} | {c['paired_episode_se']:.6f} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
