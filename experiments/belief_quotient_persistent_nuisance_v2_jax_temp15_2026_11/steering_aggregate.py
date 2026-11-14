"""Aggregate six explicit per-agent inputs; never mix smoke and scientific reports."""
import argparse
import json
from pathlib import Path

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
    table, pulse, comparisons = [], [], []
    for r in reports:
        agent = dict(study=r['study'], variant=r['variant'], sha256=r['checkpoint']['sha256'])
        for item in r['confirmation']:
            table.append(dict(**agent, **item))
        for name, condition in r['pulse']['conditions'].items():
            pulse.append(dict(**agent, condition=name, **condition))
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
    return dict(schema=1, table=table, pulse=pulse, epsilon_minus_no_epsilon=comparisons,
                inputs=[dict(path=str(p), provenance=r['provenance'], checkpoint=r['checkpoint']) for p, r in zip(paths, reports)],
                caveat='One training seed per condition; uncertainties describe matched evaluation episodes, not training-seed variability.')


def np_difference(left, right):
    import numpy as np
    delta = np.asarray(right)-left
    return dict(occupancy_difference=float(delta.mean()), paired_episode_se=float(delta.std(ddof=1)/np.sqrt(len(delta))))


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
        lines.append(f"| {'eps05' if 'eps05' in item['study'] else 'no eps'} | {item['variant']} | {item['condition']} | {item['site']} | {item['strength']:.4g} | {m['occupancy']:.6f} | {m['noop_given_reward']} | {m['action_change']:.5f} | {m['shift_norm']:.4g} | {m['relative_shift']:.4g} |")
    lines += ['', '## Pulse decay summary', '', '| Study | Variant | Condition | Measured ρ multiplier | Measured δ multiplier | Full filter ρ / δ |', '|---|---:|---|---:|---:|---|']
    for p in report['pulse']:
        measured, predicted = p['measured']['decay'], p['full_filter']['decay']
        lines.append(f"| {'eps05' if 'eps05' in p['study'] else 'no eps'} | {p['variant']} | {p['condition']} | {measured['rho']['multiplier']} | {measured['delta']['multiplier']} | {predicted['rho']['multiplier']} / {predicted['delta']['multiplier']} |")
    lines += ['', '## Epsilon minus no-epsilon paired occupancy differences', '', '| Variant | Condition | Site | Difference | Paired episode SE |', '|---:|---|---:|---:|---:|']
    for c in report['epsilon_minus_no_epsilon']:
        if c['condition'].startswith('selected_') or c['condition'].startswith('normmatched_') or c['condition'] == 'intact':
            lines.append(f"| {c['variant']} | {c['condition']} | {c['site']} | {c['occupancy_difference']:.6f} | {c['paired_episode_se']:.6f} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
