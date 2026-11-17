"""Extract immutable REINFORCE references and tabulate confirmed steering results."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .analyze import REPO, ROOT, SEEDS, read, write
from .vendor_steering import LADDER, MOVING
from .details import DETAILS, ensure_details, records

import numpy as np

RESULTS = ROOT/'results/steering'
SITES = ('Block 1', 'Block 2', 'Block 3', 'Block 4', 'Final norm')
NAMES = {1: 'Trivial Quotient', 2: 'Reward-State Quotient', 3: 'Full-State Quotient'}


def references(refresh=False):
    output = DETAILS/'steering/reinforce_reference.json'
    if output.exists() and not refresh:
        return read(output)
    inputs = []

    def load(rev, path):
        blob = subprocess.check_output(['git', '-C', str(REPO), 'show', rev+':'+path])
        inputs.append(dict(revision=rev, path=path, sha256=hashlib.sha256(blob).hexdigest()))
        return json.loads(blob)

    ladder = []
    moving = []
    for seed in range(42, 57):
        for variant in (1, 2, 3):
            path = f'experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/variant_{variant}_transformer/results/belief_steering_v2_seed{seed}_stable.json'
            r = load(LADDER, path)
            ladder.append({k: r[k] for k in ('seed', 'variant', 'confirmation', 'selected', 'norm_matching', 'pulse', 'probes')})
        path = f'experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/results/eps02_ctx32_2026_11__reward_both__seed{seed}.json'
        r = load(MOVING, path)
        moving.append(dict(seed=seed, confirmation=r['rewards']['moving']['confirmation'],
                           selected=r['rewards']['moving']['strength_tuning']['selected_strength']))
    reference = load(LADDER, 'experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/results/belief_steering_reference.json')
    result = dict(inputs=inputs, ladder=ladder, moving=moving, numerical_ladder_reference=reference)
    write(output, result)
    return result


def ladder_rows(report):
    rows = {}
    for item in report['confirmation']:
        key = f"{item['condition']}@{item['site']}"
        if key in rows:
            raise ValueError(f'duplicate ladder result {key}')
        rows[key] = dict(item['metrics'], strength=item['strength'])
    if not rows:
        raise ValueError('empty ladder results')
    return rows


def moving_rows(confirmation):
    result = {k: v for k, v in confirmation.items() if k not in ('comparisons', 'matched_null')}
    result.update(confirmation['comparisons'])
    result.update({f'null_{k}': v for k, v in confirmation['matched_null'].items()})
    return result


def aggregate(reports, rows_fn):
    if not reports:
        raise ValueError('empty agent group')
    per_seed = {str(r.get('seed', r.get('adapter_provenance', {}).get('training_seed'))): rows_fn(r) for r in reports}
    if len(per_seed) != len(reports):
        raise ValueError('duplicate training seeds')
    common = set.intersection(*(set(r) for r in per_seed.values()))
    means = {}
    for k in sorted(common):
        keys = set.intersection(*(set(r[k]) for r in per_seed.values()))
        values = {}
        for name in sorted(keys):
            data = [r[k][name] for r in per_seed.values()]
            if all(isinstance(v, (int, float, list)) for v in data):
                array = np.asarray(data, dtype=float)
                if np.isfinite(array).all():
                    values[name] = array.mean(axis=0).tolist()
        means[k] = values
    return dict(n=len(reports), means=means, per_seed=per_seed)


def pulse_rows(label, runs, ref):
    if label.startswith('REINFORCE'):
        selected = [r for r in ref['ladder'] if NAMES[r['variant']] in label and (label.startswith('REINFORCE (15)') or r['seed'] in SEEDS)]
    else:
        algorithm = 'ppo' if label.startswith('PPO') else 'sac_autoalpha'
        selected = [r for r in runs if r['adapter_provenance']['kind'] == 'ladder' and r['adapter_provenance']['algorithm'] == algorithm and NAMES[r['variant']] in label]
    return [dict(group=label, condition=condition, n=len(selected),
        lag0=np.mean([r['pulse']['conditions'][condition]['measured']['absolute_at_lag']['0'] for r in selected], axis=0).tolist(),
        lag1=np.mean([r['pulse']['conditions'][condition]['measured']['absolute_at_lag']['1'] for r in selected], axis=0).tolist(),
        last_nonzero_max=max(r['pulse']['conditions'][condition]['measured']['last_nonzero_lag'] or 0 for r in selected),
        full_filter_lag128=np.mean([r['pulse']['conditions'][condition]['full_filter']['absolute_at_lag']['128'] for r in selected], axis=0).tolist())
        for condition in selected[0]['pulse']['conditions']]


def compact_summary(groups, runs, ref):
    result = {}
    for kind, names in (('ladder', ('occupancy', 'action_change', 'shift_norm', 'relative_shift')),
                        ('moving', ('total_reward',))):
        result[kind] = [dict(group=label, condition=condition, n=group['n'],
                            **{name: value for name, value in metrics.items() if name in names})
                        for label, group in groups[kind].items() for condition, metrics in group['means'].items()]
    result['pulses'] = [row for label in groups['ladder'] for row in pulse_rows(label, runs, ref)]
    result['numerical_ladder_reference'] = [dict(condition=NAMES[int(variant)],
        **{key: values[key]['occupancy'] for key in ('intact', 'random', 'point', 'mirror', 'rho_flip', 'rho_point')})
        for variant, values in ref.get('numerical_ladder_reference', {}).get('variants', {}).items()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references-only', action='store_true')
    args = parser.parse_args()
    ensure_details()
    ref = references(refresh=args.references_only)
    if args.references_only:
        print('Extracted 45 ladder and 15 moving REINFORCE references, plus numerical ladder reference')
        return
    runs = [read(p) for p in sorted((DETAILS/'steering/agents').glob('*.json'))]
    if len(runs) != 40 or any(r['adapter_provenance']['smoke'] for r in runs):
        raise ValueError('expected exactly 30 ladder plus 10 moving full agents')
    groups = {'ladder': {}, 'moving': {}}
    for label in ('REINFORCE (15)', 'REINFORCE (5)', 'PPO (5)', 'SAC auto-alpha (5)'):
        for kind in groups:
            if label.startswith('REINFORCE'):
                selected = [r for r in ref[kind] if label.endswith('(15)') or r['seed'] in SEEDS]
            else:
                algorithm = 'ppo' if label.startswith('PPO') else 'sac_autoalpha'
                selected = [r for r in runs if r['adapter_provenance']['kind'] == kind and r['adapter_provenance']['algorithm'] == algorithm]
            if kind == 'ladder':
                for variant in (1, 2, 3):
                    group = [r for r in selected if r['variant'] == variant]
                    groups[kind][f'{label}: {NAMES[variant]}'] = aggregate(group, ladder_rows)
            else:
                groups[kind][label] = aggregate(selected, lambda r: moving_rows(r.get('confirmation', r.get('rewards', {}).get('moving', {}).get('confirmation'))))
    write(DETAILS/'aggregates/steering.json', groups)
    records(RESULTS/'summary.json', compact_summary(groups, runs, ref))
    lines = ['# Steering algorithm generality', '', '## Ladder: full strength after block 4', '',
        'Reward-state occupancy; parentheses are action-change fractions. Intact/random are controls.', '',
        '| Algorithm / condition | Intact | Random | delta point | delta mirror | rho flip | rho point |', '|---|---:|---:|---:|---:|---:|---:|']
    for label, group in groups['ladder'].items():
        m = group['means']
        cells = [f"{m[k]['occupancy']:.4f}" for k in ('intact@-1', 'random@-1')]
        cells += [f"{m['curve_'+mode+'@3']['occupancy']:.4f} ({m['curve_'+mode+'@3']['action_change']:.3f})" for mode in ('point', 'mirror', 'rho_flip', 'rho_point')]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '## Numerical exact-filter ladder reference (256 episodes)', '',
              '| Condition | Intact | Random | delta point | delta mirror | rho flip | rho point |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for variant, name in NAMES.items():
        m = ref['numerical_ladder_reference']['variants'][str(variant)]
        lines.append('| '+name+' | '+' | '.join(f"{m[k]['occupancy']:.4f}" for k in ('intact', 'random', 'point', 'mirror', 'rho_flip', 'rho_point'))+' |')
    lines += ['', '## Ladder: tuning-selected steering and decoder-orthogonal controls', '', '| Algorithm / condition | Mode | Block 4 | Null block 4 | Final norm | Null final norm |', '|---|---|---:|---:|---:|---:|']
    for label, group in groups['ladder'].items():
        for mode in ('point', 'mirror', 'rho_flip', 'rho_point'):
            m = group['means']
            lines.append('| '+label+' | '+mode+' | '+' | '.join(f"{m[k]['occupancy']:.4f}" for k in (f'selected_{mode}@3', f'orthogonal_{mode}@3', f'selected_{mode}@4', f'orthogonal_{mode}@4'))+' |')
    lines += ['', '## Ladder sites: tuning-selected occupancy', '', '| Algorithm / condition | Mode | Block 1 | Block 2 | Block 3 | Block 4 | Final norm |', '|---|---|---:|---:|---:|---:|---:|']
    for label, group in groups['ladder'].items():
        for mode in ('point', 'mirror', 'rho_flip', 'rho_point'):
            lines.append('| '+label+' | '+mode+' | '+' | '.join(f"{group['means'][f'selected_{mode}@{site}']['occupancy']:.4f}" for site in range(5))+' |')
    lines += ['', '## Ladder: norm-matched delta after block 4', '',
              'Confirmation occupancy and achieved mean shift norms relative to full-strength rho flip. Calibration uses tuning data only.', '',
              '| Algorithm / condition | delta point | delta mirror | rho flip | Point/rho norm | Mirror/rho norm |',
              '|---|---:|---:|---:|---:|---:|']
    for label, group in groups['ladder'].items():
        m = group['means']
        keys = ('normmatched_point@3', 'normmatched_mirror@3', 'curve_rho_flip@3')
        cells = [f"{m[k]['occupancy']:.4f}" for k in keys]
        cells += [f"{m[k]['shift_norm']/m['curve_rho_flip@3']['shift_norm']:.3f}" for k in keys[:2]]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '## Single-write pulses after block 1', '',
              'Mean absolute decoded-coordinate response at lags 0/1, maximum last nonzero lag across training seeds, and the full-filter response at lag 128. Mirror measures delta; flip measures rho. Blocks 2/3 are retained in agent reports.', '',
              '| Algorithm / condition | Pulse | Lag 0 | Lag 1 | Last nonzero (max) | Full filter lag 128 |',
              '|---|---|---:|---:|---:|---:|']
    for label in groups['ladder']:
        if label.startswith('REINFORCE'):
            selected = [r for r in ref['ladder'] if NAMES[r['variant']] in label and (label.startswith('REINFORCE (15)') or r['seed'] in SEEDS)]
        else:
            algorithm = 'ppo' if label.startswith('PPO') else 'sac_autoalpha'
            selected = [r for r in runs if r['adapter_provenance']['kind'] == 'ladder' and r['adapter_provenance']['algorithm'] == algorithm and NAMES[r['variant']] in label]
        for mode, coordinate in (('mirror', 1), ('rho_flip', 0)):
            pulses = [r['pulse']['conditions'][mode+'_block1'] for r in selected]
            values = [np.mean([p['measured']['absolute_at_lag'][str(lag)][coordinate] for p in pulses]) for lag in (0, 1)]
            cutoff = max(p['measured']['last_nonzero_lag'] or 0 for p in pulses)
            full = np.mean([p['full_filter']['absolute_at_lag']['128'][coordinate] for p in pulses])
            lines.append('| '+label+' | '+mode+' | '+f'{values[0]:.6f} | {values[1]:.6f} | {cutoff} | {full:.6f} |')
    lines += ['', '## Moving reward: block 4, discounted map', '', '| Algorithm | Intact | Random | Matched null (joint) | Factored | Joint | Exact-belief map |', '|---|---:|---:|---:|---:|---:|---:|']
    moving_keys = ('intact', 'random', 'null_discounted_joint', 'discounted_marginal_block_4', 'discounted_joint_block_4', 'exact_belief_discounted_map')
    for label, group in groups['moving'].items():
        lines.append('| '+label+' | '+' | '.join(f"{group['means'][k]['total_reward']:.4f}" for k in moving_keys)+' |')
    lines += ['', '## Moving reward by intervention site', '', '| Algorithm | Target | Block 1 | Block 2 | Block 3 | Block 4 | Final norm |', '|---|---|---:|---:|---:|---:|---:|']
    for label, group in groups['moving'].items():
        for target in ('marginal', 'joint'):
            keys = [f'discounted_{target}_block_{i}' for i in range(1, 5)]+[f'discounted_{target}_post_final_norm']
            lines.append('| '+label+' | '+('Factored' if target == 'marginal' else 'Joint')+' | '+' | '.join(f"{group['means'][k]['total_reward']:.4f}" for k in keys)+' |')
    lines += ['', '## Per-seed ladder: block 4, full strength', '', '| Algorithm / condition | Seed | Intact | Random | delta point | delta mirror | rho flip | rho point |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, group in groups['ladder'].items():
        if group['n'] == 15:
            continue
        for seed, values in group['per_seed'].items():
            keys = ('intact@-1', 'random@-1', 'curve_point@3', 'curve_mirror@3', 'curve_rho_flip@3', 'curve_rho_point@3')
            lines.append('| '+label+' | '+seed+' | '+' | '.join(f"{values[k]['occupancy']:.4f}" for k in keys)+' |')
    lines += ['', '## Per-seed moving reward: block 4, discounted map', '', '| Algorithm | Seed | Intact | Random | Null (joint) | Factored | Joint | Exact-belief map |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, group in groups['moving'].items():
        if group['n'] == 15:
            continue
        for seed, values in group['per_seed'].items():
            lines.append('| '+label+' | '+seed+' | '+' | '.join(f"{values[k]['total_reward']:.4f}" for k in moving_keys)+' |')
    lines += ['', 'Aggregate occupancy/reward/control and pulse metrics are in summary.json. Full per-seed/site metrics, myopic-map results, phase/component rewards and agent reports are in the verified B2 archive (../details_manifest.json). Numerical references are not certified optima.', '']
    (RESULTS/'tables.md').write_text('\n'.join(lines))
    print('Assembled 30 ladder and 10 moving agents; 45/15 REINFORCE references; summary.json and tables.md')
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    main()
