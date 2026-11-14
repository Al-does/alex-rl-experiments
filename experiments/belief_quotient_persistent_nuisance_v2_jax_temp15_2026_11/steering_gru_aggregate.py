"""Cross-agent tables for the GRU steering and pulse battery (reads compact results)."""
import json
from pathlib import Path

STUDIES = {'no eps': 'belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11',
           'eps05': 'belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11'}
LAGS = ('0', '1', '2', '4', '8', '16', '32', '64', '128')


def load(root):
    out = []
    for label, study in STUDIES.items():
        for architecture in ('gru128', 'gru92'):
            for variant in (1, 2, 3):
                path = root/study/f'variant_{variant}_{architecture}'/'results'/'belief_steering_gru_seed42.json'
                if path.exists():
                    out.append((label, architecture, variant, json.loads(path.read_text())))
    return out


def fmt(x, digits=3):
    return '—' if x is None else f'{x:.{digits}f}'


def markdown(reports):
    lines = ['# GRU belief steering and single-write pulses (v2 environment, seed 42)', '',
             'One training seed per agent. Sites: hidden state of each GRU layer (carried to the next step), all hidden',
             'layers at once, and the final LayerNorm embedding (not carried). Targets come from the unsteered shadow network.', '',
             '## Probe R² (8-step exclusion): ρ / δ per site', '',
             '| Study | Arch | Variant | Sites (layer 1 … top, final norm) |', '|---|---|---:|---|']
    for label, arch, variant, r in reports:
        lines.append(f"| {label} | {arch} | {variant} | "+'; '.join(f"{fmt(s['rho'])} / {fmt(s['delta'])}" for s in r['probes']['exclusion_8'])+' |')
    lines += ['', '## Sustained steering at the final LayerNorm, λ = 1: occupancy (fraction of actions changed)', '',
              'Writes repeated at every step into a carried hidden state compound until the state overflows (`diverged` in the',
              'per-agent results), for belief and control writes alike, so hidden-state sites are not tabulated.', '',
              '| Study | Arch | Variant | Intact | Random | Target | Final norm | Orthogonal control | Hidden-site runs diverged |',
              '|---|---|---:|---:|---:|---|---:|---:|---:|']
    for label, arch, variant, r in reports:
        top = len(r['probes']['exclusion_8'])-2
        table = {(c['condition'], c['site'], c['mode']): c['metrics'] for c in r['confirmation']}
        cell = lambda m: f"{fmt(m['occupancy'])} ({m['action_change']*100:.0f}%)"
        for mode in ('point', 'mirror', 'rho_flip', 'rho_point'):
            lines.append(f"| {label} | {arch} | {variant} | {fmt(table[('intact', -1, 'point')]['occupancy'])} | {fmt(table[('random', -1, 'point')]['occupancy'])} | {mode} | "
                         f"{cell(table[('curve_'+mode, top+1, mode)])} | {fmt(table[('orthogonal_'+mode, top+1, mode)]['occupancy'])} | "
                         f"{sum(c['diverged'] for c in r['confirmation'] if c['mode'] == mode and c['strength'] == 1 and c['site'] not in (-1, top+1))}/"
                         f"{sum(1 for c in r['confirmation'] if c['mode'] == mode and c['strength'] == 1 and c['site'] not in (-1, top+1))} |")
    lines += ['', '## Single-write pulse into all hidden layers: mean |change| in the written coordinate by lag', '',
              'Measured = decoded at the top hidden layer; filter = exact filter restarted from the written belief; r / slope = agreement of the',
              'signed measured change with the filter\'s across the 96 branches; control = decoder-orthogonal write of the same norm.', '',
              '| Study | Arch | Variant | Pulse | Series | '+' | '.join(LAGS)+' |', '|---|---|---:|---|---|'+'---:|'*len(LAGS)]
    for label, arch, variant, r in reports:
        for mode, coordinate in (('mirror', 1), ('rho_flip', 0)):
            c = r['pulse']['conditions'][f'{mode}_all_layers']
            o = r['pulse']['conditions'][f'{mode}_all_layers_orthogonal']
            series = [('measured', {k: v[coordinate] for k, v in c['measured']['absolute'].items()}),
                      ('filter', {k: v[coordinate] for k, v in c['full_filter']['absolute'].items()}),
                      ('r', c['tracking']['correlation']), ('slope', c['tracking']['slope']),
                      ('action disagreement', c['action_disagreement']),
                      ('control measured', {k: v[coordinate] for k, v in o['measured']['absolute'].items()}),
                      ('control disagreement', o['action_disagreement'])]
            if 'quotient_filter' in c and coordinate == 0:
                series.insert(2, ('quotient filter', {k: v[0] for k, v in c['quotient_filter']['absolute'].items()}))
            for name, values in series:
                lines.append(f'| {label} | {arch} | {variant} | {mode} | {name} | '+' | '.join(fmt(values.get(k), 4) for k in LAGS)+' |')
    lines += ['', '## Single-layer pulses: measured |change| in the written coordinate by lag', '',
              '| Study | Arch | Variant | Pulse | Layer | '+' | '.join(LAGS)+' |', '|---|---|---:|---|---|'+'---:|'*len(LAGS)]
    for label, arch, variant, r in reports:
        for mode, coordinate in (('mirror', 1), ('rho_flip', 0)):
            for key, c in r['pulse']['conditions'].items():
                if key.startswith(mode+'_layer') and not key.endswith('orthogonal'):
                    lines.append(f"| {label} | {arch} | {variant} | {mode} | {key.split('_layer')[1]} | "
                                 +' | '.join(fmt(c['measured']['absolute'].get(k, [None, None])[coordinate], 4) for k in LAGS)+' |')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    reports = load(root)
    out = Path(__file__).parent/'results'/'belief_steering_gru_twelve_agents.md'
    out.write_text(markdown(reports))
    print(f'{len(reports)} agents -> {out}')
