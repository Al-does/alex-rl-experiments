"""One-agent offline steering and pulse battery for the GRU arms. No training."""
import argparse
import json
from pathlib import Path
import platform
import time

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_history_jax_2026_11.shared import model_spec
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import steering_gru as gru
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering import STUDIES, restore, revision, sha256
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.environment import ladder_process
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import AFFINE_RCOND, LAMBDAS, MODES, metrics
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_storage import local_reference, upload_full

TAG = 'belief_steering_gru_seed42'
LAGS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def at_lags(values):
    return {str(k): values[k] for k in LAGS if k < len(values)}


def finite(value):
    """Undefined statistics (e.g. a correlation over constant values) become null."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def dump_compact(summary):
    """One line per condition, so the tracked summary stays a few hundred lines."""
    def block(value):
        if isinstance(value, list):
            return '[\n'+',\n'.join('    '+json.dumps(v, allow_nan=False) for v in value)+'\n  ]'
        if isinstance(value, dict) and 'conditions' in value:
            head = {k: v for k, v in value.items() if k != 'conditions'}
            rows = ',\n'.join(f'      {json.dumps(k)}: '+json.dumps(v, allow_nan=False) for k, v in value['conditions'].items())
            return json.dumps(head, allow_nan=False)[:-1]+', "conditions": {\n'+rows+'\n    }}'
        return json.dumps(value, allow_nan=False)
    return '{\n'+',\n'.join(f'  {json.dumps(k)}: {block(v)}' for k, v in summary.items())+'\n}\n'


def compact_pulse(result):
    out = dict(tv=at_lags(result['tv']), action_disagreement=at_lags(result['action_disagreement']),
               relative_shift=result['relative_shift'])
    for key in ('measured', 'full_filter', 'quotient_filter'):
        if key in result:
            out[key] = dict(absolute=at_lags(result[key]['absolute']), signed=at_lags(result[key]['signed']),
                            decay=result[key]['decay'])
    if 'tracking' in result:
        out['tracking'] = {k: at_lags(v) for k, v in result['tracking'].items()}
    return out


def run(study, variant, architecture, smoke=False):
    root = Path(__file__).resolve().parents[2]
    leaf = root/'experiments'/study/f'variant_{variant}_{architecture}'
    raw = leaf/'artifacts'/(TAG+('_smoke' if smoke else ''))
    raw.mkdir(parents=True, exist_ok=True)
    path, summary_path = raw/f'{TAG}.json', leaf/'results'/f'{TAG}.json'
    if path.exists() or (not smoke and summary_path.exists()):
        raise FileExistsError(f'refusing to overwrite {path}')
    start = time.perf_counter()
    checkpoint_path, checkpoint = restore(leaf, study, raw)
    process = ladder_process(variant)
    spec = model_spec(architecture, process)
    params = load_params(checkpoint_path, spec)
    rows, length, episodes = (256, 256, 2) if smoke else (12000, 1024, 32)
    fit_episodes = int(np.ceil(rows/(length-128)))
    fit = gru.rollout(spec, process, params, gru.empty_maps(spec), jax.random.PRNGKey(42), fit_episodes, length, capture=True)
    maps = gru.fit_maps(spec, fit, rows)
    test = gru.rollout(spec, process, params, maps, jax.random.PRNGKey(43), fit_episodes, length, capture=True)
    probes = {'exclusion_8': gru.probe_scores(spec, test, maps, rows, 8),
              'exclusion_128': gru.probe_scores(spec, test, gru.fit_maps(spec, fit, rows, warmup=128), rows, 128)}
    np.savez(raw/'maps.npz', **{k: np.asarray(v) for k, v in maps.items()})
    confirmation = []

    def confirm(condition, site=gru.NONE, mode=0, strength=0., orthogonal=False, random=False):
        records = gru.rollout(spec, process, params, maps, jax.random.PRNGKey(2042), episodes, length,
                              site, mode, strength, orthogonal, random)
        result = metrics(records)
        # A write carried in the hidden state can compound until the state overflows.
        diverged = not bool(np.isfinite(np.asarray(records['norms'])).all())
        confirmation.append(dict(condition=condition, site=site, mode=MODES[mode], strength=float(strength),
                                 diverged=diverged, metrics=result))

    top, final = spec.n_layers-1, spec.n_layers
    confirm('intact')
    confirm('random', random=True)
    for mode_index, mode in enumerate(MODES):
        for site in (top, final):
            for strength in LAMBDAS:
                confirm('curve_'+mode, site, mode_index, strength)
            confirm('orthogonal_'+mode, site, mode_index, 1., orthogonal=True)
        for site in list(range(top))+[gru.ALL_HIDDEN]:
            confirm('site_'+mode, site, mode_index, 1.)
    pulse_records = gru.rollout(spec, process, params, maps, jax.random.PRNGKey(3042), episodes, length, capture=True)
    pulse = gru.run_pulse(spec, params, maps, pulse_records, variant,
                          times=(32, 80, 128) if smoke else (256, 512, 768), horizon=16 if smoke else 128)
    here = Path(__file__).parent
    sources = {p.name: sha256(p) for p in (Path(__file__), here/'steering_gru.py', here/'steering_core.py',
                                           here/'steering_pulse.py', here/'environment.py')}
    report = dict(schema=1, study=study, variant=variant, architecture=architecture, seed=42, smoke=smoke, checkpoint=checkpoint,
        provenance=dict(experiment_revision=revision(root), source_sha256=sources, jax=jax.__version__,
                        python=platform.python_version(), devices=[str(x) for x in jax.devices()]),
        protocol=dict(fit_rows=rows, fit_seed=42, test_seed=43, exclusion=8, supplemental_exclusion=128,
                      affine_rcond=AFFINE_RCOND, confirm_seed=2042, confirm_episodes=episodes, episode_length=length,
                      lambdas=LAMBDAS, sites=f'0..{top} hidden state per GRU layer (carried); {final} final LayerNorm (not carried); {gru.ALL_HIDDEN} all hidden layers',
                      target='target belief built from the unsteered shadow network on the executed inputs'),
        probes=probes, confirmation=confirmation, pulse=pulse, total_seconds=time.perf_counter()-start)
    report = finite(report)
    confirmation, pulse = report['confirmation'], report['pulse']
    path.write_text(json.dumps(report, allow_nan=False))
    if not smoke:
        key = f'experiments/{study}/variant_{variant}_{architecture}/{TAG}/full-reports/{local_reference(path)["sha256"]}/{path.name}'
        reference = upload_full(path, key)
        summary = dict(report, payload_kind='compact_summary', full_report=reference,
                       confirmation=[dict(c, metrics={k: v for k, v in c['metrics'].items() if k != 'episode_occupancy'})
                                     for c in confirmation if not c['condition'].startswith('curve_') or c['strength'] == 1],
                       pulse=dict(pulse, conditions={k: compact_pulse(v) for k, v in pulse['conditions'].items()}))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(dump_compact(summary))
    print(f'Wrote {path} in {report["total_seconds"]:.1f}s', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES, required=True)
    parser.add_argument('--variant', choices=(1, 2, 3), type=int, required=True)
    parser.add_argument('--architecture', choices=('gru128', 'gru92'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.study, args.variant, args.architecture, args.smoke)


if __name__ == '__main__':
    main()
