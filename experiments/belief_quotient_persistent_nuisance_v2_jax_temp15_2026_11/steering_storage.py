"""Compact findings in results; full steering payloads in artifacts and B2."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from harness.storage.b2 import B2StorageConfig


def headline(item):
    return not item['condition'].startswith('curve_') or item['strength'] == 1


def metrics_summary(metrics):
    return {key: value for key, value in metrics.items() if key != 'episode_occupancy'}


def pulse_summary(pulse):
    result = {}
    for key, value in pulse.items():
        if key in ('measured', 'full_filter', 'quotient_filter'):
            absolute = np.asarray(value['absolute'])
            active = np.flatnonzero(absolute.max(-1) > 0) if absolute.size else []
            result[key] = dict(decay=value['decay'],
                absolute_at_lag={str(k): absolute[k].tolist() for k in (0, 1, 32, 33, 64, 65, 96, 97, 128) if k < len(absolute)},
                last_nonzero_lag=int(active[-1]) if len(active) else None)
            if value.get('delta_not_defined'):
                result[key]['delta_not_defined'] = True
        elif key in ('tv', 'action_disagreement'):
            result[key] = {str(k): value[k] for k in (0, 1, 32, 64, 96, 128) if k < len(value)}
        else:
            result[key] = value
    return result


def compact(report, reference):
    if report.get('payload_kind') == 'compact_summary':
        raise ValueError('expected full payload, not a compact summary')
    if 'confirmation' in report:
        keep = ('study', 'variant', 'seed', 'smoke', 'checkpoint', 'provenance', 'protocol', 'selected',
                'norm_matching', 'probes', 'reference_convergence', 'total_seconds')
        result = {key: report[key] for key in keep if key in report}
        result['confirmation'] = [dict(item, metrics=metrics_summary(item['metrics']))
                                  for item in report['confirmation'] if headline(item)]
        result['pulse'] = dict(report['pulse'], conditions={key: pulse_summary(value)
                              for key, value in report['pulse']['conditions'].items()})
    elif 'table' in report:
        keep = ('aggregation_source_sha256', 'inputs', 'caveat', 'probes', 'norm_matching')
        result = {key: report[key] for key in keep if key in report}
        result['table'] = [dict(item, metrics=metrics_summary(item['metrics'])) for item in report['table'] if headline(item)]
        result['pulse'] = [pulse_summary(p) for p in report['pulse']]
        result['epsilon_minus_no_epsilon'] = [item for item in report['epsilon_minus_no_epsilon']
            if not item['condition'].startswith('curve_') or item['no_epsilon_lambda'] == 1]
        if 'numerical_references' in report:
            result['numerical_references'] = prediction_summary(report['numerical_references'])
    elif 'results' in report and 'headline' in report:
        result = prediction_summary(report)
    else:
        raise ValueError('unrecognized steering payload')
    return dict(schema=3, payload_kind='compact_summary', full_report=reference, **result)


def prediction_summary(report):
    result = {key: report[key] for key in ('headline', 'grid_sensitivity', 'refinement', 'shared_across_epsilon',
              'seed', 'episodes', 'episode_length') if key in report}
    result['convergence'] = {key: value['convergence'] for key, value in report['results'].items()}
    candidates = {f"{variant}_{value['candidate']}" for variant, value in report['headline'].items()}
    result['headline_curves'] = {key: {mode: {strength: {k: v for k, v in point.items() if k != 'action_fractions'}
        for strength, point in curve.items()} for mode, curve in value['curves'].items()}
        for key, value in report['results'].items() if key in candidates}
    return result


def local_reference(path):
    path = Path(path)
    blob = path.read_bytes()
    return dict(local_path=str(path), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))


def require_artifacts(path):
    if 'artifacts' not in Path(path).parts:
        raise ValueError('full payload output must be under ignored artifacts/')


def upload_full(path, key):
    config = B2StorageConfig.from_env()
    client = config.s3_client()
    reference = local_reference(path)
    client.upload_file(str(path), config.bucket, key, ExtraArgs={'Metadata': {'sha256': reference['sha256']}})
    blob = client.get_object(Bucket=config.bucket, Key=key)['Body'].read()
    if hashlib.sha256(blob).hexdigest() != reference['sha256']:
        raise ValueError(f'B2 SHA256 mismatch: {key}')
    return dict(reference, bucket=config.bucket, key=key, uri=f's3://{config.bucket}/{key}')


def write_summary(path, report, reference):
    summary = compact(report, reference)
    lines = ['{']
    for key, value in summary.items():
        if isinstance(value, list):
            lines.append(f'  {json.dumps(key)}: [')
            lines.extend('    '+json.dumps(item, allow_nan=False)+',' for item in value)
            if value:
                lines[-1] = lines[-1].removesuffix(',')
            lines.append('  ],')
        else:
            lines.append(f'  {json.dumps(key)}: '+json.dumps(value, allow_nan=False)+',')
    lines[-1] = lines[-1].removesuffix(',')
    lines.append('}')
    if len(lines) > 3000:
        raise ValueError('compact report exceeds 3000 lines')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('full', type=Path)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    require_artifacts(args.full)
    if args.summary.exists() or args.summary.with_suffix('.md').exists():
        raise FileExistsError(args.summary)
    report = json.loads(args.full.read_text())
    reference = upload_full(args.full, args.key)
    write_summary(args.summary, report, reference)
    if args.full.with_suffix('.md').exists():
        shutil.copyfile(args.full.with_suffix('.md'), args.summary.with_suffix('.md'))


if __name__ == '__main__':
    main()
