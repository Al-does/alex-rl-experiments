"""One-agent offline steering battery. No training or paid infrastructure calls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import shutil

import jax
import numpy as np
import analysis

from harness.storage.b2 import B2StorageConfig
from harness.storage.b2 import upload_run_artifacts
from harness.context import RunContext
from experiments.belief_quotient_history_jax_2026_11.continuation import locate_parent_run, load_params
from experiments.belief_quotient_history_jax_2026_11.shared import model_spec
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.environment import ladder_process
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import LAMBDAS, MODES, empty_maps, fit_maps, metrics, probe_scores, rollout
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import Policy, target
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_pulse import run_pulse

STUDIES = tuple(f'belief_quotient_persistent_nuisance_v2_jax_temp15{s}_2026_11' for s in ('', '_eps05'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(leaf, study, destination):
    parent = locate_parent_run(leaf, study=study, seed=42)
    storage = B2StorageConfig.from_env()
    if storage is None:
        raise RuntimeError('B2 credentials required for manifest-verified restoration')
    client = storage.s3_client()
    manifest_key = parent['b2_prefix']+'/metadata/durability_manifest.json'
    raw_manifest = client.get_object(Bucket=parent['b2_bucket'], Key=manifest_key)['Body'].read()
    manifest = json.loads(raw_manifest)
    entries = [e for e in manifest['files'] if e['relative_path'] == parent['checkpoint_file']]
    if len(entries) != 1:
        raise ValueError('final checkpoint must occur exactly once in canonical manifest')
    path = destination/Path(parent['checkpoint_file']).name
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'durability_manifest.json').write_bytes(raw_manifest)
    if not path.exists():
        client.download_file(parent['b2_bucket'], parent['b2_prefix']+'/'+parent['checkpoint_file'], str(path))
    digest = sha256(path)
    if digest != entries[0]['sha256']:
        raise ValueError(f'checkpoint SHA256 mismatch: {path}')
    return path, dict(**parent, sha256=digest, manifest_key=manifest_key,
                      manifest_sha256=hashlib.sha256(raw_manifest).hexdigest())


def revision(directory):
    return subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()


def persist(leaf, raw, report_path, report):
    """Satisfy Vast's required-durability teardown contract without training."""
    metadata = leaf/'results'/'belief_steering_v2_seed42_run'
    metadata.mkdir(parents=True, exist_ok=True)
    for source in (report_path, report_path.with_suffix('.md')):
        shutil.copyfile(source, raw/source.name)
    manifest_path = metadata/'run_manifest.json'
    manifest = dict(status='completed', run_id='belief_steering_v2_seed42',
                    git=report['provenance'], checkpoint=report['checkpoint'])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    context = RunContext(experiment_dir=leaf, results_dir=metadata, artifacts_dir=raw,
                         seed=42, run_id='belief_steering_v2_seed42', upload_artifacts=True)
    manifest['remote_artifacts'] = upload_run_artifacts(context,
        experiment_module=f"experiments.{report['study']}.variant_{report['variant']}_transformer.belief_steering")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print('B2 analysis durability completed:', manifest['remote_artifacts']['base_uri'], flush=True)


def markdown(report):
    lines = [f"# V2 steering: {report['study']} variant {report['variant']}", '',
             f"Checkpoint SHA256: `{report['checkpoint']['sha256']}`", '',
             '| Condition | Site | Effective λ | Occupancy ± episode SE | Action change | Shift norm | Relative shift |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for item in report['confirmation']:
        m = item['metrics']
        lines.append(f"| {item['condition']} | {item['site']} | {item['strength']:.5g} | {m['occupancy']:.6f} ± {m['occupancy_se']:.6f} | {m['action_change']:.5f} | {m['shift_norm']:.5g} | {m['relative_shift']:.5g} |")
    lines += ['', 'Full curves, tuning, 8/128-exclusion probes, calibration diagnostics, pulse curves/rates and numerical reference convergence are in the companion JSON.',
              'Reference policies are discounted nearest-grid numerical approximations, not certified optima. Raw affine decoded beliefs are not clipped. Norm-matched δ uses tuning-only scalar calibration; achieved confirmation norms are reported, not assumed.',
              'Pulse rates fit lags 1..K above numerical floor; inspect lag 0/1 amplitudes and fit R² before interpreting a rate.']
    return '\n'.join(lines)+'\n'


def run(study, variant, smoke=False):
    root = Path(__file__).resolve().parents[2]
    leaf = root/'experiments'/study/f'variant_{variant}_transformer'
    tag = 'smoke' if smoke else 'seed42'
    raw = leaf/'artifacts'/f'belief_steering_v2_{tag}'
    output = raw if smoke else leaf/'results'
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'belief_steering_v2_{tag}.json'
    if path.exists() or path.with_suffix('.md').exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    start = time.perf_counter()
    checkpoint_path, checkpoint = restore(leaf, study, raw)
    process = ladder_process(variant)
    spec = model_spec('transformer', process)
    params = load_params(checkpoint_path, spec)
    rows, length = (256, 256) if smoke else (12000, 1024)
    tune_episodes, confirm_episodes = (2, 2) if smoke else (16, 32)
    episodes = int(np.ceil(rows/(length-128)))
    maps = empty_maps()
    fit = rollout(spec, process, params, maps, jax.random.PRNGKey(42), episodes, length, capture=True)
    maps = fit_maps(fit, rows)
    supplemental_maps = fit_maps(fit, rows, warmup=128)
    test = rollout(spec, process, params, maps, jax.random.PRNGKey(43), episodes, length, capture=True)
    probes = {'exclusion_8': probe_scores(test, maps, rows, 8),
              'exclusion_128': probe_scores(test, supplemental_maps, rows, 128)}
    np.savez(raw/'maps.npz', **{k: np.asarray(v) for k, v in maps.items()})
    policy = Policy(variant, n=100, gamma=.995)
    timings = []

    def evaluate(seed, episodes, site=-1, mode=0, strength=0., orthogonal=False, random=False):
        tick = time.perf_counter()
        records = rollout(spec, process, params, maps, jax.random.PRNGKey(seed), episodes, length,
                          site, mode, strength, orthogonal, random)
        result = metrics(records)
        b = np.asarray(records['belief'])
        installed = b if orthogonal or site < 0 else (1-strength)*b+strength*target(b, MODES[mode])
        result['reference_action_agreement'] = float((policy(installed) == np.asarray(records['action'])).mean())
        result['reference_agreement_caveat'] = 'exact installed-belief reference; extrapolated beliefs projected nonnegative for λ>1'
        elapsed = time.perf_counter()-tick
        timings.append(dict(seconds=elapsed, decisions=episodes*length))
        return result

    random_tune = evaluate(1042, tune_episodes, random=True)
    tuning, selected = {}, {}
    for mode_index, mode in enumerate(MODES):
        curve = {str(l): evaluate(1042, tune_episodes, 3, mode_index, l) for l in LAMBDAS}
        eligible = [l for l in LAMBDAS if curve[str(l)]['occupancy'] <= random_tune['occupancy']]
        selected[mode] = min(eligible) if eligible else min(LAMBDAS, key=lambda l: curve[str(l)]['occupancy'])
        tuning[mode] = curve
        print(f'{mode}: selected lambda {selected[mode]}', flush=True)
    confirmation = []

    def confirm(condition, site=-1, mode=0, strength=0., orthogonal=False, random=False):
        result = evaluate(2042, confirm_episodes, site, mode, strength, orthogonal, random)
        confirmation.append(dict(condition=condition, site=site, mode=MODES[mode], strength=float(strength), metrics=result))
        return result

    confirm('intact')
    confirm('random', random=True)
    for mode_index, mode in enumerate(MODES):
        for site in (3, 4):
            for l in LAMBDAS:
                confirm('curve_'+mode, site, mode_index, l)
        for site in range(5):
            confirm('selected_'+mode, site, mode_index, selected[mode])
        confirm('orthogonal_'+mode, 3, mode_index, selected[mode], orthogonal=True)
        confirm('orthogonal_'+mode, 4, mode_index, selected[mode], orthogonal=True)
    calibration = {}
    for site in (3, 4):
        reference_norm = evaluate(1042, tune_episodes, site, 2, 1.)['shift_norm']
        for mode_index, mode in enumerate(MODES[:2]):
            low, high = 0., 1.
            achieved = evaluate(1042, tune_episodes, site, mode_index, high)['shift_norm']
            while achieved < reference_norm and high < 1024:
                high *= 2
                achieved = evaluate(1042, tune_episodes, site, mode_index, high)['shift_norm']
            for _ in range(12 if not smoke else 3):
                middle = (low+high)/2
                norm = evaluate(1042, tune_episodes, site, mode_index, middle)['shift_norm']
                if norm < reference_norm:
                    low = middle
                else:
                    high = middle
            effective = (low+high)/2
            measured = confirm('normmatched_'+mode, site, mode_index, effective)
            calibration[f'{mode}_site{site}'] = dict(target_tune_norm=reference_norm, effective_lambda=effective,
                tune_achieved_norm=evaluate(1042, tune_episodes, site, mode_index, effective)['shift_norm'],
                confirmation_achieved_norm=measured['shift_norm'], bracketed=achieved >= reference_norm,
                caveat='Scalar calibration assumes approximate norm monotonicity; inspect achieved norms. Confirmation is never used to calibrate.')
    pulse_records = rollout(spec, process, params, maps, jax.random.PRNGKey(3042), confirm_episodes, length, capture=True)
    pulse = run_pulse(spec, params, maps, pulse_records, variant,
                      times=(32, 80, 128) if smoke else (256, 512, 768), horizon=16 if smoke else 128)
    harness_root = Path(analysis.__file__).resolve().parents[1]
    sources = {p.name: sha256(p) for p in Path(__file__).parent.glob('steering_*.py')}
    sources[Path(__file__).name] = sha256(__file__)
    sources['trainer_models.py'] = sha256(root/'experiments/belief_quotient_history_jax_2026_11/models.py')
    sources['trainer_env.py'] = sha256(root/'experiments/belief_quotient_history_jax_2026_11/env.py')
    sources['environment.py'] = sha256(Path(__file__).with_name('environment.py'))
    report = dict(schema=1, study=study, variant=variant, seed=42, smoke=smoke, checkpoint=checkpoint,
        provenance=dict(experiment_revision=revision(root), harness_revision=revision(harness_root),
                        source_sha256=sources, jax=jax.__version__, python=platform.python_version(), devices=[str(x) for x in jax.devices()]),
        protocol=dict(fit_rows=rows, test_rows=rows, fit_seed=42, test_seed=43, exclusion=8, supplemental_exclusion=128,
                      tune_seed=1042, tune_episodes=tune_episodes, confirm_seed=2042, confirm_episodes=confirm_episodes,
                      episode_length=length, lambdas=LAMBDAS, shadow='independent unsteered carry on executed trajectory inputs'),
        probes=probes, tuning=tuning, tuning_random=random_tune, selected=selected, confirmation=confirmation,
        norm_matching=calibration, pulse=pulse, reference_convergence=policy.convergence,
        timings=timings, total_seconds=time.perf_counter()-start)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    path.with_suffix('.md').write_text(markdown(report))
    if not smoke:
        persist(leaf, raw, path, report)
    steady = timings[1:]
    throughput = sum(x['decisions'] for x in steady)/sum(x['seconds'] for x in steady)
    print(f'Wrote {path}\nInstrumented throughput: {throughput:.1f} decisions/s\nTotal seconds: {report["total_seconds"]:.3f}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES, required=True)
    parser.add_argument('--variant', choices=(1, 2, 3), type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.study, args.variant, args.smoke)


if __name__ == '__main__':
    main()
