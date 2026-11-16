"""Offline single-factor belief steering on the epsilon-0.02 reward-factor transformers.

For each training seed: fit marginal and joint affine maps at the four
post-block residuals and the final LayerNorm, tune the write strength at the
last block on separate episodes, then confirm every site at strength 1 and at
the tuned strength, with norm-matched decoder-orthogonal controls.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import shutil
import time

import jax
import numpy as np
import analysis

from harness.context import RunContext
from harness.storage.b2 import upload_run_artifacts
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering import restore, revision, sha256
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import AFFINE_RCOND, LAMBDAS
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import Policy
from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process, model_spec
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11.shared import CONTEXT_LEN, STUDY
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11.steering_factor import (
    KINDS, MODES, SITES, empty_maps, fit_maps, metrics, probe_scores, rollout,
)

CONDITIONS = ('reward_both', 'reward_factor_1')
TUNE_SITE = 3
ROOT = Path(__file__).resolve().parents[2]


def name(condition, seed, smoke=False):
    return f"belief_steering_factor_seed{seed}{'_smoke' if smoke else ''}"


def leaf_dir(condition):
    return ROOT/'experiments'/STUDY/f'{condition}_transformer'


def persist(leaf, raw, report, run_id):
    """Upload the full payload so a rented box may be destroyed after the run."""
    metadata = leaf/'results'/f'{run_id}_run'
    metadata.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='completed', run_id=run_id, git=report['provenance'], checkpoint=report['checkpoint'])
    context = RunContext(experiment_dir=leaf, results_dir=metadata, artifacts_dir=raw,
                         seed=report['seed'], run_id=run_id, upload_artifacts=True)
    (metadata/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    manifest['remote_artifacts'] = upload_run_artifacts(
        context, experiment_module=f"experiments.{STUDY}.{report['condition']}_transformer.belief_steering")
    (metadata/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest['remote_artifacts']


def run(condition, seed=42, smoke=False):
    leaf = leaf_dir(condition)
    run_id = name(condition, seed, smoke)
    raw = leaf/'artifacts'/run_id
    path = raw/f'{run_id}.json'
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    raw.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    checkpoint_path, checkpoint = restore(leaf, STUDY, raw, seed)
    process = factor_process(condition, environment)
    spec = replace(model_spec('transformer', process), context_len=CONTEXT_LEN)
    params = load_params(checkpoint_path, spec)
    rows, length = (256, 256) if smoke else (12000, 1024)
    tune_episodes, confirm_episodes = (2, 2) if smoke else (16, 32)
    episodes = int(np.ceil(rows/(length-8)))
    blank = empty_maps('marginal', spec.width)
    fit = rollout(spec, process, params, blank, jax.random.PRNGKey(42), episodes, length, capture=True)
    test = rollout(spec, process, params, blank, jax.random.PRNGKey(43), episodes, length, capture=True)
    maps = {kind: fit_maps(fit, rows, kind) for kind in KINDS}
    probes = {kind: probe_scores(test, maps[kind], rows, kind) for kind in KINDS}
    del fit, test
    policy = Policy(3, n=100, gamma=.995, environment=environment)

    def evaluate(seed, episodes, kind='marginal', site=-1, mode=0, strength=0., orthogonal=False, random=False):
        records = rollout(spec, process, params, maps[kind], jax.random.PRNGKey(seed), episodes, length,
                          site, mode, strength, orthogonal, random, False, kind)
        return metrics(records, policy, mode, strength, steered=site >= 0 and not orthogonal)

    random_tune = evaluate(1042, tune_episodes, random=True)
    writes = [(kind, factor, index) for kind in KINDS for factor in (0, 1) for index in range(len(MODES))]
    tuning, selected, confirmation = {}, {}, []

    def confirm(label, kind='marginal', factor=None, index=0, site=-1, strength=0., **flags):
        mode = 0 if factor is None else factor*len(MODES)+index
        result = evaluate(2042, confirm_episodes, kind, site, mode, strength, **flags)
        confirmation.append(dict(condition=label, map=kind if factor is not None else None,
                                 factor=None if factor is None else factor+1,
                                 mode=None if factor is None else MODES[index], site=site,
                                 strength=float(strength), metrics=result))

    confirm('intact')
    confirm('random', random=True)
    for kind, factor, index in writes:
        key = f'{kind}_factor{factor+1}_{MODES[index]}'
        mode = factor*len(MODES)+index
        curve = {str(l): evaluate(1042, tune_episodes, kind, TUNE_SITE, mode, l) for l in LAMBDAS}
        floor = random_tune['occupancy'][factor]
        eligible = [l for l in LAMBDAS if l > 0 and curve[str(l)]['occupancy'][factor] <= floor]
        selected[key] = min(eligible) if eligible else None
        tuning[key] = curve
        print(f'{key}: selected lambda {selected[key]}', flush=True)
        for site in range(SITES):
            confirm('write', kind, factor, index, site, 1.)
            if selected[key] not in (None, 1.):
                confirm('write_selected', kind, factor, index, site, selected[key])
        for site in (TUNE_SITE, SITES-1):
            confirm('orthogonal', kind, factor, index, site, 1., orthogonal=True)
    sources = {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('steering_factor.py'),
                                           Path(environment.__file__))}
    report = dict(schema=1, study=STUDY, condition=condition, seed=seed, smoke=smoke, checkpoint=checkpoint,
        provenance=dict(experiment_revision=revision(ROOT),
                        harness_revision=revision(Path(analysis.__file__).resolve().parents[1]),
                        source_sha256=sources, jax=jax.__version__, python=platform.python_version(),
                        devices=[str(x) for x in jax.devices()]),
        protocol=dict(fit_rows=rows, test_rows=rows, fit_seed=42, test_seed=43, exclusion=8,
                      affine_rcond=AFFINE_RCOND, tune_seed=1042, tune_episodes=tune_episodes, tune_site=TUNE_SITE,
                      confirm_seed=2042, confirm_episodes=confirm_episodes, episode_length=length,
                      lambdas=LAMBDAS, modes=MODES, maps=KINDS,
                      selection='smallest positive strength putting the steered factor at or below its random-action occupancy; null when none does',
                      shadow='independent unsteered carry on executed trajectory inputs'),
        probes=probes, tuning_random=random_tune, selected=selected, confirmation=confirmation,
        reference_convergence=policy.convergence, total_seconds=time.perf_counter()-start)
    path.write_text(json.dumps(dict(report, tuning=tuning), indent=2, allow_nan=False))
    if not smoke:
        remote = persist(leaf, raw, report, run_id)
        report['full_report'] = dict(uri=f"s3://{remote['bucket']}/{remote['prefix']}/{path.name}", sha256=sha256(path))
        summary = leaf/'results'/path.name
        summary.write_text(json.dumps(report, allow_nan=False)+'\n')
    print(f'Wrote {path}\nTotal seconds: {report["total_seconds"]:.1f}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for seed in args.seeds:
        if not args.smoke and (leaf_dir(args.condition)/'results'/f'{name(args.condition, seed)}.json').exists():
            print(f'seed {seed}: compact report exists, skipping', flush=True)
            continue
        print(f'=== {STUDY} {args.condition} seed {seed} ===', flush=True)
        run(args.condition, seed, args.smoke)


if __name__ == '__main__':
    main()
