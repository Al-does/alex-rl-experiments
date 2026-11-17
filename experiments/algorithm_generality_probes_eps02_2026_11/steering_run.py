"""CPU-only final-actor steering; no training, uploads or remote compute."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

from .analyze import FINAL, REPO, ROOT, SEEDS, discover, read, restore, write

import numpy as np


def moving_runs():
    runs = []
    for algorithm in ('ppo', 'sac_autoalpha'):
        study = f'reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_{algorithm}_2026_11'
        selected = []
        for path in (REPO/'experiments'/study/'reward_both_transformer/results').glob('*/run_manifest.json'):
            m = read(path)
            seed = m['runtime']['seed']
            if seed not in SEEDS or m['runtime']['smoke']:
                continue
            if m['status'] != 'completed' or m['remote_artifacts']['status'] != 'completed':
                raise ValueError(f'incomplete run {path}')
            selected.append(dict(algorithm=algorithm, seed=seed, study=study, condition='reward_both',
                leaf='reward_both_transformer', run_id=m['run_id'], results_path=str(path.parent.relative_to(REPO))))
        if sorted(r['seed'] for r in selected) != list(SEEDS):
            raise ValueError(f'expected exactly five Joint Reward seeds: {study}')
        runs.extend(sorted(selected, key=lambda r: r['seed']))
    return runs


def worker(task):
    kind, run, smoke, cpu = task
    os.sched_setaffinity(0, {cpu})
    import jax
    from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
    from experiments.belief_quotient_history_jax_2026_11.env import make_process
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
    from . import steer_ladder, steer_moving, steer_moving_base
    from .steer_storage import compact

    tag = f"{kind}_{run['algorithm']}_{'variant'+str(run['variant']) if kind == 'ladder' else 'joint_reward'}_seed{run['seed']}"
    raw = ROOT/'artifacts/steering'/('smoke' if smoke else 'full')/tag
    output = raw/'compact.json' if smoke else ROOT/'results/steering/agents'/f'{tag}.json'
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    recipe = read(REPO/run['results_path']/'resolved_recipe.json')
    spec = ModelSpec(**recipe['model'])
    params, checkpoint = restore(run, FINAL, spec)
    with jax.default_matmul_precision('highest'):
        if kind == 'ladder':
            process = environment.ladder_process(run['variant'])
            for k, v in process._asdict().items():
                np.testing.assert_allclose(v, recipe['process'][k], atol=1e-6)
            report = steer_ladder.run(spec, process, params, raw, run['variant'], run['seed'], smoke, environment)
        else:
            checkpoint_path = ROOT/'artifacts/checkpoints'/run['run_id']/f'step_{FINAL:010d}.npz'

            def verified_actor(path, requested_spec):
                if Path(path) != checkpoint_path or requested_spec != spec:
                    raise ValueError('unexpected checkpoint or model specification')
                return params

            steer_moving_base.load_params = verified_actor
            if smoke:
                steer_moving_base.FIT_ROWS = 256
            source = dict(run, results=REPO/run['results_path'])
            report = steer_moving.analyze(source, checkpoint_path, tune_episodes=2 if smoke else 64,
                                          confirm_episodes=2 if smoke else 256, seed=42)
    provenance = dict(kind=kind, algorithm=run['algorithm'], training_seed=run['seed'], run_id=run['run_id'],
        checkpoint_verified=checkpoint, smoke=smoke, cpu_affinity=sorted(os.sched_getaffinity(0)),
        analysis_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        harness_revision=subprocess.check_output(['git', '-C', str(REPO.parent/'rl-harness'), 'rev-parse', 'HEAD'], text=True).strip(),
        training_revisions=read(REPO/run['results_path']/'run_manifest.json')['git'],
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))},
        pinned_sources=read(ROOT/'steering_sources.json'), wall_seconds=time.monotonic()-start,
        jax=jax.__version__, numpy=np.__version__, devices=[str(d) for d in jax.devices()])
    report['adapter_provenance'] = provenance
    full = raw/'full_report.json'
    write(full, report)
    if kind == 'ladder':
        result = compact(report, dict(local_path=str(full.relative_to(REPO)), sha256=hashlib.sha256(full.read_bytes()).hexdigest()))
        result['adapter_provenance'] = provenance
    else:
        result = report
    write(output, result)
    print(f'DONE {tag} smoke={smoke} {provenance["wall_seconds"]:.2f}s', flush=True)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=('ladder', 'moving', 'both'), default='both')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= args.workers <= len(cpus):
        parser.error('workers must fit available CPU affinity')
    inventory = {k: v for k, v in [('ladder', discover()), ('moving', moving_runs())] if args.analysis in (k, 'both')}
    tasks = []
    for kind, runs in inventory.items():
        if args.smoke:
            runs = [next(r for r in runs if r['algorithm'] == 'ppo' and (kind != 'ladder' or r['variant'] == 2))]
        tasks.extend((kind, r, args.smoke) for r in runs)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        for start in range(0, len(tasks), args.workers):
            batch = [(*task, cpus[i]) for i, task in enumerate(tasks[start:start+args.workers])]
            list(pool.map(worker, batch, chunksize=1))


if __name__ == '__main__':
    main()
