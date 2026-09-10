"""Targeted, read-only CPU verification of four archived Strata RLModules.

Use the project interpreter with the exact 32a15242 harness installed or on
PYTHONPATH. Paths resolve from this file and the imported harness package, not
the original analysis worktree. No Ray cluster, training, artifact listing, or
credential output.
Only canonical manifest GETs and four complete default_policy subtrees are read
from B2. Downloads remain under ignored leaf-local artifacts; compact output is
exclusive-create so an existing scientific result can never be overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

import harness

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(harness.__file__).resolve().parents[1]
EXPECTED_HARNESS_COMMIT = '32a15242b01450472c522f0bac250e1a2f7e2a6d'
STUDIES = {
    'token': ('strata_token_guess_cycle_2/ppo', '20260909T083950Z-45ea794b'),
    'controlled': ('strata_two_factor_explore_cycle_3/reward_factor_1_state_0', '20260909T084527Z-64ab39a1'),
}
OUTPUT = ROOT / 'experiments/strata_token_guess_cycle_2/ppo/results/branch_analysis_20260909/fresh_checkpoint_checks.json'
SEEDS = (0, 1, 2, 3)
N_STEPS = 10_000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def load(path):
    return json.loads(path.read_text())


def provenance(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes()), 'size_bytes': path.stat().st_size}


def remote_bytes(client, bucket, key):
    # Never propagate service exceptions: they can carry request/config details.
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response['Body']
        try:
            return body.read()
        finally:
            body.close()
    except Exception as exc:
        raise SystemExit('STOP: B2 request failed (' + type(exc).__name__ + '); no credentials/configuration exposed.') from None


def prepare_downloads():
    from harness.storage.b2 import B2StorageConfig
    try:
        config = B2StorageConfig.from_env()
        if config is None:
            raise RuntimeError('not configured')
        client = config.s3_client()
        del config
    except Exception as exc:
        raise SystemExit('STOP: B2 authentication/configuration unavailable (' + type(exc).__name__ + '); no further retrieval attempted.') from None
    studies = {}
    for study, (leaf_name, run_id) in STUDIES.items():
        leaf = ROOT / 'experiments' / leaf_name
        run = leaf / 'results' / run_id
        manifest = load(run / 'run_manifest.json')
        condition = load(run / 'condition_summary.json')
        reports = condition['checkpoint_reports']
        chosen = {
            'peak': max(reports, key=lambda r: r['policy']['mean_reward']),
            'final': max(reports, key=lambda r: r['agent_steps']),
        }
        remote = manifest['remote_artifacts']
        canonical_bytes = remote_bytes(client, remote['bucket'], remote['canonical_manifest_key'])
        canonical = json.loads(canonical_bytes)
        assert canonical['bucket'] == remote['bucket']
        assert canonical['status'] == 'completed'
        assert canonical['prefix'].strip('/') == remote['prefix'].strip('/')
        rows = canonical['files']
        assert len({r['key'] for r in rows}) == len(rows)
        condition_row = [r for r in rows if r['key'] == remote['prefix'] + '/compact-results/condition_summary.json']
        assert len(condition_row) == 1
        assert sha((run / 'condition_summary.json').read_bytes()) == condition_row[0]['sha256']
        assert (run / 'condition_summary.json').stat().st_size == condition_row[0]['size_bytes']
        entry = {
            'run_id': run_id,
            'source_dir': str(run.relative_to(ROOT)),
            'training_seed': condition['seed'],
            'canonical_manifest': {
                'key': remote['canonical_manifest_key'], 'bucket': remote['bucket'],
                'sha256': sha(canonical_bytes), 'size_bytes': len(canonical_bytes),
                'condition_summary_matches_canonical_sha256_and_size': True,
                'hash_note': 'Canonical manifest digest computed on retrieved bytes; not independently signed/pinned by run_manifest.',
            },
            'sources': [provenance(run / name) for name in ('run_manifest.json', 'condition_summary.json', 'training_curves.jsonl', 'tune_summary.json', 'resolved_recipe.json')],
            'training_environment_versions': manifest['framework_versions'],
            'training_library_commit': manifest['git']['library']['commit'],
            'training_experiment_commit': manifest['git']['experiment_repository']['commit'],
            'checkpoints': {},
        }
        for phase, report in chosen.items():
            label = report['checkpoint']
            selected = []
            prefixes = set()
            for row in rows:
                parts = PurePosixPath(row['relative_path']).parts
                if label not in parts or 'default_policy' not in parts:
                    continue
                checkpoint_index = parts.index(label)
                module_index = parts.index('default_policy')
                if checkpoint_index >= module_index:
                    continue
                if tuple(parts[checkpoint_index + 1:module_index]) != ('learner_group', 'learner', 'rl_module'):
                    continue
                assert row['kind'] == 'artifact'
                assert row['key'] == remote['prefix'] + '/' + row['relative_path']
                prefix = str(PurePosixPath(*parts[:module_index + 1]))
                prefixes.add(prefix)
                selected.append((row, PurePosixPath(*parts[module_index + 1:])))
            assert len(prefixes) == 1, (study, phase, 'ambiguous or absent module prefix')
            assert selected, (study, phase, 'empty module subtree')
            destination = leaf / 'artifacts' / 'branch_report_check' / phase / 'default_policy'
            assert git('check-ignore', str(destination / 'module_state.pkl'))
            records = []
            for row, relative in sorted(selected, key=lambda item: str(item[1])):
                assert not relative.is_absolute() and '..' not in relative.parts and relative.parts
                target = destination.joinpath(*relative.parts)
                assert destination.resolve() in target.resolve().parents
                if target.exists():
                    data = target.read_bytes()
                else:
                    data = remote_bytes(client, remote['bucket'], row['key'])
                    assert len(data) == row['size_bytes'], (study, phase, str(relative), 'size mismatch')
                    assert sha(data) == row['sha256'], (study, phase, str(relative), 'hash mismatch')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as out:
                        out.write(data)
                assert len(data) == row['size_bytes']
                assert sha(data) == row['sha256']
                records.append({**row, 'local_path': str(target.relative_to(ROOT)), 'verified_sha256': sha(data), 'verified_size_bytes': len(data)})
            expected = {record['local_path'] for record in records}
            actual = {str(path.relative_to(ROOT)) for path in destination.rglob('*') if path.is_file()}
            assert actual == expected, 'local module subtree contains unexpected files'
            entry['checkpoints'][phase] = {
                'agent_steps': report['agent_steps'], 'training_iteration': report['training_iteration'],
                'checkpoint_label': label, 'remote_module_prefix': remote['prefix'] + '/' + next(iter(prefixes)),
                'local_module_path': str(destination.relative_to(ROOT)),
                'archived_policy': report['policy'], 'archived_train_probe_policy': report['train_policy'],
                'files': records, 'total_verified_bytes': sum(r['size_bytes'] for r in records),
                'complete_default_policy_subtree': True,
            }
            print(f'{study} {phase}: iteration={report["training_iteration"]} steps={report["agent_steps"]} files={len(records)} bytes={sum(r["size_bytes"] for r in records)} sha256+size verified', flush=True)
        studies[study] = entry
    return studies


def distribution(values, n):
    import numpy as np
    return (np.bincount(np.asarray(values, dtype=np.int64), minlength=n) / len(values)).tolist()


def array_digest(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': sha(value.tobytes())}


def summarize(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {'mean': values.mean(axis=0).tolist(), 'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist()}


def training_curve_check(study):
    run = ROOT / study['source_dir']
    rows = [json.loads(line) for line in (run / 'training_curves.jsonl').read_text().splitlines()]
    metrics = ('return_mean', 'entropy', 'mean_kl', 'policy_loss', 'vf_loss', 'total_loss', 'value_explained_variance')
    jumps = {}
    for metric in metrics:
        candidates = [(abs(right[metric] - left[metric]), left, right) for left, right in zip(rows, rows[1:]) if isinstance(left.get(metric), (int, float)) and isinstance(right.get(metric), (int, float))]
        candidates.sort(key=lambda row: row[0], reverse=True)
        jumps[metric] = [{'absolute_change': delta, 'from': left, 'to': right} for delta, left, right in candidates[:3]]
    final = load(run / 'tune_summary.json')['trials'][0]['metrics']
    return {
        'n_iterations': len(rows), 'last_12_iterations': rows[-12:], 'largest_adjacent_changes_descriptive_only': jumps,
        'final_training_episode_return_mean': final['env_runners/episode_return_mean'],
        'final_training_episode_len_mean': final['env_runners/episode_len_mean'],
        'final_training_aggregate_reward_per_step': final['env_runners/episode_return_mean'] / final['env_runners/episode_len_mean'],
        'mean_kl_unique_values': sorted({row.get('mean_kl') for row in rows if row.get('mean_kl') is not None}),
        'caveat': 'Training episode aggregates are not fresh post-update checkpoint evaluations. KL loss is disabled in the recipes; logged zero mean_kl is not evidence of zero policy movement. Compact losses/entropy cannot establish optimizer root cause.',
    }


def evaluate(studies):
    import numpy as np
    import ray
    import torch
    from ray.rllib.core.rl_module.rl_module import RLModule
    from envs.hmm import HMMEnv
    from experiments.strata_token_guess_cycle_2 import analysis as token_analysis
    from experiments.strata_token_guess_cycle_2.process import environment_config as token_config
    from experiments.strata_two_factor_explore_cycle_3 import analysis as controlled_analysis
    from experiments.strata_two_factor_explore_cycle_3.process import environment_config as controlled_config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')
    assert not ray.is_initialized()
    for name, study in studies.items():
        config = token_config() if name == 'token' else controlled_config('reward_factor_1', 0)
        expected = {'alpha': 0.98, 't0': 0.30, 't1': 0.80}
        if name == 'token':
            assert config['model']['kwargs'] == expected
        else:
            assert all(factor['kwargs'] == expected for factor in config['model']['kwargs']['factors'])
            assert all(config['task']['kwargs'][key] == value for key, value in expected.items())
            assert config['task']['kwargs']['strength'] == 1.0
        study['evaluation_environment'] = config
        env = HMMEnv(config)
        try:
            study['spaces'] = {'environment_observation': str(env.observation_space), 'environment_action': str(env.action_space)}
        finally:
            env.close()
        study['training_curve_check'] = training_curve_check(study)
        paired_histories = {}
        for phase, checkpoint in study['checkpoints'].items():
            module = RLModule.from_checkpoint(str(ROOT / checkpoint['local_module_path'])).to(device).eval()
            assert not ray.is_initialized()
            assert module.sampling_temperature == 1.5
            assert {str(p.device) for p in module.parameters()} == {'cpu'}
            checkpoint['module'] = {
                'class': type(module).__module__ + ':' + type(module).__qualname__,
                'observation_space': str(module.observation_space), 'action_space': str(module.action_space),
                'sampling_temperature': module.sampling_temperature, 'parameter_count': sum(p.numel() for p in module.parameters()),
                'encoder_blocks': len(module.encoder.blocks), 'device': 'cpu',
            }
            rows = []
            for seed in SEEDS:
                start = time.monotonic()
                kwargs = dict(n_steps=N_STEPS, seed=np.random.SeedSequence(seed), device=device)
                if name == 'token':
                    data = token_analysis.collect_probe_data(module, **kwargs)
                else:
                    data = controlled_analysis.collect_probe_data(module, condition='reward_factor_1', reward_state=0, env_config=config, **kwargs)
                assert len(data.rewards) == N_STEPS
                assert data.episode_steps.min() >= 32
                with torch.inference_mode():
                    # Last block residual is precisely the existing collector's
                    # pre-final-norm actor representation. No history re-rollout.
                    residuals = torch.as_tensor(data.activations[:, -1, :], dtype=torch.float32, device=device)
                    logits = module.action_distribution_inputs(module.encoder.final_norm(residuals))
                    probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
                probs /= probs.sum(axis=1, keepdims=True)
                row = {
                    'analysis_seed': seed, 'retained_steps': len(data.rewards), 'mean_reward': float(data.rewards.mean()),
                    'action_fractions': distribution(data.actions, module.action_space.n),
                    'policy_mean_probabilities': probs.mean(axis=0).tolist(),
                    'mean_policy_entropy_nats': float(-(probs * np.log(np.maximum(probs, 1e-300))).sum(axis=1).mean()),
                    'mean_policy_max_probability': float(probs.max(axis=1).mean()),
                    'observation_digest': array_digest(data.observations), 'state_digest': array_digest(data.states),
                    'activation_shape': list(data.activations.shape), 'joint_belief_shape': list(data.joint_beliefs.shape),
                    'factor_belief_shape': list(data.factor_beliefs.shape), 'observation_mean': data.observations.mean(axis=0).tolist(),
                    'episode_step_min': int(data.episode_steps.min()), 'episode_step_max': int(data.episode_steps.max()),
                    'product_consistency_max_abs': data.product_consistency_max_abs,
                }
                if name == 'token':
                    row['hidden_token_fractions'] = distribution(data.hidden_tokens, 2)
                    row['greedy_accuracy_same_passive_histories'] = float(np.mean(probs.argmax(axis=1) == data.hidden_tokens))
                    row['policy_expected_accuracy_given_realized_hidden_tokens'] = float(probs[np.arange(N_STEPS), data.hidden_tokens].mean())
                    row['hidden_token_digest'] = array_digest(data.hidden_tokens)
                    history = (row['observation_digest'], row['state_digest'], row['hidden_token_digest'])
                    if phase == 'peak':
                        paired_histories[seed] = history
                    else:
                        assert paired_histories[seed] == history, 'passive paired environment history mismatch'
                else:
                    row['factor_action_fractions'] = [distribution(data.actions // 3, 3), distribution(data.actions % 3, 3)]
                    row['factor_policy_mean_probabilities'] = [probs.reshape(-1, 3, 3).sum(axis=2).mean(axis=0).tolist(), probs.reshape(-1, 3, 3).sum(axis=1).mean(axis=0).tolist()]
                row['elapsed_seconds'] = time.monotonic() - start
                rows.append(row)
                print(f'{name} {phase} seed={seed}: reward={row["mean_reward"]:.5f} actions={row["action_fractions"]} elapsed={row["elapsed_seconds"]:.1f}s', flush=True)
                del data, probs, residuals, logits
            checkpoint['evaluation_seeds'] = rows
            metrics = ('mean_reward', 'action_fractions', 'policy_mean_probabilities', 'mean_policy_entropy_nats', 'mean_policy_max_probability')
            metrics += ('greedy_accuracy_same_passive_histories', 'policy_expected_accuracy_given_realized_hidden_tokens') if name == 'token' else ('factor_action_fractions', 'factor_policy_mean_probabilities')
            checkpoint['over_four_evaluation_seeds'] = {metric: summarize([row[metric] for row in rows]) for metric in metrics}
            del module
        peak = study['checkpoints']['peak']['evaluation_seeds']
        final = study['checkpoints']['final']['evaluation_seeds']
        deltas = [p['mean_reward'] - f['mean_reward'] for p, f in zip(peak, final)]
        study['paired_peak_minus_final_reward'] = {'by_seed': dict(zip(map(str, SEEDS), deltas)), **summarize(deltas)}
        study['decline_persists_all_four_paired_seeds'] = all(delta > 0 for delta in deltas)
        study['passive_paired_histories_verified_identical'] = True if name == 'token' else None
    assert not ray.is_initialized()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if git('rev-parse', 'HEAD', cwd=HARNESS) != EXPECTED_HARNESS_COMMIT:
        raise SystemExit('Historical checkpoint verification requires the exact recorded harness at ' + EXPECTED_HARNESS_COMMIT + '; select that source checkout through the project environment or PYTHONPATH.')
    if git('status', '--porcelain', cwd=HARNESS):
        raise SystemExit('Historical checkpoint verification requires a clean recorded harness checkout.')
    if not args.download_only and OUTPUT.exists():
        raise SystemExit('Refusing to overwrite existing scientific output.')
    studies = prepare_downloads()
    if args.download_only:
        return
    evaluate(studies)
    sources = [ROOT / 'experiments/strata_checkpoint_followup.py', ROOT / 'experiments/strata_token_guess_cycle_2/analysis.py', ROOT / 'experiments/strata_token_guess_cycle_2/process.py', ROOT / 'experiments/strata_two_factor_explore_cycle_3/analysis.py', ROOT / 'experiments/strata_two_factor_explore_cycle_3/process.py', ROOT / 'experiments/wing_two_factor_explore_cycle_1/model.py']
    output = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment_worktree_commit': git('rev-parse', 'HEAD'), 'harness_commit': EXPECTED_HARNESS_COMMIT,
        'interpreter': sys.executable, 'python_version': sys.version,
        'framework_versions': {key: importlib.metadata.version(key) for key in ('ray', 'torch', 'numpy', 'gymnasium')},
        'analysis_sources': [provenance(path) for path in sources],
        'protocol': {
            'analysis_seeds': list(SEEDS), 'retained_steps_per_seed_per_checkpoint': N_STEPS,
            'collector': 'Existing task analysis.collect_probe_data, np.random.SeedSequence(seed), 8 environments, 32-frame warmup per episode, process-weighted retained rows.',
            'policy': 'Stochastic checkpoint policy, sampling_temperature=1.5 included exactly once by module.action_distribution_inputs.',
            'paired_environment_seeding': 'Identical SeedSequence(seed) for peak/final within each study; harness assigns distinct episode_seeds, action_spaces, policy_sampling named streams. Controlled trajectories may differ due to actions.',
            'restoration': 'RLModule.from_checkpoint, CPU, eval/inference mode, one torch thread, no Ray initialization.',
            'scope': 'Two training runs with training seed 42; four evaluation seeds are not four training-seed replicates. Means/min/max summarize evaluation-seed variability, not CIs. Retained rows are temporally dependent.',
            'greedy': 'Greedy accuracy only for passive token task on identical histories; no counterfactual greedy controlled returns are scored.',
            'optional_belief_probes': 'Not run: essential four-checkpoint verification prioritized; archived 10-fold SVD belief metrics not re-estimated.',
            'action_labels': {'token': ['guess_0', 'guess_1'], 'controlled_each_factor': ['hold', 'rotate_plus', 'rotate_minus'], 'controlled_joint_index': 'factor_1_action * 3 + factor_2_action'},
            'causality': 'Declining performance or constant actions do not identify optimizer root cause; representation geometry is not evidence of causal policy use.',
        },
        'studies': studies,
    }
    assert not git('status', '--porcelain', cwd=HARNESS)
    with OUTPUT.open('x') as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print('Saved ' + str(OUTPUT.relative_to(ROOT)), flush=True)


if __name__ == '__main__':
    main()
