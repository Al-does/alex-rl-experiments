"""Fixed-ridge token-cycle-2 controls using verified local peak/final modules.

Offline analysis only: no B2 access, Ray startup, optimization of a policy, or
existing-file overwrite. Fit/test seeds, ridge, representation sites, and the
single descriptive permutation seed are fixed before any scores are observed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import harness
import numpy as np
import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

from analysis.probes import fit_affine_probe, global_mse_metrics, r2_score
from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_2 import analysis as token_analysis
from experiments.strata_token_guess_cycle_2 import process

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(harness.__file__).resolve().parents[1]
HARNESS_COMMIT = '32a15242b01450472c522f0bac250e1a2f7e2a6d'
RESULT_DIR = ROOT / 'experiments/strata_token_guess_cycle_2/ppo/results/branch_analysis_20260909'
PRIOR = RESULT_DIR / 'fresh_checkpoint_checks.json'
OUTPUT = RESULT_DIR / 'token_probe_checks.json'
PARAMETERS = {'alpha': 0.98, 't0': 0.30, 't1': 0.80}
RIDGE = 1e-6
N_STEPS = 20_000
SPLITS = {'train': 4000, 'test': 4001}
PERMUTATION_SEED = 4002
HISTORY_FIELDS = ('observations', 'joint_beliefs', 'factor_beliefs', 'hidden_tokens', 'states', 'episode_steps')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    base = ROOT if path.is_relative_to(ROOT) else HARNESS
    return {'root': 'experiment' if base == ROOT else 'harness', 'path': str(path.relative_to(base)), 'sha256': digest(data), 'size_bytes': len(data)}


def array_source(array):
    array = np.ascontiguousarray(array)
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'sha256': digest(array.tobytes())}


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def score(prediction, target, null_direction):
    assert prediction.shape == target.shape == (N_STEPS, 3)
    assert np.isfinite(prediction).all()
    global_metrics = {**global_mse_metrics(prediction, target), 'r_squared': r2_score(prediction, target)}
    predicted_null = (prediction @ null_direction)[:, None]
    target_null = (target @ null_direction)[:, None]
    null_metrics = {**global_mse_metrics(predicted_null, target_null), 'r_squared': r2_score(predicted_null, target_null)}
    assert np.isclose(global_metrics['r_squared'], 1.0 - global_metrics['global_mse_ratio'])
    assert np.isclose(null_metrics['r_squared'], 1.0 - null_metrics['global_mse_ratio'])
    return {
        'global': global_metrics, 'unit_emission_null_contrast': null_metrics,
        'raw_prediction_min': float(prediction.min()), 'raw_prediction_max': float(prediction.max()),
        'negative_prediction_coordinate_fraction': float(np.mean(prediction < 0)),
        'above_one_prediction_coordinate_fraction': float(np.mean(prediction > 1)),
        'row_sum_max_abs_error': float(np.max(np.abs(prediction.sum(axis=1) - 1))),
        'prediction_digest': array_source(prediction),
    }


def fit_score(train_features, train_target, test_features, test_target, null_direction):
    weight, bias = fit_affine_probe(train_features, train_target, ridge=RIDGE)
    predicted = np.asarray(test_features, dtype=np.float64) @ weight + bias
    return {
        'feature_width': int(train_features.shape[1]), 'fit': 'analysis.probes.fit_affine_probe', 'ridge': RIDGE,
        'train_feature_digest': array_source(train_features), 'test_feature_digest': array_source(test_features),
        'weight': weight.tolist(), 'bias': bias.tolist(),
        **score(predicted, test_target, null_direction),
    }


def pending_ntp(arrival_belief, transition, emission):
    # The delayed task's belief_current is an arrival belief: source @ T.
    # Pending token was emitted on that unresolved edge, from the source.
    # arrival @ emission would instead predict a subsequent edge's token.
    pending = arrival_belief @ np.linalg.solve(transition, emission)
    source_belief = np.linalg.solve(transition.T, arrival_belief.T).T
    assert np.allclose(pending, source_belief @ emission, atol=1e-12, rtol=0)
    assert source_belief.min() >= -1e-12
    assert np.allclose(source_belief.sum(axis=1), 1, atol=1e-12, rtol=0)
    assert np.allclose(pending.sum(axis=1), 1, atol=1e-12, rtol=0)
    # No clipping or probability repair, including before log-NTP.
    assert np.isfinite(pending).all() and (pending > 0).all() and (pending <= 1).all()
    return pending


def policy_report(module, data):
    with torch.inference_mode():
        residual = torch.as_tensor(data.activations[:, -1], dtype=torch.float32)
        logits = module.action_distribution_inputs(module.encoder.final_norm(residual))
        probs = torch.softmax(logits, dim=-1).numpy().astype(np.float64)
    probs /= probs.sum(axis=1, keepdims=True)
    return {
        'stochastic_accuracy': float(data.rewards.mean()),
        'greedy_accuracy_same_passive_histories': float(np.mean(probs.argmax(axis=1) == data.hidden_tokens)),
        'action_fractions': (np.bincount(data.actions, minlength=2) / len(data.actions)).tolist(),
        'mean_policy_probabilities': probs.mean(axis=0).tolist(),
        'hidden_token_fractions': (np.bincount(data.hidden_tokens, minlength=2) / len(data.hidden_tokens)).tolist(),
    }


def main():
    if git('rev-parse', 'HEAD', cwd=HARNESS) != HARNESS_COMMIT:
        raise SystemExit('Historical token-probe verification requires the exact recorded harness at ' + HARNESS_COMMIT + '; select that source checkout through the project environment or PYTHONPATH.')
    if git('status', '--porcelain', cwd=HARNESS):
        raise SystemExit('Historical token-probe verification requires a clean recorded harness checkout.')
    import analysis.probes.linear as probe_source
    import analysis.rollouts as rollout_source
    import envs.strata.model as model_source
    assert all(Path(module.__file__).resolve().is_relative_to(HARNESS) for module in (probe_source, rollout_source, model_source))
    if OUTPUT.exists():
        raise SystemExit('Refusing to overwrite an existing token probe result.')
    prior_bytes = PRIOR.read_bytes()
    prior = json.loads(prior_bytes)
    archived_run = ROOT / prior['studies']['token']['source_dir']
    protected_paths = [path for path in RESULT_DIR.iterdir() if path.is_file()]
    protected_paths.extend(archived_run / name for name in ('run_manifest.json', 'condition_summary.json', 'resolved_recipe.json', 'tune_summary.json', 'training_curves.jsonl'))
    protected_hashes = {path: digest(path.read_bytes()) for path in protected_paths}
    assert process.environment_config()['model']['kwargs'] == PARAMETERS
    assert process.CONTEXT_LENGTH == token_analysis.WARMUP == 32
    assert token_analysis.N_ENVS == 8
    model = strata_model(**PARAMETERS)
    transition = np.asarray(model.transition_matrix, dtype=np.float64)
    emission = np.asarray(model.emission_matrix, dtype=np.float64)
    null_direction = np.cross(np.ones(3), emission[:, 0])
    null_direction /= np.linalg.norm(null_direction)
    assert np.isclose(np.linalg.norm(null_direction), 1)
    assert np.allclose(null_direction @ emission, 0, atol=1e-12, rtol=0)
    assert np.isclose(null_direction.sum(), 0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')
    assert not ray.is_initialized()
    permutation = np.random.default_rng(np.random.SeedSequence(PERMUTATION_SEED)).permutation(N_STEPS)
    checkpoints = {}
    shared_histories = {}
    for phase in ('peak', 'final'):
        checkpoint = prior['studies']['token']['checkpoints'][phase]
        local_path = ROOT / checkpoint['local_module_path']
        assert 'strata_token_guess_cycle_2/ppo/artifacts/branch_report_check/' in str(local_path)
        expected_files = {ROOT / record['local_path'] for record in checkpoint['files']}
        assert expected_files == {path for path in local_path.rglob('*') if path.is_file()}
        for record in checkpoint['files']:
            data = (ROOT / record['local_path']).read_bytes()
            assert digest(data) == record['sha256']
            assert len(data) == record['size_bytes']
        module = RLModule.from_checkpoint(str(local_path)).to(device).eval()
        assert module.sampling_temperature == 1.5
        assert len(module.encoder.blocks) == 3
        assert {str(parameter.device) for parameter in module.parameters()} == {'cpu'}
        assert not ray.is_initialized()
        datasets = {}
        sample_reports = {}
        for split, seed in SPLITS.items():
            start = time.monotonic()
            data = token_analysis.collect_probe_data(module, n_steps=N_STEPS, seed=np.random.SeedSequence(seed), device=device)
            assert data.activations.shape == (N_STEPS, 3, 64)
            assert data.joint_beliefs.shape == (N_STEPS, 3)
            assert data.episode_steps.min() >= 32
            assert np.array_equal(data.joint_beliefs, data.factor_beliefs[:, 0])
            if phase == 'peak':
                shared_histories[split] = {field: getattr(data, field).copy() for field in HISTORY_FIELDS}
            else:
                for field in HISTORY_FIELDS:
                    assert np.array_equal(getattr(data, field), shared_histories[split][field]), (split, field)
            ntp = pending_ntp(data.joint_beliefs, transition, emission)
            datasets[split] = (data, ntp)
            sample_reports[split] = {
                'analysis_seed': seed, 'retained_steps': N_STEPS,
                'history_digests': {field: array_source(getattr(data, field)) for field in HISTORY_FIELDS},
                'activation_digest': array_source(data.activations), 'pending_ntp_digest': array_source(ntp),
                'pending_ntp_min': float(ntp.min()), 'pending_ntp_max': float(ntp.max()),
                'pending_ntp_row_sum_max_abs_error': float(np.max(np.abs(ntp.sum(axis=1) - 1))),
                'episode_step_min': int(data.episode_steps.min()), 'episode_step_max': int(data.episode_steps.max()),
                'policy': policy_report(module, data), 'collection_elapsed_seconds': time.monotonic() - start,
            }
            print(f'{phase} {split}: stochastic_accuracy={data.rewards.mean():.5f} ({time.monotonic() - start:.1f}s)', flush=True)
        train, train_ntp = datasets['train']
        test, test_ntp = datasets['test']
        assert not np.array_equal(train.observations, test.observations)
        assert not np.array_equal(train.joint_beliefs, test.joint_beliefs)
        target_train, target_test = train.joint_beliefs, test.joint_beliefs
        probes = {}
        feature_pairs = {
            'ground_truth_pending_ntp': (train_ntp, test_ntp),
            'ground_truth_log_pending_ntp': (np.log(train_ntp), np.log(test_ntp)),
            'current_observed_token': (train.observations, test.observations),
            **{f'pre_final_ln_layer_{index + 1}': (train.activations[:, index], test.activations[:, index]) for index in range(3)},
        }
        for name, (train_features, test_features) in feature_pairs.items():
            probes[name] = fit_score(train_features, target_train, test_features, target_test, null_direction)
        probes['train_mean'] = {
            'feature_width': 0, 'fit': 'train target centroid; no fitted feature weights',
            'train_mean': target_train.mean(axis=0).tolist(),
            **score(np.broadcast_to(target_train.mean(axis=0), target_test.shape), target_test, null_direction),
        }
        probes['last_layer_permuted_training_labels_descriptive'] = {
            'permutation_seed': PERMUTATION_SEED, 'permutations': 1,
            'interpretation': 'Capacity-matched 64-feature descriptive control, not a permutation test, p-value, or CI; test labels remain true.',
            **fit_score(train.activations[:, -1], target_train[permutation], test.activations[:, -1], target_test, null_direction),
        }
        checkpoints[phase] = {
            'agent_steps': checkpoint['agent_steps'], 'training_iteration': checkpoint['training_iteration'],
            'checkpoint_label': checkpoint['checkpoint_label'], 'local_module_path': checkpoint['local_module_path'],
            'remote_module_prefix_provenance_only': checkpoint['remote_module_prefix'], 'files_reverified_locally': checkpoint['files'],
            'module': checkpoint['module'], 'samples': sample_reports, 'probes': probes,
        }
        if phase == 'final':
            for name in ('ground_truth_pending_ntp', 'ground_truth_log_pending_ntp', 'current_observed_token', 'train_mean'):
                assert probes[name] == checkpoints['peak']['probes'][name], name
        print(phase + ': ' + '; '.join(f'{name} globalR2={probes[name]["global"]["r_squared"]:.6f} nullR2={probes[name]["unit_emission_null_contrast"]["r_squared"]:.6f}' for name in ('pre_final_ln_layer_3', 'ground_truth_pending_ntp', 'ground_truth_log_pending_ntp')), flush=True)
        del datasets, module, train, test, data, feature_pairs
    assert not ray.is_initialized()
    assert not git('status', '--porcelain', cwd=HARNESS)
    assert all(digest(path.read_bytes()) == expected for path, expected in protected_hashes.items()), 'Prior output changed during follow-up'
    output = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment_worktree_root': str(ROOT), 'experiment_worktree_commit': git('rev-parse', 'HEAD'),
        'harness_root': str(HARNESS), 'harness_commit': HARNESS_COMMIT,
        'python_version': sys.version, 'interpreter': sys.executable,
        'framework_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'ray', 'gymnasium')},
        'sources': [source(path) for path in (Path(__file__), PRIOR, ROOT / 'experiments/strata_token_guess_cycle_2/analysis.py', ROOT / 'experiments/strata_token_guess_cycle_2/process.py', ROOT / 'experiments/strata_token_guess_cycle_2/task.py', ROOT / 'experiments/wing_two_factor_explore_cycle_1/model.py', ROOT / 'experiments/factored_representations_reproduction_PPO_2026_08/model.py', Path(probe_source.__file__), Path(rollout_source.__file__), Path(model_source.__file__), archived_run / 'condition_summary.json', archived_run / 'run_manifest.json')],
        'canonical_manifest_provenance_from_existing_fresh_checks': prior['studies']['token']['canonical_manifest'],
        'prior_outputs_unchanged_verified': True,
        'paired_peak_final_histories_exactly_equal': list(HISTORY_FIELDS),
        'shared_control_fit_and_score_exactly_equal_between_checkpoints': ['ground_truth_pending_ntp', 'ground_truth_log_pending_ntp', 'current_observed_token', 'train_mean'],
        'protocol': {
            'name': 'Independent-rollout fixed-ridge affine controls, not archived 10-fold SVD cutoff cross-validation',
            'ridge': RIDGE, 'fit_function': 'analysis.probes.fit_affine_probe',
            'fit_semantics': 'Float64 centered augmented least squares, sum-of-squares penalty 1e-6 on weights only; intercept unpenalized. Raw features: no standardization, row CV, tuning, or test selection.',
            'train_seed': SPLITS['train'], 'test_seed': SPLITS['test'], 'retained_steps_per_split_per_checkpoint': N_STEPS,
            'permutation_seed': PERMUTATION_SEED, 'permutation_count': 1,
            'collector': 'Existing experiments.strata_token_guess_cycle_2.analysis.collect_probe_data with np.random.SeedSequence(seed)',
            'sampling_distribution': 'process_weighted_rollout; independent train/test environment streams; identical corresponding peak/final histories',
            'n_envs': 8, 'actor_context_length': 32, 'warmup_per_episode': 32,
            'episode_length': 1024, 'randomize_first_episode_length': True,
            'belief_history': 'Exact diagnostic delayed arrival belief from full observed history since reset; may exceed actor strict 32-frame context.',
            'representation': 'Each of three encoder blocks current-position residual before final LayerNorm, separately fit; no concatenation or site selection.',
            'policy': 'Frozen stochastic checkpoint policy at sampling_temperature 1.5, CPU eval/inference mode, one torch thread, no Ray cluster.',
            'metric_semantics': 'Global MSE averages rows and three coordinates; global R2=1-MSE/test target variance. Null contrast uses the same belief prediction dotted with the fixed unit direction, not a separately tuned null probe.',
            'current_token_control': 'Current actor-visible delayed token one-hot, not hidden pending token; affine fit on training data only.',
            'pending_ntp': 'arrival_belief @ solve(T, M), equivalently recovered source_belief @ M; never arrival_belief @ M.',
            'log_ntp': 'Elementwise natural log of strictly positive ground-truth pending-token probabilities; no floor/clipping required or applied.',
            'unconstrained_predictions': 'All scores use raw affine predictions, including negative/out-of-simplex coordinates and negative R2. No simplex projection, clipping, or renormalization.',
            'uncertainty': 'Single independent train/test seed pair for one training seed 42. Correlated retained rows do not provide independent replicates; no CIs, p-values, or training-seed claims.',
            'scope': 'Accessible information versus simple predictive controls, not evidence of causal policy use. Historical peak checkpoint selection is retrospective. No downloads or policy training.',
        },
        'environment': process.environment_config(),
        'analytic_model': {'parameters': PARAMETERS, 'transition_matrix': transition.tolist(), 'emission_matrix': emission.tolist(), 'pending_projection_solve_T_M': np.linalg.solve(transition, emission).tolist(), 'unit_null_contrast': null_direction.tolist(), 'null_contrast_definition': 'cross(ones(3), emission[:,0]) / norm(cross(ones(3), emission[:,0]))'},
        'checkpoints': checkpoints,
    }
    with OUTPUT.open('x') as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print('Saved ' + str(OUTPUT.relative_to(ROOT)), flush=True)


if __name__ == '__main__':
    main()
