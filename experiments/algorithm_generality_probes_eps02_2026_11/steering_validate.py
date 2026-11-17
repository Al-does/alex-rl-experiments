"""Audit complete inventories, checkpoint hashes, controls and pulse cutoffs."""

import hashlib

from .analyze import FINAL, ROOT, SEEDS, read
from .steering_report import ladder_rows, moving_rows
from .details import DETAILS, ensure_details, checkpoint, source

import numpy as np


def main():
    ensure_details()
    runs = [read(p) for p in (DETAILS/'steering/agents').glob('*.json')]
    assert len(runs) == 40
    inventories = set()
    controls = {}
    for r in runs:
        p = r['adapter_provenance']
        kind = p['kind']
        variant = r['variant'] if kind == 'ladder' else 'joint_reward'
        key = kind, p['algorithm'], variant, p['training_seed']
        assert key not in inventories
        inventories.add(key)
        assert not p['smoke'] and p['training_seed'] in SEEDS
        assert all('cpu' in d.lower() for d in p['devices'])
        c = p['checkpoint_verified']
        assert c['step'] == FINAL
        checkpoint(p['run_id'], c)
        for name in ('analyze.py', 'steering_run.py'):
            source(name, p['source_sha256'][name])
        for name in ('steer_core.py', 'steer_prediction.py',
                     'steer_pulse.py', 'steer_moving.py', 'steer_moving_base.py', 'steer_storage.py', 'steer_reference.py'):
            body = (ROOT/name).read_bytes()
            versions = (body, body+b'\n') if name.startswith('steer_') else (body,)
            assert p['source_sha256'][name] in {hashlib.sha256(v).hexdigest() for v in versions}, name
        driver = (ROOT/'steer_ladder.py').read_bytes()
        headers = (driver, driver.replace(b'smoke=smoke,\n', b'smoke=smoke, \n'))
        assert p['source_sha256']['steer_ladder.py'] in {hashlib.sha256(body).hexdigest() for body in headers}
        if kind == 'ladder':
            assert r['protocol']['fit_rows'] == 12000
            assert r['protocol']['tune_seed'] == 1042 and r['protocol']['tune_episodes'] == 16
            assert r['protocol']['confirm_seed'] == 2042 and r['protocol']['confirm_episodes'] == 32
            rows = ladder_rows(r)
            assert len(rows) == 42
            for v in rows.values():
                for metric in ('occupancy', 'action_change', 'shift_norm', 'relative_shift'):
                    assert np.isfinite(v[metric])
                assert 0 <= v['occupancy'] <= 1 and 0 <= v['action_change'] <= 1
            value = rows['random@-1']['occupancy']
            control_key = kind, variant
            for name, result in r['pulse']['conditions'].items():
                limit = {'1': 96, '2': 64, '3': 32}[name[-1]]
                last = result['measured']['last_nonzero_lag']
                assert last is None or last <= limit
            assert len(r['pulse']['conditions']) == 6
        else:
            assert set(r['rewards']) == {'moving'}
            assert r['rewards']['moving']['steered_factors'] == [False, True]
            assert r['fit_rows'] == 60000 and r['moving_reward_period'] == 256
            assert r['tune_seed'] == 1042 and r['tune_episodes'] == 64
            assert r['confirm_seed'] == 2042 and r['confirm_episodes'] == 256
            rows = moving_rows(r['rewards']['moving']['confirmation'])
            assert len(rows) == 28
            for v in rows.values():
                assert np.isfinite(v['total_reward']) and 0 <= v['total_reward'] <= 2
            value = [rows[k]['total_reward'] for k in ('random', 'exact_belief_myopic_map', 'exact_belief_discounted_map')]
            control_key = kind, variant
        if control_key in controls:
            np.testing.assert_array_equal(controls[control_key], value)
        controls[control_key] = value
    expected = {(kind, algorithm, variant, seed) for kind in ('ladder', 'moving')
                for algorithm in ('ppo', 'sac_autoalpha') for variant in ((1, 2, 3) if kind == 'ladder' else ('joint_reward',))
                for seed in SEEDS}
    assert inventories == expected
    ref = read(DETAILS/'steering/reinforce_reference.json')
    for r in ref['moving']:
        value = [r['confirmation'][k]['total_reward'] for k in ('random', 'exact_belief_myopic_map', 'exact_belief_discounted_map')]
        np.testing.assert_array_equal(value, controls['moving', 'joint_reward'])
    print('Steering audit OK: 40 unique final actors, 40 checkpoint SHA-256/size checks, CPU-only, executed source hashes verified with documented whitespace-only normalization')
    print('Protocol audit OK: original row/episode budgets and seeds, 30 complete ladder pulse batteries with finite-cache cutoffs, moving reward only')
    print('Control audit OK: identical random-action references within each task; moving random/exact-belief references exactly match archived REINFORCE')


if __name__ == '__main__':
    main()
