"""Audit slim reports against verified historical inputs and aggregate baselines."""

from .details import DETAILS, ROOT, checkpoint, ensure_details, read, source
from .report import compact_summary as compact_probes
from .steering_report import compact_summary as compact_steering


def main():
    ensure_details()
    runs = [read(p) for p in sorted((DETAILS/'probes/agents').glob('*.json'))]
    expected = {(a, v, s) for a in ('ppo', 'sac_autoalpha') for v in (1, 2, 3) for s in (42, 44, 46, 49, 52)}
    assert len(runs) == 30 and {(r['algorithm'], r['variant'], r['seed']) for r in runs} == expected
    for run in runs:
        assert not run['smoke'] and run['final_self_replay_max_abs'] == 0
        assert {c['step'] for c in run['checkpoints']} == {0, 2012217344}
        for c in run['checkpoints']:
            checkpoint(run['run_id'], c)
        for name, digest in run['analysis_source_sha256'].items():
            if name == 'analyze.py' or name.startswith('paper_'):
                source(name, digest)
    probe = read(DETAILS/'aggregates/probes.json')
    steering = read(DETAILS/'aggregates/steering.json')
    assert probe == read(DETAILS/'baseline/probes.json')
    assert steering == read(DETAILS/'baseline/steering.json')
    assert read(ROOT/'results/summary.json') == compact_probes(probe)
    agents = [read(p) for p in sorted((DETAILS/'steering/agents').glob('*.json'))]
    reference = read(DETAILS/'steering/reinforce_reference.json')
    assert read(ROOT/'results/steering/summary.json') == compact_steering(steering, agents, reference)
    print('Probe audit OK: 30 agents, 60 checkpoint SHA-256/size checks, exact final self-replay, historical collector/probe hashes verified')
    print('Slim report audit OK: probe/steering aggregates exactly equal archived pre-slim baselines; compact summaries reproduce all retained aggregates')


if __name__ == '__main__':
    main()
