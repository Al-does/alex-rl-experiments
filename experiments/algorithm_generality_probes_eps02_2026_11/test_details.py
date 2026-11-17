import hashlib
import io
import json
import tarfile

import pytest

from . import details
from .report import compact_summary as compact_probes
from .steering_report import compact_summary as compact_steering


def bundle(path, name='inputs/one.json', duplicate=False):
    payload = b'{"value": 3}'
    index = json.dumps({'files': [dict(relative_path=name, size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest())]}).encode()
    with tarfile.open(path, 'w:gz') as archive:
        for member_name, data in [('index.json', index), (name, payload), *([(name, payload)] if duplicate else [])]:
            member = tarfile.TarInfo(member_name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_archive_roundtrip_and_refuses_overwrite(tmp_path):
    archive = tmp_path/'payload.tar.gz'
    bundle(archive)
    destination = tmp_path/'restored'
    details.extract(archive, destination)
    assert details.verify(destination) == 1
    with pytest.raises(FileExistsError):
        details.extract(archive, destination)
    (destination/'inputs/one.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity mismatch'):
        details.verify(destination)


@pytest.mark.parametrize('name,duplicate', [('../escape.json', False), ('/absolute.json', False), ('one.json', True)])
def test_archive_rejects_unsafe_or_duplicate_members(tmp_path, name, duplicate):
    archive = tmp_path/'payload.tar.gz'
    bundle(archive, name, duplicate)
    with pytest.raises(ValueError):
        details.extract(archive, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_empty_inventory_and_checksum_errors(tmp_path):
    details.write(tmp_path/'index.json', {'files': []})
    with pytest.raises(ValueError, match='empty'):
        details.verify(tmp_path)
    with pytest.raises(ValueError, match='integrity'):
        details.checked(b'tampered', {'size_bytes': 1, 'sha256': 'bad'})
    with pytest.raises(ValueError, match='integrity'):
        details.checked(b'tampered', {'size_bytes': 8, 'sha256': 'bad'})


def test_compact_summaries_keep_aggregates_not_per_agent_detail(tmp_path):
    probes = {'protocol': {}, 'groups': {'cohort': {'means': {'error_belief': .1}, 'n': 5, 'per_seed': {'42': 'large'}}}}
    compact = compact_probes(probes)
    assert compact['groups']['cohort'] == {'means': {'error_belief': .1}, 'n': 5}
    assert probes['groups']['cohort']['per_seed']
    label = 'PPO (5): Reward-State Quotient'
    pulse = {'measured': {'absolute_at_lag': {'0': [.1, .2], '1': [.01, .02]}, 'last_nonzero_lag': 96},
             'full_filter': {'absolute_at_lag': {'128': [.001, .002]}}}
    agents = [{'variant': 2, 'adapter_provenance': {'algorithm': 'ppo', 'kind': 'ladder'},
               'pulse': {'conditions': {'mirror_block1': pulse}}}]
    groups = {'ladder': {label: {'n': 1, 'per_seed': {'42': 'large'},
              'means': {'intact@-1': {'occupancy': .4, 'episode_occupancy': [1, 2]}}}}, 'moving': {}}
    steering = compact_steering(groups, agents, {'ladder': []})
    assert steering['ladder'] == [{'group': label, 'condition': 'intact@-1', 'n': 1, 'occupancy': .4}]
    assert steering['pulses'][0]['lag0'] == [.1, .2]
    assert steering['pulses'][0]['last_nonzero_max'] == 96
    assert compact_steering({'ladder': {}, 'moving': {}}, [], {}) == {'ladder': [], 'moving': [], 'pulses': [], 'numerical_ladder_reference': []}
    path = tmp_path/'summary.json'
    details.records(path, steering)
    assert details.read(path) == steering
    assert len(path.read_text().splitlines()) < 20
    with pytest.raises(ValueError):
        details.records(path, {'bad': [float('nan')]})


def test_historical_source_hash_is_explicit_and_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(details, 'ROOT', tmp_path)
    monkeypatch.setattr(details, 'DETAILS', tmp_path/'details')
    path = details.DETAILS/'executed_sources/analyze.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'historical collector')
    details.source('analyze.py', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='source mismatch'):
        details.source('analyze.py', 'incorrect')


def test_restore_requires_credentials(tmp_path, monkeypatch):
    from harness.storage.b2 import B2StorageConfig
    monkeypatch.setattr(B2StorageConfig, 'from_env', lambda: None)
    monkeypatch.setattr(details, 'ROOT', tmp_path)
    monkeypatch.setattr(details, 'MANIFEST', tmp_path/'manifest.json')
    details.write(details.MANIFEST, {})
    with pytest.raises(RuntimeError, match='B2 settings required'):
        details.restore(tmp_path/'details')
    with pytest.raises(RuntimeError, match='B2 settings required'):
        details.checkpoint('run', {'step': 0})


def test_fresh_collection_is_indexed_without_replacing_agents(tmp_path, monkeypatch):
    monkeypatch.setattr(details, 'DETAILS', tmp_path/'fresh')
    agent = details.DETAILS/'probes/agents/fresh.json'
    details.write(agent, {'fresh': True})
    restored = []

    def restore(destination):
        restored.append(destination)
        details.write(destination/'probes/reinforce_reference.json', {'probe': True})
        details.write(destination/'steering/reinforce_reference.json', {'steering': True})
        details.write(destination/'probes/agents/stale.json', {'fresh': False})

    monkeypatch.setattr(details, 'restore', restore)
    assert details.ensure_details() == details.DETAILS
    assert details.verify(details.DETAILS) == 3
    assert details.read(agent) == {'fresh': True}
    assert not (agent.parent/'stale.json').exists()
    assert len(restored) == 1
    agent.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='integrity mismatch'):
        details.ensure_details()
    assert len(restored) == 1


def test_absent_details_still_restore_and_empty_collection_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(details, 'DETAILS', tmp_path/'absent')
    restored = []

    def restore(destination):
        restored.append(destination)
        details.write(destination/'input.json', {'archived': True})
        details.index_details(destination)

    monkeypatch.setattr(details, 'restore', restore)
    assert details.ensure_details() == details.DETAILS
    assert restored == [details.DETAILS]
    monkeypatch.setattr(details, 'DETAILS', tmp_path/'empty')
    details.DETAILS.mkdir()
    with pytest.raises(ValueError, match='empty report details'):
        details.ensure_details()
