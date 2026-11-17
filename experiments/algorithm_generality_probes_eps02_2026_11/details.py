"""Verified B2 archive for ignored report inputs; no model execution."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
DETAILS = ROOT/'artifacts/report_details'
MANIFEST = ROOT/'results/details_manifest.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checked(data, row):
    if len(data) != row['size_bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('artifact integrity mismatch')
    return data


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != name:
        raise ValueError('unsafe archive path')
    return path


def verify(directory):
    index = read(directory/'index.json')
    if not index['files'] or len({r['relative_path'] for r in index['files']}) != len(index['files']):
        raise ValueError('empty or duplicate archive inventory')
    for row in index['files']:
        checked((directory/relative(row['relative_path'])).read_bytes(), row)
    return len(index['files'])


def extract(archive, destination):
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)/'payload'
        staging.mkdir()
        with tarfile.open(archive) as bundle:
            names = set()
            for member in bundle:
                name = relative(member.name)
                if not member.isfile() or member.name in names:
                    raise ValueError('non-file or duplicate archive member')
                names.add(member.name)
                path = staging/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bundle.extractfile(member).read())
        verify(staging)
        expected = {r['relative_path'] for r in read(staging/'index.json')['files']} | {'index.json'}
        if names != expected:
            raise ValueError('archive inventory mismatch')
        staging.rename(destination)


def restore(destination=DETAILS):
    from harness.storage.b2 import B2StorageConfig
    manifest = read(MANIFEST)
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError('B2 settings required to restore report details')
    row, = manifest['files']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        archive = Path(temporary)/'payload.tar.gz'
        config.s3_client().download_file(manifest['bucket'], row['key'], str(archive))
        checked(archive.read_bytes(), row)
        extract(archive, destination)
    print(f'Restored verified report details: {verify(destination)} indexed files')


def ensure_details():
    if not DETAILS.exists():
        restore(DETAILS)
    elif not (DETAILS/'index.json').exists():
        if not any(p.is_file() for p in DETAILS.rglob('*')):
            raise ValueError('empty report details')
        references = ('probes/reinforce_reference.json', 'steering/reinforce_reference.json')
        if any(not (DETAILS/name).exists() for name in references):
            with tempfile.TemporaryDirectory(dir=DETAILS.parent) as temporary:
                archived = Path(temporary)/'archived'
                restore(archived)
                for name in references:
                    destination = DETAILS/name
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(archived/name, destination)
        index_details(DETAILS)
    verify(DETAILS)
    return DETAILS


def index_details(directory):
    files = [p for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'index.json' and 'aggregates' not in p.relative_to(directory).parts]
    if not files:
        raise ValueError('empty report details')
    index = {'files': [dict(relative_path=p.relative_to(directory).as_posix(),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size_bytes=p.stat().st_size) for p in files]}
    write(directory/'index.json', index)
    return files


def publish():
    from harness.storage.b2 import B2StorageConfig, upload_artifact_directory
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError('B2 settings required to publish report details')
    files = index_details(DETAILS)
    verify(DETAILS)
    directory = ROOT/'artifacts/report_archive'
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory/'payload.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in [DETAILS/'index.json', *files]:
            bundle.add(path, arcname=path.relative_to(DETAILS).as_posix(), recursive=False)
    run_id = 'report-details-'+hashlib.sha256(archive.read_bytes()).hexdigest()[:16]
    context = SimpleNamespace(artifacts_dir=ROOT/'artifacts', experiment_dir=ROOT, run_id=run_id)
    manifest = upload_artifact_directory(context, directory, config=config)
    row, = manifest['files']
    client = config.s3_client()
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
        downloaded = Path(temporary)/'payload.tar.gz'
        client.download_file(manifest['bucket'], row['key'], str(downloaded))
        checked(downloaded.read_bytes(), row)
        extract(downloaded, Path(temporary)/'restored')
    manifest.update(download_sha256_verified=True, indexed_files=len(files),
                    canonical_manifest_key=manifest['prefix']+'/metadata/durability_manifest.json')
    write(MANIFEST, manifest)
    client.upload_file(str(MANIFEST), manifest['bucket'], manifest['canonical_manifest_key'])
    remote = client.get_object(Bucket=manifest['bucket'], Key=manifest['canonical_manifest_key'])['Body'].read()
    if json.loads(remote) != manifest:
        raise ValueError('remote durability manifest mismatch')
    print(f'B2 archive verified by download: {len(files)} files, {row["size_bytes"]} bytes, SHA-256 {row["sha256"]}')


def checkpoint(run_id, row):
    path = ROOT/'artifacts/checkpoints'/run_id/f'step_{row["step"]:010d}.npz'
    if not path.exists():
        from harness.storage.b2 import B2StorageConfig
        config = B2StorageConfig.from_env()
        if config is None:
            raise RuntimeError('B2 settings required to restore checkpoints')
        client = config.s3_client()
        manifest = json.loads(client.get_object(Bucket=row['bucket'], Key=row['manifest_key'])['Body'].read())
        entry, = [r for r in manifest['files'] if r['key'] == row['key']]
        if any(entry[k] != row[k] for k in ('sha256', 'size_bytes')):
            raise ValueError('canonical checkpoint manifest mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(row['bucket'], row['key'], str(path))
    checked(path.read_bytes(), row)


def source(name, digest):
    candidates = [ROOT/name]
    if name in ('analyze.py', 'steering_run.py'):
        candidates.append(DETAILS/'executed_sources'/name)
    if not any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in candidates):
        raise ValueError(f'executed source mismatch: {name}')


def records(path, sections):
    lines = ['{']
    for i, (key, rows) in enumerate(sections.items()):
        lines.append('  '+json.dumps(key)+': [')
        lines.extend('    '+json.dumps(row, allow_nan=False)+(',' if j+1 < len(rows) else '') for j, row in enumerate(rows))
        lines.append('  ]'+(',' if i+1 < len(sections) else ''))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join([*lines, '}'])+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        publish()
    else:
        ensure_details()
        print(f'Report detail integrity OK: {verify(DETAILS)} indexed files')


if __name__ == '__main__':
    main()
