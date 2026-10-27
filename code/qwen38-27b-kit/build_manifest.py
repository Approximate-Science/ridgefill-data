"""Verify the build's source variant against its pinned archive and bind binary/library hashes."""
import argparse
import hashlib
import os
from pathlib import Path
import tarfile

from common import atomic, sha
from instrument_sources import instrument_text

HOOK_PATH='tools/server/server-common.cpp'


def verify_sources(source,archive,variant):
    assert variant in ('pure','instrumented')
    source=Path(source)
    expected_paths=set()
    digests={}
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            name=member.name.removeprefix('./')
            if not member.isfile() and not member.issym():
                continue
            path=source/name
            expected_paths.add(name)
            if member.issym():
                assert path.is_symlink() and os.readlink(path)==member.linkname,name
                digests[name]=hashlib.sha256(member.linkname.encode()).hexdigest()
                continue
            expected=tar.extractfile(member).read()
            if variant=='instrumented' and name==HOOK_PATH:
                expected=instrument_text(expected.decode()).encode()
            assert not path.is_symlink() and path.read_bytes()==expected, f'{variant} source differs from archive: {name}'
            digests[name]=hashlib.sha256(expected).hexdigest()
    actual_paths={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() or p.is_symlink()}
    assert actual_paths==expected_paths, f'Unexpected/missing source files: {actual_paths ^ expected_paths}'
    assert HOOK_PATH in expected_paths
    return digests


def record(source,archive,build,arm,variant):
    source_hashes=verify_sources(source,archive,variant)
    build=Path(build)
    binaries=[build/'bin/llama-server']
    if variant=='pure':
        binaries += [build/'bin/llama-tokenize',build/('bin/exact-score' if arm=='stock' else 'bin/llama-ridgefill-score')]
    binaries += list((build/'bin').glob('*.so*'))
    files={str(path.relative_to(build)):sha(path) for path in sorted(binaries)}
    manifest=dict(arm=arm,variant=variant,source_archive_sha256=sha(archive),source_files=source_hashes,files=files)
    atomic(build/'BUILD-MANIFEST.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--arm',choices=('stock','e3'),required=True)
    p.add_argument('--variant',choices=('pure','instrumented'),required=True)
    args=p.parse_args()
    result=record(args.source,args.archive,args.build,args.arm,args.variant)
    print(f"Verified {args.variant} {args.arm}: {len(result['source_files'])} source files, {len(result['files'])} artifacts")


if __name__=='__main__':
    main()
