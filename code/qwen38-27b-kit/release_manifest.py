"""Write the exact transfer file list and SHA256SUMS, excluding local build scratch and outputs."""
from pathlib import Path

from common import HERE, sha


def release_files(root=HERE):
    root=Path(root)
    selected=set()
    for pattern in ('*.py','*.sh','*.cpp','README.md','NOTES.md','models.sha256',
                    'cpu-gate.txt','unit-tests.txt','identity-unit-tests.txt','identity-data-verify.txt',
                    'cpu-native-smoke.txt','cpu-pure-bridge-smoke.txt','source-finalize.txt'):
        selected.update(path for path in root.glob(pattern) if path.is_file())
    for directory in ('data','sources','models','tests','vendor','validation'):
        for path in (root/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith(('.pyc','.tmp')):
                selected.add(path)
    return sorted(selected)


def main():
    files=release_files()
    names=sorted([str(path.relative_to(HERE)) for path in files]+['FILES.txt','SHA256SUMS'])
    listing=HERE/'FILES.txt'
    listing.write_text('\n'.join(names)+'\n')
    checksums=[f'{sha(path)}  {path.relative_to(HERE)}' for path in sorted(files+[listing])]
    (HERE/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
    print(f'{len(names)} transfer files; {len(checksums)} checksums (SHA256SUMS excludes itself)')
    print(f'Transfer bytes: {sum(path.stat().st_size for path in files)+listing.stat().st_size+(HERE/"SHA256SUMS").stat().st_size}')


if __name__=='__main__':
    main()
