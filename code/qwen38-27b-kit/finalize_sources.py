"""Package the committed scorer extension, requiring b0439f7bb as its ancestor."""
import argparse
import json
from pathlib import Path
import subprocess
from common import HERE, atomic, sha

RUNTIME_PATHS=('src','ggml','common','tools/server')


def validate_changes(changed):
    # The scorer commit includes its worker's top-level documentation note.
    assert changed and all(path=='NOTES-A.md' or path.startswith(('tools/ridgefill/','tests/','docs/'))
                           for path in changed),changed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    def git(*cmd):
        return subprocess.check_output(['git','-C',str(args.repo),*cmd],text=True).strip()
    base=git('rev-parse','b0439f7bb')
    commit=git('rev-parse',args.commit)
    subprocess.run(['git','-C',str(args.repo),'merge-base','--is-ancestor',base,commit],check=True)
    assert git('rev-list','--count',f'{base}..{commit}')=='1','Expected exactly one scorer commit over b0439f7bb'
    changed=git('diff','--name-only',base,commit).splitlines()
    validate_changes(changed)
    subprocess.run(['git','-C',str(args.repo),'diff','--exit-code',base,commit,'--',*RUNTIME_PATHS],check=True)
    source=git('show',f'{commit}:tools/ridgefill/score.cpp')
    assert '--save-logprobs' in source and '--kl-against' in source
    stock=HERE/'sources/stock-1242be66f.tar.gz'
    target=HERE/'sources/e3.tar.gz'
    temporary=target.with_name(target.name+'.tmp')
    with temporary.open('wb') as f:
        subprocess.run(['git','-C',str(args.repo),'archive','--format=tar.gz',commit],stdout=f,check=True)
    temporary.replace(target)
    atomic(HERE/'sources/LOCK.json',dict(stock_commit=git('rev-parse','1242be66f'),e3_base=base,e3_commit=commit,
           changed_from_base=changed,archives={stock.name:sha(stock),target.name:sha(target)},
           server_runtime_commit=base,server_runtime_paths=list(RUNTIME_PATHS),server_runtime_diff_empty=True,
           scorer_contract=dict(reference_payload_dtype='float32',dtype_in_header=True,documents_per_invocation=1),
           instrumentation='Pure builds use unmodified archive contents for all measurements. Separate instrumented builds apply the raw-logit observation hook for checks.py only.'))
    (HERE/'sources/SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.name}\n'
                                                  for path in (stock,target,HERE/'sources/LOCK.json')))
    print(json.dumps(dict(commit=commit,archive_sha256=sha(target))))

if __name__=='__main__':
    main()
