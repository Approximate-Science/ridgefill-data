"""Run all identity measurements in budget priority order, with durable checkpoints."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import DATA, HERE, atomic, load, rows, sha, run_command
from reproduction import require_reproduction

STEPS = ('ruler', 'kl', 'ttft', 'longbench')


def result_files(out, step):
    out = Path(out)
    if step == 'ruler':
        return [out/'ruler/identity'/(item['id']+'.json') for item in rows(DATA/'ruler.jsonl')]
    if step == 'kl':
        return [out/'kld-identity'/str(item['length'])/(item['id']+'.json') for item in load(DATA/'kld-index.json')]
    if step == 'ttft':
        return [out/'timing-identity'/str(length)/arm/'result.json' for length in (16384,32768,65536,120000)
                for arm in ('stock','resident','streamed','identity')]+[out/'timing-identity/SESSION.json']
    if step == 'longbench':
        return [out/'longbench/identity'/(item['id']+'.json') for item in rows(DATA/'longbench-saved.jsonl')]
    if step == 'report':
        return [out/'SCORECARD-IDENTITY.json', out/'SCORECARD-IDENTITY.md']
    raise ValueError(step)


def completion(out, step):
    out = Path(out)
    files = result_files(out, step)
    assert len(set(files)) == len(files), 'Duplicate expected result IDs'
    # Atomic item writes are complete JSON documents. Reject missing/truncated
    # files before claiming a whole step; never remove or rewrite finished items.
    for path in files:
        if path.suffix == '.json':
            load(path)
    return dict(step=step, count=len(files), manifest_sha256=sha(out/'IDENTITY-MANIFEST.json'),
                files={str(path.relative_to(out)):sha(path) for path in files})


def marker_path(out, step):
    number = (*STEPS, 'report').index(step)+1
    return Path(out)/'identity-steps'/f'{number:02d}-{step}.complete.json'


def run_step(out, step, command):
    path = marker_path(out, step)
    if path.exists():
        saved = load(path)
        current = completion(out, step)
        assert {k:saved[k] for k in current} == current, f'Completed {step} results or manifest changed'
        print(f'IDENTITY SKIP {step}: verified {current["count"]} completed results', flush=True)
        return
    started = time.time()
    print(f'IDENTITY START {step}', flush=True)
    run_command(command)
    value = completion(out, step)
    atomic(path, dict(**value, started_at=started, completed_at=time.time()))
    print(f'IDENTITY COMPLETE {step}: {value["count"]} results; {path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpus',type=int,choices=(2,4),default=4)
    args = p.parse_args()
    args.out = args.out.resolve()
    # run_all.sh performs preflight before entering this orchestrator.
    manifest=load(args.out/'IDENTITY-MANIFEST.json')
    assert manifest['gpu_count']==args.gpus, 'GPU count differs from identity manifest'
    require_reproduction(args.out)
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    commands = {
        'ruler': [sys.executable, str(HERE/'drive.py'), 'ruler_replay.py', str(args.out), '--identity-only','--gpus',str(args.gpus)],
        'kl': [sys.executable, str(HERE/'drive.py'), 'kld_run.py', str(args.out), '--identity-only','--gpus',str(args.gpus)],
        'ttft': [sys.executable, str(HERE/'timing.py'), '--identity-only', '--gpu', '0', '--out', str(args.out),'--gpus',str(args.gpus)],
        'longbench': [sys.executable, str(HERE/'drive.py'), 'longbench_identity.py', str(args.out), '--identity-only','--gpus',str(args.gpus)],
        'report': [sys.executable, str(HERE/'report.py'), '--identity-only', '--out', str(args.out),'--gpus',str(args.gpus)],
    }
    for step in (*STEPS, 'report'):
        run_step(args.out, step, commands[step])


if __name__ == '__main__':
    main()
