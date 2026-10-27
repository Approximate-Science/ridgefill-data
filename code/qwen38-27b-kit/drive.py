"""Run N disjoint shards; stop every worker/process group on the first failure."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import HERE


def main():
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('component',choices=('ruler_replay.py','kld_run.py','longbench_identity.py','reproduction.py'))
    p.add_argument('out',type=Path)
    p.add_argument('--identity-only',action='store_true')
    p.add_argument('--gpus',type=int,choices=(2,4),default=4)
    args=p.parse_args()
    component=args.component
    extra=['--identity-only'] if args.identity_only else []
    out=args.out.resolve()
    children=[]
    logs=[]
    try:
        for gpu in range(args.gpus):
            path=out/'workers'/f'{component}-gpu{gpu}-{time.time_ns()}.log'
            path.parent.mkdir(parents=True,exist_ok=True)
            log=path.open('w');logs.append(log)
            children.append(subprocess.Popen([sys.executable,'-u',str(HERE/component),'--gpu',str(gpu),'--port',str(18200+gpu),
                            '--workers',str(args.gpus),'--gpus',str(args.gpus),'--out',str(out),*extra],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in children):
            if any(p.returncode not in (None,0) for p in children):
                raise RuntimeError(f'{component}: worker failed; inspect {out}/workers')
            time.sleep(1)
        assert all(p.returncode==0 for p in children),'Worker failed'
    finally:
        for proc in children:
            # The worker can have exited while its server still holds GPU memory.
            try:
                os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in children:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for log in logs:log.close()

if __name__=='__main__':
    main()
