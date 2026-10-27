"""Shared CUDA runner, provenance, atomic records and seeded paired bootstrap."""
import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
E3_FLAGS = ['--ridgefill-mode', 'quality+qattn', '--ridgefill-tail', '4096', '--ridgefill-min-prompt', '8192',
            '--ridgefill-share', '0.5', '--ridgefill-digits', 'on', '--ridgefill-special-exact', 'on',
            '--ridgefill-alpha', '0', '--ridgefill-qattn-share', '0.2', '--ridgefill-qattn-window', '64',
            '--ridgefill-header-rows', '0', '--ridgefill-qattn-window-mode', 'last', '--ridgefill-qattn-turn-cap', '256',
            '--ridgefill-s1-window', '256', '--ridgefill-s1-factor', '1.0', '--ridgefill-vtail', 'on',
            '--ridgefill-vtail-rows', '512', '--ridgefill-lexical', 'on', '--ridgefill-lexical-window', '64',
            '--ridgefill-lexical-min-words', '4', '--ridgefill-lexical-span-rows', '80',
            '--ridgefill-lexical-max-occurrences', '2', '--ridgefill-stream-projector', 'off', '--ridgefill-trim-kv', 'on']
E3_FIELDS = dict(ridgefill=True)
for key, value in zip(E3_FLAGS[::2], E3_FLAGS[1::2]):
    if key in ('--ridgefill-stream-projector', '--ridgefill-trim-kv'):
        continue
    try:
        value = float(value) if '.' in value else int(value)
    except ValueError:
        value = {'on': True, 'off': False}.get(value, value)
    E3_FIELDS[key[2:].replace('-', '_')] = value


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@functools.lru_cache(maxsize=4096)
def _sha(path, size, mtime_ns):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha(path):
    path=Path(path).resolve()
    stat=path.stat()
    # Hash each immutable artifact once per worker, while detecting changed files.
    return _sha(path,stat.st_size,stat.st_mtime_ns)


def canonical(body):
    return {k: v for k, v in body.items() if not k.startswith('ridgefill')}


def request(body, arm):
    result = dict(canonical(body))
    if arm in ('e3', 'resident', 'streamed', 'identity'):
        result.update(E3_FIELDS)
    elif arm == 'loaded-off':
        result.update(ridgefill=False, ridgefill_mode='off')
    assert canonical(result) == canonical(body)
    return result


def bootstrap(groups, repeats=10000, seed=1):
    """Equal-weight group means, sampling paired item differences within each group."""
    groups = list(groups)
    assert groups and all(groups)
    assert all(math.isfinite(v) for g in groups for v in g)
    rng = random.Random(seed)
    bs = sorted(sum(sum(rng.choice(g) for _ in g) / len(g) for g in groups) / len(groups)
                for _ in range(repeats))
    return dict(mean=statistics.mean(statistics.mean(g) for g in groups),
                ci95=[bs[int(.025 * repeats)], bs[min(repeats-1, int(.975 * repeats))]],
                resamples=repeats, seed=seed)


def distribution(values):
    if not values:
        return None
    v = sorted(values)
    assert all(math.isfinite(x) for x in v)
    return dict(n=len(v), mean=statistics.mean(v), median=statistics.median(v),
                p99=v[math.ceil(.99 * len(v))-1], maximum=v[-1], minimum=v[0])


def parser(description):
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--out', type=Path, default=HERE / 'results')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--gpus', type=int, choices=(2,4), default=4, help='Physical GPU count / shard count')
    p.add_argument('--port', type=int, default=18200)
    p.add_argument('--model', type=Path, default=HERE / 'models/Qwen3.8-27B-UD-Q4_K_XL.gguf')
    p.add_argument('--projector', type=Path, default=HERE / 'models/ridgefill-projector-qwen3.8-27b-i8-S16.gguf')
    p.add_argument('--identity-projector', type=Path, default=HERE / 'models/identity-projector-qwen3.8-27b-i8-S16.gguf')
    p.add_argument('--identity-only', action='store_true', help='Add only identity measurements to completed stock/E3 results')
    p.add_argument('--stock-bin', type=Path, default=HERE / 'build/pure/stock/bin/llama-server')
    p.add_argument('--e3-bin', type=Path, default=HERE / 'build/pure/e3/bin/llama-server')
    p.add_argument('--scorer', type=Path, default=HERE / 'build/pure/e3/bin/llama-ridgefill-score')
    return p


def validate_binary(binary,variant,arm):
    binary=Path(binary).resolve()
    build=binary.parent.parent
    manifest=load(build/'BUILD-MANIFEST.json')
    assert manifest['variant']==variant and manifest['arm']==arm, 'Wrong binary build variant/arm'
    assert str(binary.relative_to(build)) in manifest['files'], 'Binary absent from verified build manifest'
    for name,digest in manifest['files'].items():
        assert sha(build/name)==digest, f'Build artifact changed: {build/name}'
    lock=load(HERE/'sources/LOCK.json')
    archive='stock-1242be66f.tar.gz' if arm=='stock' else 'e3.tar.gz'
    assert manifest['source_archive_sha256']==lock['archives'][archive], 'Build uses a different source archive'
    return manifest


def server_binary(args,arm,instrumented=False):
    selected='stock' if arm=='stock' else 'e3'
    key=selected+('_instrumented_bin' if instrumented else '_bin')
    binary=getattr(args,key)
    validate_binary(binary,'instrumented' if instrumented else 'pure',selected)
    return binary


def env(gpu, binary):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), LD_LIBRARY_PATH=str(Path(binary).resolve().parent))
    for key in list(e):
        if key.startswith('LLAMA_RIDGEFILL'):
            del e[key]
    e.pop('RIDGEFILL_AUDIT_LOGITS',None)
    return e


def provenance(folder, args, binaries):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'nvidia-smi-q.txt').open('w') as f:
        subprocess.run(['nvidia-smi', '-q'], stdout=f, stderr=subprocess.STDOUT, check=True)
    paths = set(Path(b).resolve() for b in binaries)
    for b in list(paths):
        paths.update(p.resolve() for p in b.parent.glob('*.so*'))
    atomic(folder / 'provenance.json', dict(time=time.time(), gpu=args.gpu,
           binaries={str(b): sha(b) for b in sorted(paths)}, model=sha(args.model), projector=sha(args.projector),
           identity_projector=sha(args.identity_projector) if getattr(args,'identity_only',False) else None,
           data_manifest=sha(DATA / 'SHA256SUMS'), source_manifest=load(HERE / 'sources/LOCK.json')))


def post(port, path, body=None, timeout=3600):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
          data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        value = json.load(r)
    if isinstance(value, dict) and 'error' in value:
        raise RuntimeError(value['error'])
    return value


def stream(port, body, path='/completion'):
    started = time.monotonic()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={'Content-Type': 'application/json'})
    events, first, delivered = [], None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b'data:'):
                continue
            raw = line[5:].strip()
            if raw == b'[DONE]':
                break
            v = json.loads(raw)
            if 'error' in v:
                raise RuntimeError(v['error'])
            at = time.monotonic() - started
            events.append(dict(t=at, data=v))
            count = v.get('tokens_predicted', delivered)
            if count > delivered or v.get('tokens') or v.get('content'):
                if first is None:
                    first = at
            delivered = count
    assert first is not None and events and events[-1]['data'].get('stop'), 'Incomplete SSE response'
    return events, first, time.monotonic() - started


class VRAM:
    def __init__(self, folder, pid):
        self.path, self.pid = Path(folder) / 'vram.jsonl', pid
        self.stop = threading.Event()
        self.peak = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            with self.path.open('w') as f:
                while not self.stop.is_set():
                    data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                                                    '--format=csv,noheader,nounits'], text=True)
                    entries = []
                    for line in data.splitlines():
                        cols = [x.strip() for x in line.split(',')]
                        if len(cols) == 3 and cols[0] == str(self.pid):
                            used = int(cols[1])
                            self.peak = max(self.peak, used)
                            entries.append(dict(pid=self.pid, used_mib=used, uuid=cols[2]))
                    f.write(json.dumps(dict(t=time.time(), processes=entries)) + '\n')
                    f.flush()
                    self.stop.wait(.25)
        except BaseException as e:
            self.error = e

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.error:
            raise RuntimeError('VRAM sampler failed') from self.error


@contextlib.contextmanager
def server(args, arm, folder, ctx=33792, slots=1, unified=False, extra_env=None, instrumented=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    binary = server_binary(args,arm,instrumented=instrumented)
    provenance(folder, args, [binary])
    command = [str(binary), '-m', str(args.model), '-ngl', '99', '-t', '4', '-b', '2048', '-ub', '512',
               '-ctk', 'f16', '-ctv', 'f16', '-fa', 'on', '-c', str(ctx), '-np', str(slots), '--jinja',
               '--reasoning', 'off', '--offline', '--verbosity', '4',
               '--cache-ram', '0', '--host', '127.0.0.1', '--port', str(args.port)]
    if unified:
        command += ['--kv-unified']
    if arm in ('e3', 'resident', 'streamed', 'identity'):
        flags = list(E3_FLAGS)
        flags[flags.index('--ridgefill-stream-projector')+1] = 'on' if arm == 'streamed' else 'off'
        command += ['--ridgefill', str(args.identity_projector if arm == 'identity' else args.projector), *flags]
    elif arm == 'loaded-off':
        command += ['--ridgefill', str(args.projector), '--ridgefill-mode', 'off']
    e = env(args.gpu, binary)
    assert instrumented or not extra_env or 'RIDGEFILL_AUDIT_LOGITS' not in extra_env,'Raw logit dump requires a checks-only build'
    e.update(extra_env or {})
    atomic(folder / 'command.json', dict(command=command, environment={k:e[k] for k in e if k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','RIDGEFILL_AUDIT_LOGITS')}))
    with (folder / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, env=e, stdout=log, stderr=subprocess.STDOUT)
        sampler = VRAM(folder, proc.pid)
        try:
            with sampler:
                for _ in range(600):
                    if proc.poll() is not None:
                        raise RuntimeError(f'Server died ({proc.returncode}); see {folder}/server.log')
                    try:
                        if post(args.port, '/health', timeout=2).get('status') == 'ok':
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError('Server startup exceeded 600 s')
                yield proc, sampler
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            atomic(folder / 'exit.json', dict(rc=proc.returncode, peak_mib=sampler.peak))


def assert_idle(except_gpu=None,gpu_count=None):
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    if gpu_count is not None:
        assert {int(line.split(',')[0]) for line in gpu_rows.splitlines()}==set(range(gpu_count)), 'GPU inventory differs from requested count'
    allowed = {line.split(',')[1].strip() for line in gpu_rows.splitlines()
               if except_gpu is not None and int(line.split(',')[0]) == except_gpu}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert all(line.split(',')[0].strip() in allowed for line in apps.splitlines() if line.strip()), 'Other GPUs have active compute processes'


def run_command(command):
    """Run a child session and forward interruption through its cleanup path."""
    proc=subprocess.Popen(command,start_new_session=True)
    try:
        rc=proc.wait()
        if rc:
            raise subprocess.CalledProcessError(rc,command)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
