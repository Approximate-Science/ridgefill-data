"""Load-time checks against unreliable host RAM (seen on the development machine 2026-10-03: 64-byte runs of random bits in
page-cache pages of two projector files that were intact on disk, heap-corruption aborts). The host runs 2x64 GB
non-ECC DDR5-6400 at its rated speed; the corruption cause is not established.

A file is hashed as this process reads it (through the page cache, where corrupted pages live) and compared with a
reference. On a mismatch it is read again from the disk itself (O_DIRECT, around the page cache): if the disk copy
matches the reference, RAM was at fault and the caller reloads from clean pages; if not, the file itself differs
from its reference and IntegrityError refuses it.
Models: the reference is <model dir>/SHA256SUMS (sha256sum format, every top-level *.safetensors and *.json),
written once from O_DIRECT reads. Projectors: the sha256 recorded when the file was written (results/*.json
provenance, recorded_sha256), else the disk copy.
"""
import errno
import hashlib
import json
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 1 << 24       # 16 MiB reads: a multiple of every O_DIRECT alignment
THREADS = 4           # files hashed in parallel (hashlib and reads release the GIL)
MANIFEST = "SHA256SUMS"


class IntegrityError(RuntimeError):
    """A file differs from its reference on disk too, or stays corrupted in RAM after a reload: never use it."""


def log(msg):
    print(time.strftime("%H:%M:%S"), "integrity:", msg, flush=True)


def _read_direct(path, sink):
    """Feed sink() the file's bytes read with O_DIRECT, so they come from the disk and not from cached pages (which
    other processes' mappings can keep resident). A filesystem without O_DIRECT gets its clean cached pages dropped
    (posix_fadvise DONTNEED) before a normal read."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    buf = mmap.mmap(-1, CHUNK)          # page-aligned, as O_DIRECT needs
    try:
        while n := os.readv(fd, [buf]):
            with memoryview(buf) as view:
                sink(view[:n])
    finally:
        os.close(fd)
        buf.close()


def read_direct(path):
    out = bytearray()
    _read_direct(path, out.extend)
    return bytes(out)


def file_sha256(path, direct=False):
    """sha256 of the file as this process reads it (through the page cache), or from the disk itself (direct)."""
    h = hashlib.sha256()
    if direct:
        _read_direct(path, h.update)
        return h.hexdigest()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def drop_cache(path):
    """Drop the file's clean cached pages (those no process has mapped), so the next read comes from the disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _hash_all(paths, direct=False):
    with ThreadPoolExecutor(THREADS) as pool:
        return dict(zip(paths, pool.map(lambda p: file_sha256(p, direct), paths)))


def model_manifest(model_dir):
    """{file name: sha256} from <model_dir>/SHA256SUMS, which must list exactly the top-level *.safetensors and
    *.json files (a truncated manifest must not pass as a short one). Written on first use, fsynced, from O_DIRECT
    reads, so the reference never comes from cached pages."""
    model_dir = Path(model_dir)
    path = model_dir / MANIFEST
    files = sorted(p for p in model_dir.iterdir() if p.suffix in (".safetensors", ".json") and p.is_file())
    if path.exists():
        sums = {name: sha for sha, name in (line.split(None, 1) for line in path.read_text().splitlines())}
        if set(sums) != {p.name for p in files}:
            raise IntegrityError(f"{path} does not list exactly the model's files (differences: "
                                 f"{sorted(set(sums) ^ {p.name for p in files})}): delete it to rebuild it from disk")
        return sums
    t0 = time.time()
    sums = {p.name: sha for p, sha in _hash_all(files, direct=True).items()}
    tmp = path.with_name(f"{MANIFEST}.{os.getpid()}.tmp")      # two jobs may start on one model at once
    with open(tmp, "w") as f:
        f.write("".join(f"{sha}  {name}\n" for name, sha in sums.items()))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    log(f"wrote {path} ({len(sums)} files, read from disk in {time.time() - t0:.1f}s)")
    return sums


def verify_model(model_dir):
    """Hash the model's files as this process reads them (after loading: the page-cache pages the weights came
    from) and compare with SHA256SUMS. Returns the files whose cached bytes differ while the disk copy matches
    (RAM corruption: drop them and reload); raises IntegrityError if a file differs on disk too."""
    model_dir = Path(model_dir)
    ref = model_manifest(model_dir)
    t0 = time.time()
    got = _hash_all([model_dir / name for name in ref])
    bad = [p.name for p, sha in got.items() if sha != ref[p.name]]
    for name in bad:
        log(f"MEMORY: {model_dir / name} read through the page cache has sha256 {got[model_dir / name][:16]}, "
            f"SHA256SUMS says {ref[name][:16]}; re-reading it from disk")
        disk = file_sha256(model_dir / name, direct=True)
        if disk != ref[name]:
            raise IntegrityError(f"{model_dir / name}: sha256 on disk {disk}, {model_dir / MANIFEST} says {ref[name]}: "
                                 f"the file changed or the manifest is wrong (delete it to rebuild it from disk)")
    gib = sum((model_dir / name).stat().st_size for name in ref) / 2**30
    log(f"{model_dir.name}: {len(ref)} files ({gib:.1f} GiB) checked against {MANIFEST} in {time.time() - t0:.1f}s"
        + (f"; {len(bad)} corrupted in RAM, intact on disk" if bad else ""))
    return bad


def recorded_sha256(path, results):
    """The sha256 that provenance in results/*.json records for a file of this name ('X' + 'X_sha256', e.g.
    projector / projector_sha256 in fit reports, or 'path' + 'sha256'); None if there is none, or if records
    disagree (one of them hashed a corrupted read, so none can be trusted)."""
    name = Path(path).name
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and Path(v).name == name:
                    sha = o.get("sha256" if k == "path" else k + "_sha256")
                    if isinstance(sha, str):
                        found.add(sha)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for f in sorted(Path(results).glob("*.json")):
        walk(json.loads(f.read_text()))
    if len(found) > 1:
        log(f"{name}: results record {len(found)} different sha256 ({', '.join(s[:12] for s in sorted(found))}); "
            f"using the disk copy as the reference")
    return found.pop() if len(found) == 1 else None


def read_verified(path, ref):
    """(bytes, sha256) of the file: read through the page cache, checked against ref (None: the disk copy, read
    around the cache). On a mismatch the bytes come from the disk itself and its cached pages are dropped;
    IntegrityError if the disk copy differs from ref too."""
    path = Path(path)
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if ref is None:
        ref = file_sha256(path, direct=True)
    if got == ref:
        return data, got
    log(f"MEMORY: {path} read through the page cache has sha256 {got[:16]}, reference {ref[:16]}; "
        f"re-reading it from disk")
    data = read_direct(path)
    got = hashlib.sha256(data).hexdigest()
    if got != ref:
        raise IntegrityError(f"{path}: sha256 on disk {got}, reference {ref}: the file changed after it was recorded")
    drop_cache(path)
    return data, got
