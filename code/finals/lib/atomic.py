#!/usr/bin/env python3
"""Atomic result writes with sha256 sidecars, and a results-tree hash for push receipts.

write_result(): tmp file next to the target, fsync, read back (the bytes must hash and parse back to
what we wrote; host RAM flips bits, 2026-10-03), rename, fsync the directory, then the same treatment
for the '<file>.sha256' sidecar in sha256sum -c format. A file is 'finished' only when it exists, its
sidecar exists and matches (finished()). treehash(): sha256 over the sorted (relpath, file sha256)
pairs of a results tree, so push receipts can name exactly which bytes they carried.

CLI:
  atomic.py treehash DIR                     -> the tree hash on stdout
  atomic.py sidecar FILE                     -> writes/refreshes FILE.sha256
  atomic.py check FILE...                    -> exit 0 iff every file matches its sidecar
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


class AtomicWriteError(RuntimeError):
    """The written bytes did not read back from the disk as they were written: nothing is trusted."""


def sha256_file(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_buf):
            h.update(chunk)
    return h.hexdigest()


def _atomic_bytes(path, data, attempts=2):
    """path never holds a partial write: tmp + fsync + read back + rename + dir fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            with open(tmp, "rb") as f:                     # read back from the disk
                if f.read() != data:
                    raise AtomicWriteError(f"{path}: read-back differs (attempt {attempt})")
            os.replace(tmp, path)
        except BaseException:
            tmp = Path(tmp)
            if tmp.exists():
                tmp.unlink()
            raise
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return
    raise AtomicWriteError(f"{path}: could not write it atomically")   # unreachable; for the reader


def write_bytes(path, data):
    """Public wrapper: raw atomic bytes (used for dotfile markers like .done)."""
    _atomic_bytes(path, data)


def sidecar_line(path, digest):
    """sha256sum -c format: '<hex>  <basename>'."""
    return f"{digest}  {Path(path).name}\n"


def write_result(path, obj):
    """Write obj as JSON atomically plus its '<path>.sha256' sidecar; return the sha256."""
    data = json.dumps(obj, indent=1, sort_keys=True).encode() + b"\n"
    _atomic_bytes(path, data)
    digest = hashlib.sha256(data).hexdigest()
    _atomic_bytes(f"{path}.sha256", sidecar_line(path, digest).encode())
    return digest


def read_sidecar(path):
    """The sha256 a sidecar records for path, or None (missing/corrupt sidecar)."""
    sc = Path(f"{path}.sha256")
    if not sc.exists():
        return None
    try:
        line = sc.read_text().splitlines()[0]
        hexd, name = line.split("  ", 1)
    except (ValueError, IndexError):
        return None
    if name != Path(path).name or len(hexd) != 64 or any(c not in "0123456789abcdef" for c in hexd):
        return None
    return hexd


def finished(path):
    """A result file counts as finished only with a sidecar that matches its bytes."""
    p = Path(path)
    if not p.exists():
        return False
    want = read_sidecar(p)
    return want is not None and want == sha256_file(p)


def refresh_sidecar(path):
    """(Re)write a sidecar for an existing file (the kva backend's collect pass)."""
    return _atomic_bytes(f"{path}.sha256", sidecar_line(path, sha256_file(path)).encode())


def treehash(root):
    """sha256 over '<relpath> <file sha256>' of every regular file (sidecars excluded), sorted."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".sha256"):
            h.update(f"{p.relative_to(root).as_posix()} {sha256_file(p)}\n".encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("treehash", "sidecar"):
        s = sub.add_parser(name)
        s.add_argument("path")
    c = sub.add_parser("check")
    c.add_argument("paths", nargs="+")
    args = ap.parse_args()
    if args.cmd == "treehash":
        print(treehash(args.path))
    elif args.cmd == "sidecar":
        refresh_sidecar(args.path)
    else:
        bad = [p for p in args.paths if not finished(p)]
        for p in bad:
            print(f"MISMATCH {p}: does not match its sidecar or has none", file=sys.stderr)
        sys.exit(2 if bad else 0)


if __name__ == "__main__":
    main()