"""Provenance block written into every raw result file (plan section 8.2)."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
from kva import integrity, paths

WORK = Path(os.environ.get("KVA_WORK", paths.REPO))      # results/, ledger/ (committed)
BULK = Path(os.environ.get("KVA_BULK", paths.DATA))      # proj/, stats/, sterm/ (local, not committed)


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def _baked_commit():
    """The commit a build without .git was made from (finals/build.sh writes CODE_COMMIT from the tagged commit;
    the image's code is a git archive of it, so it is clean by construction); '' if there is none."""
    f = REPO / "CODE_COMMIT"
    return f.read_text().strip() if f.is_file() else ""


# the code this process runs: HEAD and dirty state when the module was imported (a job's start); in the finals
# image (no .git) the baked commit
COMMIT_AT_START = _git("rev-parse", "HEAD") or _baked_commit()
DIRTY_AT_START = bool(_git("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)ledger"))   # code only


def model_commit(model_path):
    name = Path(model_path).name
    src = REPO / "configs" / "model-commits.txt"
    for line in src.read_text().splitlines():
        if line.startswith(name + " "):
            return line.split()[-1]
    return None


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def provenance(model_path=None, config=None, gpu=None):
    import torch
    import transformers
    return dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node(),
                code_commit=COMMIT_AT_START, code_dirty=DIRTY_AT_START,
                code_commit_at_write=_git("rev-parse", "HEAD"),
                model=str(model_path) if model_path else None,
                model_commit=model_commit(model_path) if model_path else None,
                torch=torch.__version__, hip=torch.version.hip, transformers=transformers.__version__,
                python=sys.version.split()[0], command=" ".join([sys.executable] + sys.argv),
                env={k: os.environ.get(k) for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")},
                gpu=gpu, seeds="none (deterministic greedy/eval; NIAH generator seed 20260930)",
                config=config)


def write_json(path, obj):
    """Write obj as JSON so that path never holds a partial or empty file, even after a crash or a hard hang (a
    0-byte result file appeared after one): the bytes go to a temporary file next to it and are fsynced, read back
    from the disk and checked (_reads_back; host RAM flips bits, 2026-10-03: one retry, then IntegrityError and
    nothing written), then the file is renamed over path and the directory fsynced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")    # concurrent writers never share a temporary file
    for attempt in (1, 2):
        with open(tmp, "wb") as f:
            f.write(json.dumps(obj).encode())
            f.flush()
            os.fsync(f.fileno())
        if _reads_back(tmp, obj):
            break
        print(f"write_json: {tmp} read back from disk differs from the record (attempt {attempt})", flush=True)
    else:
        tmp.unlink()
        raise integrity.IntegrityError(f"{path}: the written JSON read back different twice; {path} left as it was")
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _reads_back(path, obj):
    """True if the file, read from the disk around the page cache, parses as JSON that serialises as obj does (a
    second serialisation of obj, so a flip in the first one's buffer is caught too)."""
    try:
        return json.dumps(json.loads(integrity.read_direct(path))) == json.dumps(obj)
    except ValueError:                   # not JSON (or not UTF-8) any more
        return False
