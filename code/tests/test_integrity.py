"""Host-RAM hardening (2026-10-03: non-ECC RAM flips bits in page-cache pages and heap buffers):
- model.load writes <model>/SHA256SUMS once from O_DIRECT reads and checks every later load against it;
- a byte flipped in a model file on disk is refused, naming the file;
- a file whose cached bytes are corrupt but whose disk copy is intact is dropped from the cache and the model loaded
  once more; if the cache stays corrupt, the load fails;
- a truncated manifest is refused, not read as a shorter one;
- Projector.load checks the sha256 recorded at write time (else the disk copy), builds the maps from the checked
  bytes, and falls back to the disk copy when the cached read is corrupt;
- recorded_sha256 finds the sha256 in fit reports and per-file records, and trusts none when records disagree;
- prov.write_json never leaves a partial or empty file: a crash before the rename (also a hard exit) leaves the
  old file, and a read-back that differs is retried once, then refused."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kva import integrity, prov
from kva import model as M
from kva.kva import Projector

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def llama_copy(tiny_llama, tmp_path):
    """A private copy of the tiny Llama (its files get flipped)."""
    path = tmp_path / "llama"
    shutil.copytree(tiny_llama[0], path, ignore=shutil.ignore_patterns(integrity.MANIFEST))
    return path


def _flip_last_byte(path):
    """Flip one bit in the file's last byte: tensor data, so the header still parses."""
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x10
    path.write_bytes(bytes(data))


def test_model_load_writes_the_manifest_once_then_verifies(llama_copy):
    M.load(llama_copy, dtype=torch.float32)
    sums = dict(line.split()[::-1] for line in (llama_copy / integrity.MANIFEST).read_text().splitlines())
    assert set(sums) == {"config.json", "generation_config.json", "model.safetensors"}
    assert sums["model.safetensors"] == hashlib.sha256((llama_copy / "model.safetensors").read_bytes()).hexdigest()
    first = (llama_copy / integrity.MANIFEST).stat().st_mtime_ns
    M.load(llama_copy, dtype=torch.float32)
    assert (llama_copy / integrity.MANIFEST).stat().st_mtime_ns == first


def test_model_load_refuses_a_byte_flipped_on_disk(llama_copy):
    M.load(llama_copy, dtype=torch.float32)
    _flip_last_byte(llama_copy / "model.safetensors")
    with pytest.raises(integrity.IntegrityError, match=r"model\.safetensors: sha256 on disk"):
        M.load(llama_copy, dtype=torch.float32)


def test_model_load_refuses_a_truncated_manifest(llama_copy):
    M.load(llama_copy, dtype=torch.float32)
    lines = (llama_copy / integrity.MANIFEST).read_text().splitlines()
    (llama_copy / integrity.MANIFEST).write_text(lines[0] + "\n")
    with pytest.raises(integrity.IntegrityError, match="does not list exactly"):
        M.load(llama_copy, dtype=torch.float32)


def _corrupt_cached_reads(monkeypatch, name, times):
    """Make the first `times` page-cache reads of file `name` hash wrong (a corrupted cached page); disk reads stay
    intact. Returns the list of files drop_cache was called on."""
    real, left, dropped = integrity.file_sha256, [times], []

    def fake(path, direct=False):
        if not direct and Path(path).name == name and left[0]:
            left[0] -= 1
            return "0" * 64
        return real(path, direct)
    monkeypatch.setattr(integrity, "file_sha256", fake)
    monkeypatch.setattr(integrity, "drop_cache", lambda path: dropped.append(Path(path).name))
    return dropped


def test_model_load_reloads_after_ram_corruption(llama_copy, monkeypatch, capsys):
    M.load(llama_copy, dtype=torch.float32)                  # writes the manifest
    dropped = _corrupt_cached_reads(monkeypatch, "model.safetensors", 1)
    m = M.load(llama_copy, dtype=torch.float32)
    assert dropped == ["model.safetensors"] and m.n_layers == 6
    assert "corrupted in RAM, intact on disk" in capsys.readouterr().out


def test_model_load_fails_if_the_cache_stays_corrupt(llama_copy, monkeypatch):
    M.load(llama_copy, dtype=torch.float32)
    _corrupt_cached_reads(monkeypatch, "model.safetensors", 2)
    with pytest.raises(integrity.IntegrityError, match="still read corrupted"):
        M.load(llama_copy, dtype=torch.float32)


def _projector(path):
    d = 8
    Projector(4, {4: torch.eye(d), 5: 2 * torch.eye(d)}, {4: torch.zeros(d), 5: torch.ones(d)}).save(path, {"k": "v"})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_projector_load_checks_the_recorded_sha256(tmp_path):
    path = tmp_path / "p.safetensors"
    sha = _projector(path)
    p = Projector.load(path, "cpu", sha256=sha)
    assert p.sha256 == sha and p.split == 4 and p.meta["k"] == "v"
    assert torch.equal(p.W[5], 2 * torch.eye(8).bfloat16()) and torch.equal(p.b[5], torch.ones(8).bfloat16())
    assert Projector.load(path, "cpu").sha256 == sha          # no record: the disk copy is the reference
    with pytest.raises(integrity.IntegrityError, match="changed after it was recorded"):
        Projector.load(path, "cpu", sha256="f" * 64)


def test_projector_load_falls_back_to_the_disk_copy_on_a_corrupt_cached_read(tmp_path, monkeypatch, capsys):
    path = tmp_path / "p.safetensors"
    sha = _projector(path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x10
    monkeypatch.setattr(Path, "read_bytes", lambda self: bytes(data))   # the cached read returns a flipped byte
    p = Projector.load(path, "cpu", sha256=sha)
    assert p.sha256 == sha and torch.equal(p.b[5], torch.ones(8).bfloat16())
    assert "MEMORY" in capsys.readouterr().out


def test_recorded_sha256_reads_fit_reports_and_file_records(tmp_path):
    (tmp_path / "fit-a.json").write_text(json.dumps(dict(projector="/old/place/proj/a.safetensors",
                                                         projector_sha256="a" * 64)))
    (tmp_path / "lowrank-b.json").write_text(json.dumps(dict(files={"64": dict(path="x/b.safetensors",
                                                                               sha256="b" * 64)})))
    assert integrity.recorded_sha256("data/proj/a.safetensors", tmp_path) == "a" * 64
    assert integrity.recorded_sha256("b.safetensors", tmp_path) == "b" * 64
    assert integrity.recorded_sha256("c.safetensors", tmp_path) is None
    (tmp_path / "screen.json").write_text(json.dumps([dict(projector="a.safetensors", projector_sha256="c" * 64)]))
    assert integrity.recorded_sha256("a.safetensors", tmp_path) is None


def test_read_direct_returns_the_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(256)) * 70001)              # not a multiple of the read size
    assert integrity.read_direct(path) == path.read_bytes()
    assert integrity.file_sha256(path, direct=True) == integrity.file_sha256(path)


REC = {"runs": {"exact": {"logprobs": [-0.5, -1.25, float("nan")]}}, "n": 3, "rows": [(0, 4)]}


def test_write_json_writes_the_record(tmp_path):
    path = tmp_path / "r" / "p.json"
    prov.write_json(path, REC)
    assert path.read_text() == json.dumps(REC)
    assert [p.name for p in path.parent.iterdir()] == ["p.json"]


def test_write_json_crash_before_rename_keeps_the_old_file(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    prov.write_json(path, {"old": 1})

    def crash(self, target):
        raise KeyboardInterrupt("crash between write and rename")
    monkeypatch.setattr(Path, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        prov.write_json(path, REC)
    assert json.loads(path.read_text()) == {"old": 1}


def test_write_json_hard_exit_before_rename_leaves_no_partial_file(tmp_path):
    """A process killed between write and rename (no handlers run) leaves the target absent, never empty."""
    path = tmp_path / "p.json"
    code = (f"import os, sys; sys.path.insert(0, {str(ROOT)!r}); from pathlib import Path; from kva import prov; "
            f"Path.replace = lambda self, target: os._exit(9); prov.write_json(Path({str(path)!r}), {{'a': 1}})")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 9
    assert not path.exists()
    assert [json.loads(p.read_text()) for p in tmp_path.glob("p.json.*.tmp")] == [{"a": 1}]


def test_write_json_retries_a_bad_read_back_once(tmp_path, monkeypatch, capsys):
    path = tmp_path / "p.json"
    real, calls = integrity.read_direct, []

    def flaky(p):
        calls.append(p)
        data = real(p)
        return data[:-2] + b"9}" if len(calls) == 1 else data
    monkeypatch.setattr(integrity, "read_direct", flaky)
    prov.write_json(path, {"a": 1})
    assert len(calls) == 2 and json.loads(path.read_text()) == {"a": 1}
    assert "differs from the record (attempt 1)" in capsys.readouterr().out


def test_write_json_refuses_after_two_bad_read_backs(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    prov.write_json(path, {"old": 1})
    monkeypatch.setattr(integrity, "read_direct", lambda p: b"")      # e.g. an empty file on disk
    with pytest.raises(integrity.IntegrityError, match="read back different twice"):
        prov.write_json(path, {"new": 2})
    assert json.loads(path.read_text()) == {"old": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]
