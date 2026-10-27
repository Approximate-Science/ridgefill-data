"""The clean refit (protocol 1.1, T0.2's calibration exclusion list): scripts/fit_sweep.py records the projector's
calibration provenance (rows, calib_exclude_sha256, code commit) in its metadata, and refuses by name an existing
projector fitted on another calibration list instead of skipping it as done or overwriting it; the refit job file
names the splits the finals cells use; the stale/clean row counts of the real projectors (when present locally)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kva import data
from kva.kva import Projector

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"


def _fit_sweep():
    spec = importlib.util.spec_from_file_location("fit_sweep", ROOT / "scripts" / "fit_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _proj(path, rows):
    Projector(2, {i: torch.eye(4) for i in (2, 3)}, {i: torch.zeros(4) for i in (2, 3)}).save(path, dict(rows=rows))
    return path


def test_meta_of_reads_the_header_only(tmp_path):
    p = _proj(tmp_path / "m-sweep-s2.safetensors", 1234)
    assert Projector.meta_of(p) == dict(rows="1234", split="2")


def test_stale_projectors_are_the_existing_files_of_another_list(tmp_path):
    fs = _fit_sweep()
    clean, stale = _proj(tmp_path / "a.safetensors", 100), _proj(tmp_path / "b.safetensors", 140)
    missing = tmp_path / "c.safetensors"
    assert fs.stale_projectors([clean, stale, missing], 100) == [(stale, "140")]
    assert fs.stale_projectors([clean, missing], 100) == []


def test_fit_sweep_refuses_a_stale_projector_before_the_capture(tmp_path):
    """End to end on make_tiny.py's Llama (CPU dry run, one calibration document): a fit, a rerun that skips the done
    split, and a rerun on another list (two documents) that refuses the existing projector by name."""
    if not (MODELS / "llama-3.1-8b-instruct" / "tokenizer.json").exists():
        pytest.skip("needs the local Llama-3.1 tokenizer")
    env = dict(os.environ, KVA_CPU_DRYRUN="1", KVA_WORK=str(tmp_path / "work"), KVA_BULK=str(tmp_path / "bulk"),
               KVA_STATS_DIR=str(tmp_path / "sums"), HF_HUB_OFFLINE="1", HIP_VISIBLE_DEVICES="-1",
               ROCR_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", KVA_LIMIT_DOCS="1")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tiny.py"), str(tmp_path / "models")], env=env,
                   check=True, capture_output=True)
    run = lambda **kw: subprocess.run([sys.executable, str(ROOT / "scripts" / "fit_sweep.py"),  # noqa: E731
                                       str(tmp_path / "models" / "llama-3.1-8b-instruct"), "28"],
                                      env=dict(env, **kw), capture_output=True, text=True)
    first = run()
    assert first.returncode == 0, first.stderr[-2000:]
    path = tmp_path / "bulk" / "proj" / "llama-3.1-8b-instruct-sweep-s28.safetensors"
    meta = Projector.meta_of(path)
    assert meta["calib_exclude_sha256"] == __import__("hashlib").sha256(data.CALIB_EXCLUDE_FILE.read_bytes()).hexdigest()
    from kva import prov                 # the same tree's commit: a hash in a checkout, '' in an export without .git
    assert int(meta["rows"]) > 0 and meta["code_commit"] == prov.COMMIT_AT_START and meta["code_dirty"] in ("True", "False")
    assert meta["docs"] == "1"                                                       # KVA_LIMIT_DOCS=1
    report = json.loads((tmp_path / "work" / "results" / "fit-sweep-llama-3.1-8b-instruct-s28.json").read_text())
    assert (report["docs"], report["rows"], report["projector_sha256"]) == (1, int(meta["rows"]), Projector.load(
        path, "cpu").sha256)
    again = run()
    assert again.returncode == 0 and "to fit []" in again.stdout
    other = run(KVA_LIMIT_DOCS="2")
    assert other.returncode != 0 and "StaleProjector" in other.stderr and path.name in other.stderr
    assert "capture" not in other.stdout and Projector.meta_of(path) == meta          # refused before the capture


def test_the_refit_job_covers_the_cells_splits():
    jobs = (ROOT / "jobs" / "paper-refit-clean.jobs").read_text().splitlines()
    fit = [x.split() for x in jobs if x.strip() and not x.startswith("#")]
    assert [(f[1], f[2], f[3]) for f in fit] == [
        ("scripts/fit_sweep.py", "data/models/llama-3.1-8b-instruct", "8,12,16,20"),
        ("scripts/fit_sweep.py", "data/models/qwen3.5-9b", "8,12,16,20")]


def test_the_clean_calibration_list_has_71_documents():
    """The count the cells declare (projector.provenance.calib_docs): 99 calibration documents minus the 28 excluded;
    tokenizer-free, from the committed inputs (the private copy has the same calibration list)."""
    if not (data.INPUTS / "plan-train.json").is_file():
        pytest.skip("needs the calibration texts (plan-train.json, not in the public code)")
    assert len(data.calib_docs("train")) == 71 and len(data.calib_exclude("train")) == 28


@pytest.mark.parametrize("model,clean", [("llama-3.1-8b-instruct", 834541), ("qwen3.5-9b", 839652),
                                         ("qwen3-8b", 845043)])
def test_the_clean_calibration_row_counts(model, clean):
    """The counts the finals cells pin (finals/make_cells.py CALIB_ROWS), recomputed from the clean lists with the
    model's tokenizer; and the stale projectors on disk, if present, carry the pre-exclusion count (the control:
    the same instrument reproduces the rows those projectors recorded)."""
    if not (MODELS / model / "tokenizer.json").exists() or data.INPUTS.name != "inputs-private":
        pytest.skip("needs the local tokenizer and the private calibration inputs")
    os.environ.pop("KVA_LIMIT_DOCS", None)
    tok = data.tokenizer(MODELS / model)
    assert sum(len(d["ids"]) for d in data.calib_ids(tok, "train")) == clean
    stale = ROOT / "data" / "proj" / f"{model}-sweep-s12.safetensors"
    if stale.exists() and "calib_exclude_sha256" not in Projector.meta_of(stale):
        bos = data.bos_ids(tok)
        every = [x for x in json.loads((data.INPUTS / "plan-train.json").read_text()) if x["kind"] == "calib"]
        assert int(Projector.meta_of(stale)["rows"]) == sum(len(data.cap(bos + data.encode(tok, x["text"]), bos))
                                                            for x in every) != clean
