"""finals/tables.py --no-ledger on a public copy of a results tree (no ledger, no dotfiles, no projector files, no
paper/tex): it writes every table it can compute, prints \\TBDcell for the cells that need what the copy lacks and lists
them with their reasons; without the flag a missing ledger is refused by name; a tampered sample record still refuses."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FINALS = REPO / "finals"
sys.path.insert(0, str(FINALS))
sys.path.insert(0, str(FINALS / "lib"))
import tables as T                                           # noqa: E402


def _mock():
    spec = importlib.util.spec_from_file_location("finals_mock_tables_tree_public", FINALS / "mock" / "tables_tree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def public_tree(tmp_path_factory):
    """The mock tree as a public copy: no ledger, no projector files, no dotfiles (.done, .pushed)."""
    tmp = tmp_path_factory.mktemp("public")
    _mock().build(tmp)
    shutil.rmtree(tmp / "ledger")
    shutil.rmtree(tmp / "proj")
    for f in (tmp / "finals-results").rglob(".*"):
        f.unlink()
    return tmp


def run_tables(tree, out, *extra):
    return subprocess.run([sys.executable, "-I", str(FINALS / "tables.py"), str(tree / "finals-results"),
                           "--repo", str(tree), "--out", str(out), *extra], capture_output=True, text=True)


def test_no_ledger_writes_the_tables_and_lists_what_it_cannot_compute(public_tree, tmp_path):
    r = run_tables(public_tree, tmp_path, "--no-ledger")
    assert r.returncode == 0, r.stderr
    assert sorted(p.stem for p in (tmp_path / "tables").glob("*.tex")) == sorted(T.TEMPLATES)
    cells = json.loads((tmp_path / "tables" / "cells.json").read_text())
    assert cells["ledger"] == "no-ledger"
    reasons = {m["reason"] for m in cells["missing"]}
    assert any("no ledger record" in x for x in reasons) and any("--proj-dir" in x for x in reasons)
    assert not [m for m in cells["missing"] if m["cell"].startswith("battery ")]          # the results tables are whole
    assert [c for c in cells["cells"] if c["cell"].startswith("battery ")]
    assert (tmp_path / "numbers.tex").read_text().startswith("\\newcommand{")
    assert "cell(s) not computed" in r.stdout


def test_a_missing_ledger_without_the_flag_is_refused_by_name(public_tree, tmp_path):
    r = run_tables(public_tree, tmp_path)
    assert r.returncode == 2 and "ledger" in r.stderr and "--no-ledger" in r.stderr


def test_no_ledger_still_refuses_a_tampered_sample_record(public_tree, tmp_path):
    tree = tmp_path / "tree"
    shutil.copytree(public_tree / "finals-results", tree / "finals-results")
    f = sorted((tree / "finals-results").rglob("samples/*/*.json"))[0]
    f.write_text(f.read_text() + " ")
    r = run_tables(tree, tmp_path / "out", "--no-ledger")
    assert r.returncode == 1 and "VERIFY FAILED" in r.stderr and f.name in r.stderr
