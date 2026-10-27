"""Protocol 1.1: the calibration exclusion list (datasets/calibration-exclude.json) is applied by kva.data.calib_docs
for train and sterm, and a missing file is an error (never an empty list)."""
import json
from pathlib import Path

import pytest

from kva import data


def test_exclusion_file_lists_the_t02_documents():
    d = json.loads(data.CALIB_EXCLUDE_FILE.read_text())
    assert len(d["train"]) == 28 and "trainc31" in d["train"] and d["sterm"] == ["sterm4"]


def test_calib_docs_apply_the_exclusions():
    if not (data.INPUTS / "plan-train.json").exists():
        pytest.skip("calibration texts not available here")
    names = {x["name"] for x in data.calib_docs("train")}
    assert not names & data.calib_exclude("train")
    assert "sterm4" not in {x["name"] for x in data.calib_docs("sterm")}
    raw = {x["name"] for x in json.loads((data.INPUTS / "plan-train.json").read_text()) if x["kind"] == "calib"}
    assert len(raw) - len(names) == 28


def test_missing_file_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CALIB_EXCLUDE_FILE", tmp_path / "none.json")
    with pytest.raises(FileNotFoundError):
        data.calib_exclude("train")


def test_held_documents_are_excluded_by_the_sha256_of_their_name(monkeypatch, tmp_path):
    plan = [dict(name="held-a", text="a", tokens=2048), dict(name="held-b", text="b", tokens=2048)]
    (tmp_path / "plan-held.json").write_text(json.dumps(plan))
    (tmp_path / "exclude.json").write_text(json.dumps({"held": []}))
    monkeypatch.setattr(data, "INPUTS", tmp_path)
    monkeypatch.setattr(data, "CALIB_EXCLUDE_FILE", tmp_path / "exclude.json")
    monkeypatch.setattr(data, "HELD_EXCLUDE_SHA256", {data.name_sha256("held-b")})
    assert [d["name"] for d in data.calib_docs("held")] == ["held-a"]


def test_the_shipped_held_exclusion_removes_one_document():
    if not (data.INPUTS / "plan-held.json").exists():
        pytest.skip("calibration texts not available here")
    plan = json.loads((data.INPUTS / "plan-held.json").read_text())
    hit = [x["name"] for x in plan if data.name_sha256(x["name"]) in data.HELD_EXCLUDE_SHA256]
    assert len(hit) == 1 and hit[0] not in {d["name"] for d in data.calib_docs("held")}
