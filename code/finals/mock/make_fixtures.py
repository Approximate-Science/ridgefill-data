#!/usr/bin/env python3
"""Build the mock pipeline fixtures: a small fake locked set, mock cells valid against
finals/cell.schema.json, and a mock ledger with FINAL- records for them.

The fake locked set mirrors the real one's shape exactly (FROZEN.json with manifest_sha256, a
MANIFEST.tsv with the same columns, per-file sha256 rows) so the runner's manifest checks run for
real. The texts are fake: nothing here is or derives from a real locked-set text.

CLI: make_fixtures.py DEST [FRESH]   (writes DEST/{FROZEN.json,MANIFEST.tsv,<files>,cells/,ledger.jsonl}; with
FRESH, also a fake fresh long-document set there (name it <dir of DEST>/fresh-long-20261008: the runner looks for a
cell's 'dataset' beside the locked set) and the mock cell mock-qwen-nll-fresh that reads it)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

PROVENANCE = dict(calib_rows=834541, calib_docs=71, fitted_after="cba2d91", calib_exclude_sha256="0" * 64)
MARGINS = {"ppl_ratio_upper_pct": 1.0, "ruler_points_lower": -2.0}
CELLS = [
    {
        "name": "mock-llama-ruler", "kind": "ruler",
        "model": {"id": "mock-org/Llama-3.1-8B-Instruct", "name": "llama-3.1-8b-instruct", "commit": "0" * 40},
        "swiftkv_model": {"id": "mock-org/Llama-3.1-SwiftKV-8B-Instruct",
                          "name": "llama-3.1-swiftkv-8b-instruct", "commit": "1" * 40},
        "tail": 4096, "split": "3/8", "margins": MARGINS,
        "runs": [{"id": "ruler-3-8", "set": "ruler", "split": "3/8", "role": "primary",
                  "arms": ["exact", "nofill", "identity", "pop-native", "ridgefill", "identity+rows",
                           "ridgefill+rows", "identity+random"]},
                 {"id": "ruler-swiftkv", "set": "ruler", "split": "1/2", "role": "swiftkv", "model": "swiftkv",
                  "arms": ["exact", "swiftkv-native", "swiftkv"]}],
        "caps": {"ruler_max_len": 65536},
        "projector": {"provenance": PROVENANCE, "files": [
            {"layer": 12, "file": "llama-3.1-8b-instruct-sweep-s12.safetensors", "sha256": "1" * 64}]},
        "deterministic": True, "registration": "FINAL-MOCK-LLAMA-RULER", "code_tag": "protocol-v1.1",
        "note": "mock pipeline cell: T1.1 shape, fake texts",
    },
    {
        "name": "mock-qwen-nll", "kind": "nll",
        "model": {"id": "mock-org/Qwen3.5-9B", "name": "qwen3.5-9b", "commit": "2" * 40},
        "tail": 4096, "split": "3/8", "margins": MARGINS,
        "runs": [{"id": "nll-3-8", "set": "longppl", "split": "3/8", "role": "primary",
                  "arms": ["exact", "identity", "ridgefill", "ridgefill+kexact"]}],
        "keytoken": {"tail": 4096, "threshold": 2.0},
        "projector": {"provenance": dict(PROVENANCE, calib_rows=839652), "files": [
            {"layer": 12, "file": "qwen3.5-9b-sweep-s12.safetensors", "sha256": "2" * 64}]},
        "deterministic": True, "registration": "FINAL-MOCK-QWEN-NLL", "code_tag": "protocol-v1.1",
        "note": "mock pipeline cell: T1.3 shape, fake texts",
    },
]

# a longppl cell on the fresh set (cell field 'dataset'); written only when make_fixtures is given a FRESH dir
FRESH_CELL = {
    "name": "mock-qwen-nll-fresh", "kind": "nll", "dataset": "fresh-long-20261008",
    "model": {"id": "mock-org/Qwen3.5-9B", "name": "qwen3.5-9b", "commit": "2" * 40},
    "tail": 4096, "split": "3/8", "margins": MARGINS,
    "runs": [{"id": "nll-3-8", "set": "longppl", "split": "3/8", "role": "primary",
              "arms": ["exact", "identity", "ridgefill"]}],
    "keytoken": {"tail": 4096, "threshold": 2.0},
    "projector": {"provenance": dict(PROVENANCE, calib_rows=839652), "files": [
        {"layer": 12, "file": "qwen3.5-9b-sweep-s12.safetensors", "sha256": "2" * 64}]},
    "deterministic": True, "registration": "FINAL-MOCK-QWEN-NLL-FRESH", "code_tag": "protocol-v1.1",
    "note": "mock pipeline cell: longppl on the fresh set (cell field dataset), fake texts",
}
FRESH_FILES = [("long-prompts/chat_8192_0.txt", 1), ("long-prompts/legal_16384_0.txt", 1)]

# (path, rows) of the fake locked set; each row is one tiny fake jsonl record
FILES = [
    ("ruler/niah_single_1_8192.jsonl", 3),
    ("ruler/qa_1_16384.jsonl", 3),
    ("longbench/narrativeqa.jsonl", 3),
    ("longbench/lcc.jsonl", 3),
    ("long-prompts/prose_16384_0.txt", 1),
    ("long-prompts/papers_16384_1.txt", 1),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fake_set(dest, files):
    """A fake frozen set at dest (MANIFEST.tsv + FROZEN.json + files): its manifest sha256."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel, n in files:
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        body = ("\n".join(json.dumps({"row": i, "path": rel}) for i in range(n)) + "\n"
                if rel.endswith(".jsonl") else f"fake long prompt text {rel}\n")
        p.write_text(body)
        rows.append(f"{rel}\t{sha(body.encode())}\t{len(body.encode())}\tfake/domain\tfake-source\tfake-licence\t{1000 * (n + 1)}\t{1000 * n}\t{999 * n}\n")
    header = "path\tsha256\tbytes\tdomain/task\tsource_url\tlicence\tllama-3.1-8b\tqwen3.5-9b\tqwen3.8-flash-next\n"
    manifest = (header + "".join(rows)).encode()
    man_sha = sha(manifest)
    (dest / "MANIFEST.tsv").write_bytes(manifest)
    (dest / "FROZEN.json").write_text(json.dumps(
        {"manifest_sha256": man_sha, "final_frozen_time": "2026-10-04T00:00:00Z",
         "amendments": "mock fixtures", "dev_overlap_hits": {}}, indent=1) + "\n")
    return man_sha


def build(dest, fresh=None):
    dest = Path(dest)
    man_sha = fake_set(dest, FILES)
    cells_ = [dict(c, dataset_manifest_sha256=man_sha) for c in CELLS]
    if fresh is not None:
        cells_.append(dict(FRESH_CELL, dataset_manifest_sha256=fake_set(fresh, FRESH_FILES)))
    cells = dest / "cells"
    cells.mkdir(exist_ok=True)
    for c in cells_:
        (cells / f"{c['name']}.json").write_text(json.dumps(c, indent=1) + "\n")
    # the mock ledger: FINAL- hypothesis records whose claim pins the cell's and its configs' sha256, as a real
    # registration does (finals/lib/registration.py; KVA_REPO = this repo, whose protocol.json the configs use)
    os.environ.setdefault("KVA_REPO", str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    import registration
    with (dest / "ledger.jsonl").open("w") as f:
        for c in cells_:
            rec = {"type": "hypothesis", "id": c["registration"],
                   "claim": f"mock registration for {c['name']} | " + registration.pins_text(registration.pins(c)),
                   "predict": "mock", "kill": "mock", "utc": "2026-10-04T00:00:00+00:00"}
            f.write(json.dumps(rec) + "\n")
    return man_sha


if __name__ == "__main__":
    print("manifest_sha256", build(sys.argv[1] if len(sys.argv) > 1 else "mock-fixtures",
                                   sys.argv[2] if len(sys.argv) > 2 else None))