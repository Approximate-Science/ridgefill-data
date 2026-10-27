#!/usr/bin/env python3
"""Which paper-table cells the declared finals cells can fill: the alignment instrument between finals/cells/*.json
(finals/make_cells.py) and finals/tables.py.

It builds a synthetic results tree whose sample records carry exactly what the declared cells run: per cell, per
harness config (finals/lib/kvabackend.harness_configs, the configs the FINAL- records pin), the config's arms as
finals/convert.py names them (protocol kind, split key), on the config's rows (RULER tasks x lengths, LongBench rows,
the locked documents under the cell's cap, T1.7's documents, the two-key rows), merged per unit and with the cell's
declared skips, as convert.py writes them. Values are seeded noise around fixed arm offsets (non-degenerate, so every
statistic the layout asks for can be computed); nothing here is evidence. Projector files are header-only stand-ins
re-pinned in the tree's own ledger (pins are a separate check: make_cells.py, register.sh). The Flash-Next cell and the
pre-final ledger records come from the mock tree (finals/mock/tables_tree.py): they are not reference cells. The speed
cell has no harness path (TODO-HARNESS-VLLM-TTFT), so its TTFT cells are listed too.
Then it runs finals/tables.py's compute() on that tree and lists every required table cell or macro without data,
grouped by the reason tables.py gives (cells declared not run print "--" and are not required: DEV-8, and protocol
1.1.3's cuts: DEV-11). Since the coordinator's decisions of 2026-10-04 night the list is empty. Protocol 1.1.3's one
declared double fill (tables.SUPERSESSION: the nll-causal cells re-run the nll cells' rows / ktail / vtail arms on the
same documents) is read from the same constant: the audit checks that every model whose superseded cell runs those arms
declares the superseding cell with them and its companions, and plants a leak on the superseded values (LEAK) so the
superseded-value macros are not zero by construction. finals/align-gaps.json is that list as of the last
reviewed change;
tests/test_finals_align.py fails when the cells or the tables drift from it in either direction (a gap closed or a new
one), so every change of the gap list is a reviewed edit of that file.
Usage: align_audit.py [--write]    (--write: refresh finals/align-gaps.json)
Exit: 0 the list equals finals/align-gaps.json (or was written); 1 it differs (printed).
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FINALS = Path(__file__).resolve().parent
REPO = FINALS.parent
GAPS = FINALS / "align-gaps.json"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FINALS / "lib"))
sys.path.insert(0, str(FINALS))
os.environ.setdefault("KVA_REPO", str(REPO))
import atomic                                                # noqa: E402
import kvabackend as KB                                      # noqa: E402
from registration import pins_text                           # noqa: E402

ROWS = 10                                                    # rows per RULER file (enough pairs that no bootstrap
                                                             # resample of a ratio's denominator is exactly 0)
LB_ROWS = 3                                                  # rows per LongBench task
NEEDLE_TASKS = ("niah_single_1", "niah_multikey_1", "niah_multivalue", "niah_multiquery")
OFFSET = {"exact": 0.0, "skip": 0.30, "noproj": 0.14, "pop-native": 0.12, "kva": 0.07, "noproj+qattn10": 0.06,
          "kva+qattn10": 0.03, "noproj+random": 0.13, "kva+random": 0.065, "noproj+tailmatch": 0.12,
          "kva+tailmatch": 0.06, "skip+tailbudget": 0.25, "kva+kexact": 0.02, "kva+vexact": 0.06, "kvexact": 0.0,
          "kva+ktail": 0.04, "kva+vtail": 0.065, "swiftkv": 0.04, "swiftkv-native": 0.09}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TABLES = _load("finals_tables_audit", FINALS / "tables.py")
SUPERSESSION = TABLES.SUPERSESSION
LEAK = -0.01                                                 # added to a superseded arm's offset in the superseded cell


def supersession_problems(cells):
    """[problem] where the declared cells do not realise SUPERSESSION: a model whose superseded cell runs a superseded
    arm at the split needs one superseding cell that runs every superseded arm its superseded cell runs, with the
    companions (tables.py refuses any other double fill)."""
    sp, out = SUPERSESSION, []
    kinds = lambda c, split: {KB.PAPER.get(a, a) for r in c.get("runs", []) if r.get("split") == split  # noqa: E731
                              for a in r["arms"]}
    for model in sorted({c["model"]["name"] for c in cells}):
        of = [c for c in cells if c["model"]["name"] == model and c["kind"] == sp["of"]]
        by = [c for c in cells if c["model"]["name"] == model and c["kind"] == sp["by"]]
        need = set().union(*(kinds(c, sp["split"]) for c in of)) & set(sp["arms"])
        if not need:
            continue
        if len(by) != 1:
            out.append(f"{model}: {len(by)} {sp['by']} cells for the superseded {sorted(need)}")
            continue
        got = kinds(by[0], sp["split"]) | {KB.PAPER.get(a, a) for r in by[0]["runs"] for a in r["arms"]}
        missing = sorted(need - got) + sorted(set(sp["with_"]) - got)
        if missing:
            out.append(f"{model}: {by[0]['name']} lacks {missing} (declared supersession)")
    return out


def _rng(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def manifest(dataset=None):
    """(path, domain) rows of the locked set's manifest, or of another frozen set's (a cell's 'dataset')."""
    with (REPO / "datasets" / (dataset or "test-set-locked") / "MANIFEST.tsv").open() as f:
        return [(r["path"], r["domain/task"].split("/")[0]) for r in csv.DictReader(f, delimiter="\t")]


def units(cell, cfg, set_):
    """[(sample set, unit id, fields)] of one harness config's rows (names and nominal lengths only; no text)."""
    from kva import locked as L
    if set_ in ("ruler", "multiturn"):
        tasks = ["niah_multikey_1"] if set_ == "multiturn" else cfg.get("ruler_tasks") or list(L.RULER_TASKS)
        lengths = [n for n in cfg.get("ruler_lengths") or list(L.RULER_LENGTHS) if n <= cfg.get("ruler_max_len", n)]
        out_set = "multiturn" if set_ == "multiturn" else "ruler-qfirst" if cfg.get("ruler_variant") else "ruler"
        out = []
        for t in tasks:
            for n in lengths:
                for i in range(ROWS):
                    f = dict(task=t, length=n, tokens=n)
                    if t in NEEDLE_TASKS and set_ == "ruler":
                        f["needle_filled"] = i != ROWS - 1
                    out.append((out_set, f"ruler/{t}_{n}.jsonl#{i:02d}", f))
        return out
    if set_ == "longbench":
        return [("longbench", f"longbench/{t}.jsonl#{i:02d}", dict(task=t, length=10000))
                for t in L.LONGBENCH_PROMPT for i in range(LB_ROWS)]
    docs = [(p, d) for p, d in manifest(cell.get("dataset")) if p.startswith("long-prompts/") and p.endswith(".txt")]
    if set_ == "decode":
        docs = [(p, d) for p, d in docs if f"lp-{p.rsplit('/', 1)[-1][:-4]}" in cfg["only"].split(",")]
    out = []
    for p, d in docs:
        n = int(p.rsplit("/", 1)[-1][:-4].rsplit("_", 2)[-2])
        if n <= cfg.get("longppl_max_len", n):
            out.append((set_, p, dict(doc=p.rsplit("/", 1)[-1][:-4], domain=d, tokens=n, length=n)))
    return out


def metrics(set_, sid, kind, split, cell, leak=False):
    """Seeded values around the arm's offset; units share their noise across arms (the pairing). leak: a superseded
    arm's value in the superseded cell (SUPERSESSION), offset by LEAK."""
    u, a = _rng(cell, set_, sid), _rng(cell, set_, sid, kind, split)
    d = OFFSET.get(kind, 0.1) * (1.0 + 0.2 * a.normal()) + (LEAK if leak else 0.0)
    if set_ in ("ruler", "ruler-qfirst", "multiturn"):
        v = float(0.8 * u.random() + 0.2 * a.random() < 0.92 - 2.5 * d)
        return {"turn1": v, "turn2": v} if set_ == "multiturn" else {"score": v}
    if set_ == "longbench":
        return {"score": float(np.clip(0.5 + 0.2 * u.normal() - d, 0, 1))}
    if set_ == "longppl":
        base = 2.0 + 0.3 * u.normal()
        return dict(novel_nll=base + d, key_nll=base + 1.5 + 2 * d, n_key=30, n_novel=400, all_nll=base + d)
    if kind == "exact":
        return {}
    return dict(top1_agree=float(np.clip(0.95 - d + 0.02 * a.normal(), 0, 1)), kl_mean=abs(d / 2 + 0.01 * a.normal()))


def cell_records(cell, p, CV):
    records = []
    for cfg, set_ in KB.harness_configs(cell):
        keys = CV.arm_keys(cfg, cell, p)
        swift = cfg["model"].rsplit("/", 1)[-1] != cell["model"]["name"]
        for out_set, sid, fields in units(cell, cfg, set_):
            arms = {}
            for name, (kind, split) in keys.items():
                if not (swift and kind == "exact"):
                    leak = (cell["kind"] == SUPERSESSION["of"] and kind in SUPERSESSION["arms"]
                            and split == SUPERSESSION["split"])
                    arms.setdefault(kind, {})[split] = metrics(out_set, sid, kind, split, cell["name"], leak)
            records.append(dict(set=out_set, sample=sid, protocol=p["version"], **CV.record_tier(cell, set_),
                                cell=cell["name"], configs=[cfg["name"]], **fields, arms=arms))
    return CV.merge(records) + CV.skipped_records(cell, [], p["version"])


def build(dest, cells=None):
    """The audit tree under dest for the declared cells (default: finals/make_cells.py's); returns tables.py's args."""
    from kva import protocol as PR
    mc = _load("finals_make_cells_audit", FINALS / "make_cells.py")
    mock = _load("finals_mock_tables_tree_audit", FINALS / "mock" / "tables_tree.py")
    CV = _load("finals_convert_audit", FINALS / "convert.py")
    p = PR.load()
    mock.build(dest, {"drop_cells": [c["name"] for c in mock.cells() if c["kind"] != "flashnext"]})
    ledger = dest / "ledger" / "paper" / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    for cell in (copy.deepcopy(c) for c in (cells if cells is not None else mc.cells()) if c["kind"] != "speed"):
        prov = cell.get("projector", {}).get("provenance", {})
        for f in cell.get("projector", {}).get("files", []):
            path = dest / "proj" / f["file"]
            path.write_bytes(mock._safetensors_header(dict(model=cell["model"]["name"], lam=0.01, split=f["layer"],
                                                          rows=prov["calib_rows"], docs=prov["calib_docs"],
                                                          note="align_audit stand-in header")))
            f["sha256"] = atomic.sha256_file(path)
        d = dest / "finals-results" / cell["name"]
        atomic.write_result(d / "cell.json", cell)
        atomic.write_result(d / "host.json", {"fingerprint": "audit", "host": {"host": "audit", "gpu_name": "NVIDIA AUDIT-GPU"}})
        atomic.write_result(d / "canary.json", {"arms": {"exact": {"logits_sha256": "0" * 64}}})
        for r in cell_records(cell, p, CV):
            atomic.write_result(d / "samples" / r["set"] / f"{CV.safe(r['sample'])}.json", r)
        tree = atomic.treehash(d)
        atomic.write_bytes(d / ".pushed", json.dumps({"ssh": {"tree": tree}, "hf": {"tree": tree}}).encode())
        atomic.write_bytes(d / ".done", tree.encode())
        pin = pins_text({"cell": {cell["name"]: atomic.sha256_file(d / "cell.json")}, "config": {}})
        lines.append(json.dumps({"type": "hypothesis", "id": cell["registration"], "claim": "ALIGN AUDIT | " + pin}))
    ledger.write_text("\n".join(lines) + "\n")
    shutil.copytree(REPO / "paper" / "tex", dest / "paper" / "tex", ignore=shutil.ignore_patterns("main.pdf", "__pycache__"))
    return argparse.Namespace(results=str(dest / "finals-results"), repo=str(dest), ledger=str(ledger),
                              protocol=str(REPO / "protocol.json"), out=str(dest / "paper" / "tex"),
                              proj_dir=str(dest / "proj"), levers=None, battery_v10=None, timing=None, pending=[])


def gaps(cells=None):
    """{reason: [where, ...]} of every required table cell or macro the declared cells leave without data."""
    mc = _load("finals_make_cells_audit", FINALS / "make_cells.py")
    declared = cells if cells is not None else mc.cells()
    with tempfile.TemporaryDirectory(prefix="align-audit-") as tmp:
        run, _, _ = TABLES.compute(build(Path(tmp), cells))
    out = {}
    for where, why in run.missing:
        out.setdefault(why, []).append(where)
    for c in run.cellslog:                           # a pending cell (tables.py --pending) is never a filled cell
        if c.get("status") == "pending":
            out.setdefault("pending (not computed: tables.py --pending)", []).append(c["cell"])
    for p in supersession_problems(declared):
        out.setdefault("the declared cells do not realise the declared supersession (tables.SUPERSESSION)", []).append(p)
    return {k: sorted(v) for k, v in sorted(out.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    got = gaps()
    n = sum(len(v) for v in got.values())
    for why, wheres in got.items():
        print(f"[{len(wheres)}] {why}")
        for w in wheres:
            print(f"      {w}")
    print(f"align_audit: {n} table cell(s) / macro(s) the declared cells cannot fill, {len(got)} reason(s)")
    if args.write:
        GAPS.write_text(json.dumps(got, indent=1) + "\n")
        print(f"wrote {GAPS}")
        return 0
    want = json.loads(GAPS.read_text()) if GAPS.is_file() else None
    if got != want:
        print(f"align_audit: DIFFERS from {GAPS} (review, then --write)", file=sys.stderr)
        return 1
    print(f"align_audit: equals {GAPS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
