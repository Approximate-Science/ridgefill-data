#!/usr/bin/env python3
"""Validate a cell config against finals/cell.schema.json (fail closed).

The repo's pinned requirements have no jsonschema package, so this is a hand-rolled validator that
implements exactly the keywords the schema uses: type, enum, const, pattern, required, properties,
additionalProperties(false), items, minItems, uniqueItems, and $comment/$semantic (ignored). Any other
keyword in the schema is an error naming it (BRIEF-COMMON gate rule 2: nothing unimplemented succeeds).
Cross-field rules from the schema's "$semantic" block are checked by semantic().

CLI: cellcheck.py --schema finals/cell.schema.json CELL.json        exit 0 = valid, 2 = invalid, 3 = schema error
"""
import argparse
import json
import re
import sys
from pathlib import Path

IGNORED = {"$comment", "$schema", "title", "$semantic"}   # metadata, never a constraint
TYPES = {"object", "array", "string", "boolean", "integer", "number"}


class SchemaError(RuntimeError):
    """The schema file uses a keyword this validator does not implement: refuse to guess."""


def _check_type(schema, val, where):
    t = schema["type"]
    ok = {"object": dict, "array": list, "string": str, "boolean": bool,
          "integer": int, "number": (int, float)}
    if not isinstance(val, ok[t]) or (t in ("integer", "number") and isinstance(val, bool)):
        return [f"{where}: expected {t}, got {type(val).__name__}"]
    return []


def _check_pattern(schema, val, where):
    if not re.match(schema["pattern"], val):     # JSON Schema anchors via the pattern itself
        return [f"{where}: {val!r} does not match {schema['pattern']!r}"]
    return []


def validate(schema, val, where="$"):
    """All problems with val against schema, as a list of strings (empty = valid)."""
    for k in schema:
        if k not in IGNORED and k not in ("type", "enum", "const", "pattern", "required", "properties",
                                          "additionalProperties", "items", "minItems", "uniqueItems"):
            raise SchemaError(f"cellcheck does not implement schema keyword {k!r} (at {where})")
    errs = []
    if "type" in schema:
        if schema["type"] not in TYPES:
            raise SchemaError(f"unknown type {schema['type']!r} at {where}")
        errs += _check_type(schema, val, where)
    if "enum" in schema and val not in schema["enum"]:
        errs.append(f"{where}: {val!r} not in {schema['enum']}")
    if "const" in schema and val != schema["const"]:
        errs.append(f"{where}: must be {schema['const']!r}, got {val!r}")
    if isinstance(val, str) and "pattern" in schema:
        errs += _check_pattern(schema, val, where)
    if isinstance(val, dict):
        for k in schema.get("required", []):
            if k not in val:
                errs.append(f"{where}: missing required field {k!r}")
        props = schema.get("properties", {})
        extra = [k for k in val if k not in props]
        if schema.get("additionalProperties") is False and extra:
            errs.append(f"{where}: unknown field(s) {extra}")
        for k, v in val.items():
            if k in props:
                errs += validate(props[k], v, f"{where}.{k}")
    if isinstance(val, list):
        if "minItems" in schema and len(val) < schema["minItems"]:
            errs.append(f"{where}: needs >= {schema['minItems']} items, has {len(val)}")
        if schema.get("uniqueItems") and len(set(map(json.dumps, val))) != len(val):
            errs.append(f"{where}: duplicate items")
        if "items" in schema:
            for i, v in enumerate(val):
                errs += validate(schema["items"], v, f"{where}[{i}]")
    return errs


# ---- cross-field rules (the schema's $semantic block, kept in sync by hand)

CONTROLS = {"identity+random": "identity+rows", "identity+tailmatch": "identity+rows",
            "ridgefill+random": "ridgefill+rows", "ridgefill+tailmatch": "ridgefill+rows"}
SWIFTKV_ARMS = ("swiftkv", "swiftkv-native")
QBLIND_ARMS = ("identity+qblind", "ridgefill+qblind")


def _projector_arm(a):
    return a.startswith("ridgefill") or a == "kvexact"


FLASHNEXT_ONLY = ("arms", "sets", "declared_share", "positive_control", "split_layer")


def semantic(cell):
    """(errors, warnings) for cross-field rules; error strings name the rule."""
    errs, warns = [], []
    if cell.get("kind") == "flashnext":
        need = [k for k in FLASHNEXT_ONLY if k not in cell] + [k for k in ("ttft_min_speedup", "ttft_length")
                                                               if k not in cell.get("margins", {})]
        if need or "runs" in cell or cell.get("split") != "-":
            errs.append(f"flashnext_cell: needs {need or 'split -'} and no runs (it runs on its own driver)")
        return errs, warns
    if "runs" not in cell:
        errs.append("primary_run: a reference cell needs 'runs'")
    want_split = {"ruler-deep": "5/8", "nll-fresh": "1/2"}.get(cell.get("kind"), "3/8")   # the 2026-10-08 follow-ups
    if cell.get("split") != want_split or cell.get("tail") != 4096:
        errs.append(f"primary_run: a {cell.get('kind')} cell declares split {want_split} and tail 4096, not "
                    f"{cell.get('split')} / {cell.get('tail')}")
    extra = [k for k in FLASHNEXT_ONLY if k in cell]
    if extra:
        errs.append(f"flashnext_cell: {extra} are flashnext fields")
    for f in cell.get("projector", {}).get("files", []):
        if "sha256" not in f:
            warns.append(f"projector_unpinned: {f['file']} has no declared sha256 (the clean refit's): the cell cannot be "
                         "registered or run until make_cells.py finds its clean fit report")
    runs = cell.get("runs", [])
    ids = [r.get("id") for r in runs]
    if len(set(ids)) != len(ids):
        errs.append(f"run_ids_unique: duplicate run ids {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    if runs and not any(r.get("role") == "primary" and r.get("split") == cell.get("split") for r in runs):
        errs.append(f"primary_run: no run with role 'primary' at the primary split {cell.get('split')}")
    derived = cell.get("derived", {})
    for r in runs:
        arms, rid = r.get("arms", []), r.get("id")
        if "exact" not in arms:
            errs.append(f"exact_in_every_run: run {rid} does not list 'exact'")
        swift = [a for a in arms if a in SWIFTKV_ARMS]
        if r.get("model") == "swiftkv":
            if set(arms) - {"exact", *SWIFTKV_ARMS} or not swift:
                errs.append(f"swiftkv_runs: run {rid} on the SwiftKV model lists {arms}: only exact, swiftkv, swiftkv-native")
            if "swiftkv_model" not in cell or not str(cell.get("model", {}).get("name", "")).startswith("llama"):
                errs.append(f"swiftkv_runs: run {rid} needs 'swiftkv_model' and a llama model (OUTLINE.md T1.6)")
        elif swift:
            errs.append(f"swiftkv_runs: run {rid} lists {swift} but does not run on the SwiftKV model")
        for c, rows in CONTROLS.items():
            if c in arms and rows not in arms:
                errs.append(f"controls_need_their_rows_arm: run {rid}: {c} needs {rows} in the same run")
        if any(a in QBLIND_ARMS for a in arms) and r.get("split") != "3/8":
            errs.append(f"qblind_at_the_blind_split: run {rid} has a qblind arm at {r.get('split')}, not 3/8")
        if "nofill+tailbudget" in arms and "tailbudget" not in cell:
            errs.append(f"tailbudget_declared: run {rid} has nofill+tailbudget but the cell declares no 'tailbudget'")
        if any(_projector_arm(a) for a in arms) and "projector" not in cell:
            errs.append(f"ridgefill_requires_projector: run {rid} has a ridgefill arm but the cell has no 'projector'")
        need = {"longppl": ["keytoken"], "decode": ["decode"], "multiturn": ["secondq"]}.get(r.get("set"), [])
        for k in need:
            if k not in cell:
                errs.append(f"set_blocks: run {rid} (set {r.get('set')}) needs the cell's '{k}'")
        if r.get("set") == "multiturn" and "ruler-twokey" not in derived:
            errs.append(f"set_blocks: run {rid} (set multiturn) needs derived 'ruler-twokey'")
        if r.get("variant") == "qfirst" and (r.get("set") != "ruler" or "ruler-qfirst" not in derived):
            errs.append(f"set_blocks: run {rid} (variant qfirst) runs on set ruler and needs derived 'ruler-qfirst'")
        if ("tasks" in r and r.get("set") != "ruler") or ("lengths" in r and r.get("set") not in ("ruler", "multiturn")):
            errs.append(f"row_selection: run {rid}: tasks only on ruler runs, lengths on ruler and multiturn runs")
        cap = cell.get("max_len")
        if cap and any(L > cap for L in r.get("lengths", [])):
            errs.append(f"row_selection: run {rid} selects lengths over max_len {cap}")
    if "dataset" in cell and (cell.get("kind") not in ("nll", "nll-causal", "nll-fresh")
                              or any(r.get("set") != "longppl" for r in runs)):
        errs.append(f"dataset_longppl_only: dataset {cell['dataset']!r} holds long-document NLL prompts only: an nll "
                    f"cell whose runs are all set longppl (kind {cell.get('kind')}, sets "
                    f"{sorted({r.get('set') for r in runs})})")
    if cell.get("registration") == "FINAL-TBD":
        warns.append("final_tbd_flagged: registration 'FINAL-TBD': no FINAL- ledger record exists yet; "
                     "run_final.sh refuses to run this cell until it is registered (labbook hypothesis)")
    return errs, warns


def check_file(cell_path, schema_path, out=sys.stderr):
    """Validate one cell file; print problems and warnings; True iff valid. No output = valid and clean."""
    schema = json.loads(Path(schema_path).read_text())
    cell = json.loads(Path(cell_path).read_text())
    errs = validate(schema, cell, Path(cell_path).name)
    serrs, warns = ([], []) if errs else semantic(cell)
    for w in warns:
        print(f"WARNING {Path(cell_path).name}: {w}", file=out)
    for e in errs + serrs:
        print(f"INVALID {Path(cell_path).name}: {e}", file=out)
    return not (errs or serrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("cell")
    args = ap.parse_args()
    try:
        sys.exit(0 if check_file(args.cell, args.schema) else 2)
    except SchemaError as e:
        print(f"SCHEMA ERROR: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()