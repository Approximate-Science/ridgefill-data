#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the per-document loss table from the reference cells' per-document records.

One row per (cell, document, arm, depth). Deltas are arm minus the same cell's exact run (SwiftKV arms: minus
the SwiftKV model's own exact run, stored as `exact_swiftkv`). `set` is `fresh` for the 30 fresh documents
(cell `*-nll-fresh`) and `locked` for the frozen evaluation set; 25 fresh document names repeat locked names,
so (model, set, doc) is the key.

Usage: python3 tools/make_per_document_csv.py results/reference derived/per-document-nll.csv
"""
import csv
import json
import sys
from pathlib import Path

FIELDS = ["model", "cell", "set", "protocol", "doc", "domain", "length", "tokens", "arm", "depth",
          "novel_nll", "key_nll", "all_nll", "n_novel", "n_key",
          "exact_novel_nll", "exact_key_nll", "delta_novel_nll", "delta_key_nll"]


def minus(a, b):
    return None if a is None or b is None else a - b


def rows(root):
    paths = sorted(root.glob("*-nll*/samples/longppl/*.json"), key=lambda p: p.as_posix())
    if not paths:
        sys.exit(f"no per-document records under {root}/*-nll*/samples/longppl/; is this the results/reference tree?")
    for path in paths:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if "skipped" in rec:   # Qwen3-8B's documents above its 32K limit
            continue
        cell = path.parents[2].name
        exact = rec["arms"]["exact"]["-"]
        for arm, depths in sorted(rec["arms"].items()):
            if arm == "exact":
                continue
            ref = rec["exact_swiftkv"] if arm.startswith("swiftkv") else exact
            for depth, value in sorted(depths.items()):
                yield {"model": cell.split("-nll")[0], "cell": cell,
                       "set": "fresh" if cell.endswith("-nll-fresh") else "locked",
                       "protocol": rec.get("protocol"),
                       "doc": rec["doc"], "domain": rec["domain"], "length": rec["length"], "tokens": rec["tokens"],
                       "arm": arm, "depth": depth,
                       **{k: value.get(k) for k in ("novel_nll", "key_nll", "all_nll", "n_novel", "n_key")},
                       "exact_novel_nll": ref["novel_nll"], "exact_key_nll": ref["key_nll"],
                       "delta_novel_nll": minus(value["novel_nll"], ref["novel_nll"]),
                       "delta_key_nll": minus(value.get("key_nll"), ref["key_nll"])}


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    root, out = Path(argv[0]), Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows(root):
            writer.writerow(row)
            count += 1
    print(f"{out}: {count} rows")


if __name__ == "__main__":
    main(sys.argv[1:])
