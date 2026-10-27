#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the released pre-registration ledger extract.

1. index.tsv: every entry's `prev` equals the previous entry's `sha256` (the hash chain), seq runs 0..N-1.
2. cited.jsonl: every line hashes (sha256 of the line's UTF-8 bytes, without the newline) to the index row of
   its `seq`, and its `prev` equals that row's `prev`.
Exit 0 when both hold; 1 otherwise, naming the first failing row.

Usage: python3 tools/verify_ledger.py provenance/ledger
"""
import csv
import hashlib
import json
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    folder = Path(argv[0])
    with (folder / "index.tsv").open(encoding="utf-8", newline="") as f:
        index = list(csv.DictReader(f, delimiter="\t"))
    for i, row in enumerate(index):
        if int(row["seq"]) != i:
            sys.exit(f"FAIL index row {i}: seq {row['seq']}")
        expected_prev = index[i - 1]["sha256"] if i else ""
        if row["prev"] != expected_prev:
            sys.exit(f"FAIL index seq {i}: prev {row['prev']} != sha256 of seq {i - 1}")
    cited = 0
    for line in (folder / "cited.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        row = index[entry["seq"]]
        if hashlib.sha256(line.encode("utf-8")).hexdigest() != row["sha256"]:
            sys.exit(f"FAIL cited seq {entry['seq']}: line hash differs from the index")
        if (entry.get("prev") or "") != row["prev"]:
            sys.exit(f"FAIL cited seq {entry['seq']}: prev differs from the index")
        cited += 1
    print(f"PASS chain of {len(index)} entries; {cited} cited lines match their index hashes")


if __name__ == "__main__":
    main(sys.argv[1:])
