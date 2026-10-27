#!/usr/bin/env python3
"""Build (or --check) the manifests of the prompt sets derived from the LOCKED RULER rows (kva/derived.py): the two-key
set of T1.8 (datasets/derived/ruler-twokey.json) and the question-before-document set of T1.4
(datasets/derived/ruler-qfirst.json). Reads the locked rows (datasets/test-set-locked, or KVA_LOCKED_DIR) after the
MANIFEST / FROZEN checks; writes sample ids, row numbers, needle ordinals and sha256 hashes only, never a text; prints
counts only. Deterministic (seed 20261004): a rebuild is byte for byte the committed file, which --check verifies.

Usage: build_derived.py [--check] [--out DIR]        (default DIR: datasets/derived, or KVA_DERIVED_DIR)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kva import derived  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if a committed manifest differs from a rebuild")
    ap.add_argument("--out", default=str(derived.DIR))
    args = ap.parse_args()
    out, bad = Path(args.out), []
    for which in derived.SETS:
        text = derived.dumps(derived.build_manifest(which))
        path = out / derived.manifest_path(which).name
        n = text.count('"id": ')
        if args.check:
            same = path.is_file() and path.read_text() == text
            print(f"{path.name}: {n} samples, {'identical' if same else 'DIFFERS from the rebuild'}")
            bad += [] if same else [path.name]
        else:
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{path.name}: {n} samples written")
    if bad:
        sys.exit(f"build_derived: {', '.join(bad)} not reproduced by the builder")


if __name__ == "__main__":
    main()
