#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write MANIFEST.sha256 and MANIFEST.tsv for the repository.

MANIFEST.sha256: `sha256sum` format, sorted by path, every tracked file except itself.
MANIFEST.tsv: one row per file except the two manifests, with columns
  path, bytes, sha256, licence, kind, status, claims, origin, origin_sha256, transform.
`origin` and `transform` come from provenance/STRIPPED.tsv for transformed files, and from the longest matching
prefix in provenance/ORIGINS.tsv otherwise (origin = alias + the path below the prefix, transform from ORIGINS).

--verify-origins ALIASES.tsv (build machine only; alias<TAB>local folder) checks that every file whose transform is
`copy` is byte-identical to its origin and that every STRIPPED.tsv origin hash matches its origin file.
Usage: python3 tools/make_manifest.py . [--verify-origins aliases.tsv]
"""
import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

SKIP = {"MANIFEST.sha256", "MANIFEST.tsv"}

LICENCES = [  # first match wins
    ("code/qwen38-27b-kit/vendor/", "upstream (MIT: LongBench pieces; Apache-2.0: RULER metrics)"),
    ("code/", "Apache-2.0"), ("recompute/", "Apache-2.0"), ("tools/", "Apache-2.0"), (".github/", "Apache-2.0"),
    ("LICENSE", "licence text"), ("", "CC-BY-4.0"),
]
KINDS = [("results/", "result"), ("derived/", "derived"), ("provenance/", "provenance"), ("code/", "code"), (".github/", "code"),
         ("recompute/", "code"), ("tools/", "code"), ("figures/", "figure"), ("", "documentation")]
CLAIMS = [  # path prefix -> claim ids (CLAIMS.md); first match wins
    ("results/reference/llama-3.1-8b-instruct-ruler-deep/", "LB11"),
    ("results/reference/qwen3-8b-ruler-deep/", "LB11"),
    ("results/reference/llama-3.1-8b-instruct-nll-fresh/", "LB11,S8"),
    ("results/reference/llama-3.1-8b-instruct-ruler/", "LB8,R1,S9,S10"),
    ("results/reference/qwen3.5-9b-ruler/", "LB8,R2,S9,S10"),
    ("results/reference/qwen3-8b-ruler/", "LB8,R4,S9,S10"),
    ("results/reference/llama-3.1-8b-instruct-nll-causal/", "S7,S10"),
    ("results/reference/qwen3.5-9b-nll-causal/", "S7,S10"),
    ("results/reference/qwen3-8b-nll-causal/", "S7,S10"),
    ("results/reference/llama-3.1-8b-instruct-nll/", "S7,S8,R3,R6,LB10"),
    ("results/reference/qwen3.5-9b-nll/", "S7,S8"),
    ("results/reference/qwen3-8b-nll/", "S7,S8"),
    ("results/reference/llama-3.1-8b-instruct-decode/", "LB9"),
    ("results/reference/qwen3.5-9b-decode/", "LB9"),
    ("results/reference/qwen3-8b-decode/", "LB9"),
    ("results/reference/llama-3.1-8b-instruct-longbench/", "S11"),
    ("results/reference/llama-3.1-8b-instruct-secondq/", "S12"),
    ("results/reference/qwen3.5-9b-secondq/", "S12"),
    ("results/reference/qwen3-8b-secondq/", "S12"),
    ("results/flashnext/final-det-20261010/results/graded/", "S5,S6,LB7d"),
    ("results/flashnext/", "LB7a,LB7b,LB7c,LB7d,S6"),
    ("results/qwen38-27b/longbench/", "LB4"),
    ("results/qwen38-27b/results/timing/", "LB2"),
    ("results/qwen38-27b/results/ruler/", "LB3"),
    ("results/qwen38-27b/results/kld", "LB5,S1"),
    ("results/qwen38-27b/results/checks/", "LB6"),
    ("results/qwen38-27b/results/mix/", "S3"),
    ("results/qwen38-27b/results/MIX-HARNESS-AMENDMENT.json", "S3"),
    ("results/qwen38-27b/results/RUN-MANIFEST.json", "LB2,LB6"),
    ("results/qwen38-27b/results/SCORECARD", "S4,LB2,LB3,LB5"),
    ("results/qwen38-27b/PER-DOC-KL.json", "LB5,S1"),
    ("results/qwen38-27b/BF16-", "S2"),
    ("results/qwen38-27b/bf16/", "S2"),
    ("derived/per-document-nll.csv", "S7,S8,R6"),
    ("provenance/fits/", "LB1a"),
    ("provenance/inputs/qwen38-27b/LOCK.json", "LB6"),
    ("results/development/closures/", "D1"),
    ("results/development/whiteness/", "D1"),
    ("results/development/needles-dev/", "D2"),
    ("results/development/needles-reference/", "D2"),
    ("results/development/token-classes/", "D3"),
    ("results/development/certificate-27b/", "D4"),
    ("results/development/needles-vs-ruler-27b/", "D5"),
]


def first(table, path, default=""):
    return next((value for prefix, value in table if path.startswith(prefix)), default)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tracked_files(root):
    """Every file under root except .git and Python caches; fails if git would leave any of them out."""
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and ".git" not in p.relative_to(root).parts and "__pycache__" not in p.parts
                   and p.suffix != ".pyc")
    ignored = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--others", "--ignored", "--exclude-standard"],
                             capture_output=True, check=True).stdout.decode().split("\0")
    left_out = sorted(set(files) & set(ignored))
    if left_out:
        sys.exit(f"git ignores {len(left_out)} files that the manifest would list, e.g. {left_out[:3]}")
    return [f for f in files if f not in SKIP]


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--verify-origins", type=Path)
    args = p.parse_args(argv)
    root = args.root
    stripped = {r["path"]: r for r in read_tsv(root / "provenance/STRIPPED.tsv")}
    origins = sorted(read_tsv(root / "provenance/ORIGINS.tsv"), key=lambda r: -len(r["prefix"]))
    aliases = dict(r.split("\t") for r in args.verify_origins.read_text().splitlines() if r) if args.verify_origins else None
    rows, problems = [], []
    for rel in tracked_files(root):
        path = root / rel
        digest = sha256(path)
        if rel in stripped:
            s = stripped[rel]
            origin, origin_sha, transform = s["origin"], s["origin_sha256"], s["transform"]
            if s["sha256"] != digest:
                problems.append(f"{rel}: sha256 differs from STRIPPED.tsv")
        else:
            o = next((o for o in origins if rel == o["prefix"] or o["prefix"] == ""
                      or (o["prefix"].endswith("/") and rel.startswith(o["prefix"]))), None)
            if o is None:
                problems.append(f"{rel}: no ORIGINS.tsv prefix covers it")
                continue
            below = rel[len(o["prefix"]):] if o["prefix"].endswith("/") or o["prefix"] == "" else ""
            origin = o["alias"] + below
            origin_sha, transform = (digest if o["transform"] == "copy" else ""), o["transform"]
        if aliases is not None and (transform == "copy" or rel in stripped) and ":" not in origin.split("/")[0]:
            alias, _, below = origin.partition("/")
            base = aliases.get(alias)
            if base is None:
                problems.append(f"{rel}: alias {alias!r} missing from --verify-origins")
            else:
                source = Path(base) / below
                if origin_sha == "withheld":
                    if not source.is_file():
                        problems.append(f"{rel}: origin {origin} missing")
                elif not source.is_file() or sha256(source) != origin_sha:
                    problems.append(f"{rel}: origin {origin} does not match")
        status = ("n/a" if not rel.startswith("results/") or rel.endswith(".md")
                  else "DEVELOPMENT" if rel.startswith("results/development/") else "FINAL")
        rows.append([rel, str(path.stat().st_size), digest, first(LICENCES, rel), first(KINDS, rel), status,
                     first(CLAIMS, rel), origin, origin_sha, transform])
    if problems:
        print("\n".join(problems[:50]), file=sys.stderr)
        sys.exit(f"{len(problems)} problems; manifest not written")
    with (root / "MANIFEST.tsv").open("w", encoding="utf-8", newline="") as f:
        f.write("path\tbytes\tsha256\tlicence\tkind\tstatus\tclaims\torigin\torigin_sha256\ttransform\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    listing = [(r[0], r[2]) for r in rows] + [("MANIFEST.tsv", sha256(root / "MANIFEST.tsv"))]
    with (root / "MANIFEST.sha256").open("w", encoding="utf-8", newline="") as f:
        for rel, digest in sorted(listing):
            f.write(f"{digest}  {rel}\n")
    print(f"MANIFEST.tsv: {len(rows)} rows; MANIFEST.sha256: {len(listing)} files")


if __name__ == "__main__":
    main(sys.argv[1:])
