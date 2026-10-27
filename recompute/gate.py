"""The recompute gate (DATA-PLAN Step 12): every row of derived/claims.csv must appear in the recompute output with
exactly its expected string. Usage: python3 -I recompute/gate.py [--claims derived/claims.csv] recompute.out
Exit 0 when every claim matches; exit 1 listing each mismatch (expected vs got), each claim missing from the output,
each claim printed twice, and each printed claim that derived/claims.csv does not list."""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path, help="the claim lines of recompute/run_all.sh")
p.add_argument("--claims", type=Path, default=REPO / "derived" / "claims.csv")
args = p.parse_args()
for f in (args.claims, args.output):
    if not f.is_file():
        sys.exit(f"missing input: {f}")
with args.claims.open(newline="") as f:
    rows = list(csv.DictReader(f))
expected = {r["id"]: r["expected"] for r in rows}
if len(expected) != len(rows):
    sys.exit(f"{args.claims}: duplicate ids")
got, twice = {}, []
for n, line in enumerate(args.output.read_text().splitlines(), 1):
    if "\t" not in line:
        sys.exit(f"{args.output}:{n}: not a claim line (no TAB): {line[:80]!r}")
    key, value = line.split("\t", 1)
    if key in got:
        twice.append(key)
    got[key] = value
bad = [k for k in expected if k in got and got[k] != expected[k]]
missing = [k for k in expected if k not in got]
extra = [k for k in got if k not in expected]
for k in bad:
    print(f"MISMATCH {k}: expected {expected[k]!r}, got {got[k]!r}")
for k in missing:
    print(f"MISSING  {k}: expected {expected[k]!r}, not in the output")
for k in sorted(set(twice)):
    print(f"TWICE    {k}: printed more than once")
for k in extra:
    print(f"UNLISTED {k}: printed ({got[k]!r}) but not in {args.claims.name}")
ok = len(expected) - len(bad) - len(missing)
print(f"gate: {ok} of {len(expected)} claims match; {len(bad)} mismatched, {len(missing)} missing, "
      f"{len(set(twice))} printed twice, {len(extra)} unlisted")
sys.exit(1 if bad or missing or twice or extra else 0)
