"""Shared helpers of the recompute scripts: the repository root, inputs, claim lines and number formats.

Every script prints claim lines "<id>\t<value>" (one TAB) and human lines without a TAB. recompute/run_all.sh sends
the claim lines to stdout and everything else to stderr; recompute/gate.py compares the claim lines with
derived/claims.csv by exact string match, so a value is printed exactly as the paper prints it (ASCII minus).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse(description, extra=None):
    """--root: the tree to read (default: this repository). A negative control points it at a modified copy."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--root", type=Path, default=REPO, help="repository root holding results/, derived/, provenance/")
    if extra:
        extra(p)
    return p.parse_args()


def need(path):
    """The path itself, or exit 2 naming the missing input."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"missing input: {path}")
    return path


def load(path):
    return json.loads(need(path).read_text())


def rows(path):
    return [json.loads(line) for line in need(path).read_text().splitlines() if line.strip()]


def claim(key, value):
    value = str(value)
    if "\t" in key or "\t" in value or "\n" in value:
        raise ValueError(f"claim {key!r}: a TAB or newline in the id or value")
    print(f"{key}\t{value}", flush=True)


def say(*parts):
    """A human line (never a claim line: TABs become spaces)."""
    print(" ".join(str(p) for p in parts).replace("\t", " "), flush=True)


def num(x, nd):
    """Plain fixed-point number; a value that rounds to zero prints unsigned ("0.0", never "-0.0")."""
    s = f"{x:.{nd}f}"
    return s.lstrip("-") if float(s) == 0 else s


def sgn(x, nd):
    """Signed fixed-point number: "+0.010", "-0.004"; a value that rounds to zero prints unsigned ("0.000")."""
    s = num(abs(x), nd)
    return s if float(s) == 0 else ("+" if x > 0 else "-") + s


def ci(lo, hi, nd, signed=True):
    f = sgn if signed else num
    return f(lo, nd), f(hi, nd)


def claims_ci(key, est, lo, hi, nd, signed=True):
    """Three claim lines: <key>.delta, <key>.ci_lo, <key>.ci_hi."""
    f = sgn if signed else num
    claim(key + ".delta", f(est, nd))
    claim(key + ".ci_lo", f(lo, nd))
    claim(key + ".ci_hi", f(hi, nd))
