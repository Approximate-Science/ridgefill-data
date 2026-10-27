#!/usr/bin/env python3
"""The calibration documents a sweep fit consumed, read from the fit's own capture state, for fits whose projector
headers and fit reports predate the 'docs' field (scripts/fit_sweep.py records it since e86e7a1; the Qwen3-8B sweep
ran at cba2d91). fit_sweep.py writes docs=[names] into each split's sums file (<sums>/<model>-s<S>.pt, a torch zip);
this reads only the pickle inside it (no tensor is loaded), checks that every split lists the same documents and
counted the fit report's rows, and writes results/calib-docs-<model>-sweep.json: the names, their count, per sums file
the sha256 of its pickle, its rows and code commit, and per split the projector sha256 of its fit report.
finals/lib/refit.py docs_count() reads the count by that sha256 (finals/tables.py, finals/make_cells.py);
tests/test_finals_wiring.py checks the names against the committed clean calibration list.
Usage: calib_docs_of_sums.py MODEL SPLITS [--sums DIR] [--out FILE]
       (SPLITS e.g. 9,13,18,22,24; DIR default $KVA_DATA/sums; FILE default results/calib-docs-<MODEL>-sweep.json)
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class _NoTensors(pickle.Unpickler):
    """Unpickles a torch.save dict without torch: tensors and torch objects become None."""

    def find_class(self, module, name):
        if module.startswith("torch"):
            return lambda *a, **k: None
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None


def capture_state(path):
    """(pickle sha256, the saved dict with tensors as None) of one sums file."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1:
            sys.exit(f"{path}: not a torch.save zip with one data.pkl ({names})")
        raw = z.read(names[0])
    return hashlib.sha256(raw).hexdigest(), _NoTensors(io.BytesIO(raw)).load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("splits")
    ap.add_argument("--sums", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sums = Path(args.sums) if args.sums else (Path(os.environ["KVA_DATA"]) / "sums" if os.environ.get("KVA_DATA")
                                              else None)
    if sums is None or not sums.is_dir():
        sys.exit("calib_docs_of_sums: name the sums directory (--sums, or KVA_DATA with a sums/ inside)")
    out = dict(model=args.model, rule="the docs list scripts/fit_sweep.py stored in each split's capture state",
               sums={}, projectors={})
    docs = None
    for s in (int(x) for x in args.splits.split(",")):
        f = sums / f"{args.model}-s{s}.pt"
        digest, state = capture_state(f)
        report = json.loads((REPO / "results" / f"fit-sweep-{args.model}-s{s}.json").read_text())
        if state["model"] != args.model or state["n"] != report["rows"]:
            sys.exit(f"{f}: model {state['model']} rows {state['n']}, the fit report says {args.model} {report['rows']}")
        if docs is not None and state["docs"] != docs:
            sys.exit(f"{f}: another document list than the other splits")
        docs = state["docs"]
        out["sums"][f.name] = dict(pickle_sha256=digest, rows=state["n"], code_commit=state["code_commit"])
        out["projectors"][str(s)] = report["projector_sha256"]
    out.update(docs=docs, count=len(docs))
    dest = Path(args.out) if args.out else REPO / "results" / f"calib-docs-{args.model}-sweep.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{dest}: {len(docs)} documents, {len(out['sums'])} splits")


if __name__ == "__main__":
    main()
