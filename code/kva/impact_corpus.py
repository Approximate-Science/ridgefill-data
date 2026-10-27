"""Impact corpus (BIGDATA): word 50-gram fingerprints for the overlap and near-duplicate checks, and the scored
window cut points.

Grams are 50 consecutive whitespace-separated words of decoded text, the locked set's rule
(datasets/test-set-locked/finalize.py), hashed to uint64 so reference sets of tens of millions of grams fit in
sorted numpy arrays. JSON files contribute the grams of each string value on its own (decoded text, not the
escaped file bytes).
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

GRAM = 50
_BASE = np.uint64(0x9E3779B97F4A7C15)
_word = {}

# cut points: fraction of the document (reference tokens), the shortest allowed prompt and the least distance
# between two cuts of one document (one tail, so the windows' exact tails never overlap)
FRACS = (0.5, 0.75, 1.0)
MIN_CUT = 8192
GAP = 1024


def _word_ids(ws):
    out = np.empty(len(ws), np.uint64)
    for i, w in enumerate(ws):
        h = _word.get(w)
        if h is None:
            h = _word[w] = int.from_bytes(hashlib.blake2b(w.encode("utf-8", "surrogatepass"), digest_size=8).digest(),
                                          "little")
        out[i] = h
    return out


def grams(text, n=GRAM):
    """Hashes of every word n-gram of text, in order (a polynomial hash of stable per-word hashes)."""
    w = _word_ids(text.split())
    if len(w) < n:
        return np.zeros(0, np.uint64)
    h = np.zeros(len(w) - n + 1, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _BASE + w[k:k + len(h)]
    return h


def strings_of(obj):
    """Every string value in a parsed JSON object (keys excluded)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings_of(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings_of(v)


def file_grams(path):
    """Unique grams of a reference file: .txt as text, .json / .jsonl per decoded string."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".json":
        parts = list(strings_of(json.loads(raw)))
    elif path.suffix == ".jsonl":
        parts = [s for line in raw.splitlines() if line.strip() for s in strings_of(json.loads(line))]
    else:
        parts = [raw]
    g = [grams(s) for s in parts]
    return np.unique(np.concatenate(g)) if g else np.zeros(0, np.uint64)


def reference(paths, texts=()):
    """Sorted unique grams of the files plus extra texts (e.g. kva.data.calib_docs)."""
    parts = [file_grams(p) for p in paths] + [grams(t) for t in texts]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.uint64)


def shared(g, ref):
    """Number of distinct grams of g that are in the sorted reference array ref."""
    u = np.unique(g)
    if not len(u) or not len(ref):
        return 0
    i = np.minimum(np.searchsorted(ref, u), len(ref) - 1)
    return int((ref[i] == u).sum())


def self_repeat(g):
    """Share of a document's gram positions that repeat an earlier gram of the same document: long repeated
    boilerplate (navigation, licence blocks, generated tables) shows up here."""
    return 1.0 - len(np.unique(g)) / len(g) if len(g) else 0.0


class Seen:
    """The accepted documents' grams, for exact and near-duplicate rejection within the corpus."""

    def __init__(self):
        self.owner = {}

    def containment(self, g):
        """(share of g's distinct grams found in the most-overlapping accepted document, that document)."""
        u = np.unique(g).tolist()
        hits = {}
        for x in u:
            d = self.owner.get(x)
            if d is not None:
                hits[d] = hits.get(d, 0) + 1
        if not hits:
            return 0.0, None
        d = max(hits, key=hits.get)
        return hits[d] / len(u), d

    def add(self, name, g):
        for x in np.unique(g).tolist():
            self.owner.setdefault(x, name)


def cut_points(n, fracs=FRACS, min_cut=MIN_CUT, gap=GAP):
    """Prompt lengths (reference tokens) of a document's scored windows: each fraction of n, raised to min_cut,
    kept if it is at least gap away from every cut already kept (the longest first: the document's own end is
    always a window). Ascending; [] if the document is shorter than min_cut."""
    cuts = []
    for f in sorted(fracs, reverse=True):
        c = max(min_cut, round(f * n))
        if c <= n and all(abs(c - k) >= gap for k in cuts):
            cuts.append(c)
    return sorted(cuts)


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fields=None):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
