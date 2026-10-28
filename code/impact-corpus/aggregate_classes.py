#!/usr/bin/env python3
"""Reduce the private impact corpus to counts and absolute first-order impact.

Requires NumPy and the original corpus; the public recompute script does not.
Usage: python3 -B aggregate_classes.py --corpus-root CORPUS --output AGGREGATE

CORPUS contains the two model directories below, each with index.json and its
listed NPZ files. No token IDs, text, document names, or runtime metadata are
written. This implements the token-class part of s4_extras.py with load.py's
float32 conversion and row-0 exclusion, streaming one window at a time.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import numpy as np

MODELS = ("llama-3.1-8b-instruct", "qwen3.5-9b")
CLASSES = ("special", "newline", "space", "number", "punct", "word-start",
           "word-start-cap", "word-piece", "word-piece-cap", "mixed")


def digest_list(digests):
    """Hash sorted lowercase hex digests, one per LF-terminated ASCII line."""
    return hashlib.sha256("".join(d + "\n" for d in sorted(digests)).encode("ascii")).hexdigest()


def aggregate_model(corpus_root, model):
    model_root = corpus_root / model
    index_bytes = (model_root / "index.json").read_bytes()
    index = json.loads(index_bytes)
    if tuple(index["classes"]) != CLASSES:
        raise ValueError("unexpected class definitions")
    windows = sorted(index["windows"], key=lambda w: (w["doc"], w["window"]))
    if not windows or index["n_windows"] != len(windows):
        raise ValueError("inconsistent window count")
    if len({w["file"] for w in windows}) != len(windows):
        raise ValueError("duplicate NPZ file in index")
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    sums = [[] for _ in CLASSES]
    total_sums, digests = [], []
    source_bytes = 0
    for window in windows:
        relative = Path(window["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npz":
            raise ValueError("expected a relative NPZ filename")
        data = (model_root / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", window["sha256"]) or digest != window["sha256"]:
            raise ValueError("NPZ checksum differs from index")
        digests.append(digest)
        source_bytes += len(data)
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            impact = arrays["impact"]
            classes = arrays["cls"]
        if impact.shape != (window["P"],) or classes.shape != impact.shape:
            raise ValueError("NPZ rows disagree with prefix length")
        # load.py casts impact to float32; s4_extras.py sums |impact| as float64.
        absolute = np.abs(impact[1:].astype(np.float32)).astype(np.float64)
        classes = classes[1:]
        if (not np.issubdtype(classes.dtype, np.integer) or
                np.any(classes >= len(CLASSES)) or np.any(classes < 0) or
                not np.all(np.isfinite(absolute))):
            raise ValueError("invalid class IDs or nonfinite impact")
        counts += np.bincount(classes, minlength=len(CLASSES))
        total_sums.append(float(absolute.sum(dtype=np.float64)))
        for class_id in range(len(CLASSES)):
            sums[class_id].append(float(absolute[classes == class_id].sum(dtype=np.float64)))
    return {
        "n_documents": len({w["doc"] for w in windows}),
        "n_windows": len(windows),
        "totals": {"N": int(counts.sum()), "sum_abs_impact": math.fsum(total_sums)},
        "classes": {
            label: {"N_c": int(counts[c]), "sum_abs_impact": math.fsum(sums[c])}
            for c, label in enumerate(CLASSES)
        },
        "provenance": {
            "index": f"impact-corpus/{model}/index.json",
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "npz_count": len(digests),
            "npz_bytes": source_bytes,
            "npz_checksums_verified_against_index": True,
            "sorted_npz_sha256_list_sha256": digest_list(digests),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Verify that both models measure the same document set, without publishing names.
    document_sets = [
        {w["doc"] for w in json.loads((args.corpus_root / model / "index.json").read_bytes())["windows"]}
        for model in MODELS
    ]
    if document_sets[0] != document_sets[1]:
        raise ValueError("model document sets differ")
    models = {}
    for model in MODELS:
        models[model] = aggregate_model(args.corpus_root, model)
        print(f"{model}: {models[model]['n_windows']} windows aggregated", file=sys.stderr, flush=True)
    result = {
        "schema": "impact-corpus-classes-v1",
        "method": {
            "quantity": "absolute first-order impact",
            "pooling": "all indexed windows, separately for each model; repeated prefix rows count once per window",
            "excluded_rows": "row 0 of every window, for both class and total counts and impact sums",
            "arithmetic": "impact cast to float32 before abs, then float64 sums within windows and math.fsum across windows",
            "enrichment": "E_c = (sum_abs_impact_c / sum_abs_impact_total) / (N_c / N)",
            "class_assignment": "existing NPZ cls IDs, in index.json classes order; no reclassification",
            "protected_classes": ["word-start-cap", "word-piece-cap", "word-piece", "mixed"],
            "range": "minimum and maximum over individual model/class enrichments, without pooling models",
            "reference_code": ["results/bigdata/analysis-opus/s4_extras.py", "results/bigdata/analysis-opus/load.py"],
            "checksum_encoding": "per model: SHA-256 of lexicographically sorted lowercase NPZ SHA-256 hex digests, each followed by LF, encoded as ASCII; duplicates retained",
        },
        "models": models,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
