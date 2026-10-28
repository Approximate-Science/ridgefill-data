"""Section 8: longest shared calibration documents and calibration/LongBench v2 overlap counts and run lengths.

Read per-document tokenizer counts in provenance/inputs/calibration-docs.tsv, restricted to role=fit. Add Llama's
one BOS token as declared in the manifest; Qwen3.5 and Qwen3.8 must have the same maximum to support their joint
claim. Aggregate the per-pair measurements in provenance/checks/calib-vs-longbench-v2.json, never its summary or
verdict: eval_set=lbv2, a shared 50-word shingle or whole-text hash, users including qwen38-27b, and role fit or
excluded-token-table. Count distinct item ids across both roles and report fit-run extrema and the excluded-run
maximum. tools/overlap_lbv2.py defines whitespace-word runs measured from 13-shingle anchors; rerunning that text
scan would require withheld texts, but its released per-pair measurements suffice for these aggregates.
Human diagnostics go to stderr; stdout contains only claims.
"""
import csv
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, parse, say            # noqa: E402

args = parse(__doc__)


def describe(*parts):
    with redirect_stdout(sys.stderr):
        say(*parts)


manifest = need(args.root / "provenance" / "inputs" / "calibration-docs.tsv")
with manifest.open(encoding="utf-8", newline="") as stream:
    documents = [r for r in csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t")
                 if r["role"] == "fit"]
if not documents:
    raise SystemExit(f"{manifest}: no fit documents")

maximum = {}
for model in ("qwen3.5-9b", "qwen3.8", "qwen3-8b", "llama-3.1-8b-instruct"):
    bos = int(model == "llama-3.1-8b-instruct")
    maximum[model] = max(int(row["tokens_" + model]) + bos for row in documents)
    describe(f"{model}: longest fit document {maximum[model]:,} tokens, BOS={bos}, n={len(documents)} documents")
if maximum["qwen3.5-9b"] != maximum["qwen3.8"]:
    raise SystemExit("Qwen3.5 and Qwen3.8 maxima differ; their joint claim must be split")
claim("F.calibration.longest.qwen35_qwen38", f"{maximum['qwen3.5-9b']:,}")
claim("F.calibration.longest.qwen3", f"{maximum['qwen3-8b']:,}")
claim("F.calibration.longest.llama_with_bos", f"{maximum['llama-3.1-8b-instruct']:,}")

measurements = load(args.root / "provenance" / "checks" / "calib-vs-longbench-v2.json")["overlaps_50_or_hash"]
pairs = [r for r in measurements if r["eval_set"] == "lbv2"
         and "qwen38-27b" in r["used_by"].split(",")
         and r["role"] in ("fit", "excluded-token-table")
         and (r["shared_50"] > 0 or r["sha256_match"])]
fit_runs = [int(r["longest_run_words"]) for r in pairs if r["role"] == "fit"]
excluded_runs = [int(r["longest_run_words"]) for r in pairs if r["role"] == "excluded-token-table"]
if not fit_runs or not excluded_runs:
    raise SystemExit("missing input: LongBench v2 overlap pairs for fit and excluded-token-table documents")
items = len({r["item"] for r in pairs})
describe(f"27B calibration overlaps: {items} distinct LBv2 items across {len(pairs)} pairs; "
         f"fit runs {min(fit_runs)}-{max(fit_runs)} words, excluded maximum {max(excluded_runs)} words")
claim("F.calibration.lbv2.overlapping_items", items)
claim("F.calibration.lbv2.fit_run_words_min", min(fit_runs))
claim("F.calibration.lbv2.fit_run_words_max", max(fit_runs))
claim("F.calibration.lbv2.excluded_run_words_max", max(excluded_runs))
