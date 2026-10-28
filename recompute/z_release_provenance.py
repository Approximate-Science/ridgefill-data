"""Z: Appendices F and G, release payloads, calibration lengths and overlaps.

Sum int8 map byte counts in provenance/external.tsv (binary GiB), excluding GGUF
conversions and the later bf16 source. Flash-Next's tensor package adds the
correction and row table; its file count also includes the manifest required by
code/c5/convert.py. The manifest's own byte length is not released.
Read per-document token lengths from calibration-docs.tsv, adding one BOS only
for Llama. Report medians/maxima and the Flash-Next extra corpus maximum rounded
to the nearest thousand. Bound Flash-Next row positions by the longest full fit
input, including fit chat copies. Count excluded document ids, and aggregate
per-pair word-overlap records from calib-vs-longbench-v2.json by evaluation set
and role. The fresh-set overlap is the recorded per-document word statistic;
the underlying text and the original token-level overlap report are withheld.
"""
import ast
import csv
import re
import statistics
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say        # noqa: E402

args = parse(__doc__)
P = args.root / "provenance"
with need(P / "external.tsv").open(newline="", encoding="utf-8") as f:
    external = list(csv.DictReader(f, delimiter="\t"))

for model, stem, nd in (("qwen38_27b", "Qwen3.8-27B", 1), ("flashnext", "Qwen3.8-Flash-Next", 2)):
    repo = "TheCacheIsALie/RidgeFill-Projector-" + stem + "-i8"
    entries = [r for r in external if r["kind"] == "map" and r["name"] == repo]
    maps = [r for r in entries if re.fullmatch(r"proj8\.L\d+\.safetensors", Path(r["file"]).name)]
    if not maps or len({r["file"] for r in maps}) != len(maps):
        raise ValueError(f"{repo}: missing or duplicated int8 map records")
    size = sum(int(r["bytes"]) for r in maps)
    claim(f"Z.release.{model}.maps_gib", num(size / 2**30, nd))
    with redirect_stdout(sys.stderr):
        say(f"{repo}: {len(maps)} int8 maps, {size} bytes, {size / 2**30:.9f} GiB")
    if model == "flashnext":
        ancillary = [r for r in entries if r["file"] in ("correction.safetensors", "rowsel.safetensors")]
        if {r["file"] for r in ancillary} != {"correction.safetensors", "rowsel.safetensors"} or len(ancillary) != 2:
            raise ValueError("missing or duplicated Flash-Next correction/row table")
        payload = maps + ancillary
        total = sum(int(r["bytes"]) for r in payload)
        claim("Z.release.flashnext.tensor_package_gib", num(total / 2**30, 2))
        # Read the loader without importing its torch/safetensors dependencies.
        source = need(args.root / "code" / "c5" / "convert.py").read_text()
        module = ast.parse(source)
        loader, = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "load_release"]
        manifest, = {n.right.value for n in ast.walk(loader)
                     if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                     and isinstance(n.left, ast.Name) and n.left.id == "src"
                     and isinstance(n.right, ast.Constant) and isinstance(n.right.value, str)
                     and n.right.value.endswith(".json")}
        filenames = {r["file"] for r in payload} | {manifest}
        claim("Z.release.flashnext.package_files", len(filenames))
        with redirect_stdout(sys.stderr):
            say(f"Flash-Next package: {total} tensor bytes ({total / 2**30:.9f} GiB), "
                f"{len(filenames)} files including {manifest}; manifest byte length unavailable")

with need(P / "inputs" / "calibration-docs.tsv").open(encoding="utf-8") as f:
    docs = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
fit = [r for r in docs if r["role"] == "fit"]
for short, column, bos in (("llama", "llama-3.1-8b-instruct", 1), ("qwen35", "qwen3.5-9b", 0),
                           ("qwen3", "qwen3-8b", 0), ("qwen38_27b", "qwen3.8", 0)):
    lengths = [int(r["tokens_" + column]) + bos for r in fit]
    claim(f"Z.calibration.{short}.median_tokens", f"{statistics.median(lengths):,.0f}")
    claim(f"Z.calibration.{short}.max_tokens", f"{max(lengths):,}")
    with redirect_stdout(sys.stderr):
        say(f"{short}: {len(lengths)} fit documents, BOS={bos}, sum={sum(lengths)}, "
            f"median={statistics.median(lengths)}, max={max(lengths)}")

extra = [int(r["tokens_qwen3.8"]) for r in docs if r["role"] == "flashnext-fit"]
claim("Z.calibration.flashnext.extra_max_tokens_approx", f"{round(max(extra), -3):,}")
fn_fit = [r for r in docs if "flashnext" in r["used_by"].split(",")
          and (r["role"] in ("fit", "flashnext-fit")
               or (r["role"] == "chat-copy" and r["doc_id"].startswith(("chattrain/", "chatbig/"))))]
row_bound = max(int(r["tokens_qwen3.8"]) for r in fn_fit)
claim("Z.calibration.flashnext.row_position_upper_bound", f"{row_bound:,}")

excluded = load(P / "inputs" / "calibration-exclude.json")["train"]
excluded_ids = {r["doc_id"] for r in docs if r["role"] == "excluded-token-table"}
if len(excluded) != len(set(excluded)) or set(excluded) != excluded_ids:
    raise ValueError("calibration exclusion list and per-document manifest disagree")
claim("Z.calibration.excluded_documents", len(excluded_ids))

pairs = load(P / "checks" / "calib-vs-longbench-v2.json")["overlaps_50_or_hash"]
lbv2 = [r for r in pairs if r["eval_set"] == "lbv2"]
content = [r["longest_run_words"] for r in lbv2
           if r["role"] == "flashnext-fit" and r["calib_doc"] == "prose-009"
           and r["item"] == "66ebd2475a08c7b9b35e0155"]
content_words, = content
claim("Z.overlap.lbv2.prose009.words", content_words)
fit_words = [r["longest_run_words"] for r in lbv2 if r["role"] == "fit"]
claim("Z.overlap.lbv2.fit.words_min", min(fit_words))
claim("Z.overlap.lbv2.fit.words_max", max(fit_words))
excluded_words = [r["longest_run_words"] for r in lbv2 if r["role"] == "excluded-token-table"]
claim("Z.overlap.lbv2.excluded.words_max", max(excluded_words))

fresh = need(P / "inputs" / "fresh-long-20261008" / "PROVENANCE.md").read_text()
header, = [line for line in fresh.splitlines() if line.startswith("| document |")]
columns = [cell.strip() for cell in header.strip("|").split("|")]
calibration_column = columns.index("calibration")
fresh_words = {}
for line in fresh.splitlines():
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    if len(cells) == len(columns) and cells[0].endswith(".txt"):
        fresh_words[cells[0]] = int(cells[calibration_column].split("/")[1].strip())
claim("Z.overlap.fresh.source_code_128000.calibration_words", fresh_words["source_code_128000_0.txt"])
with redirect_stdout(sys.stderr):
    say(f"Flash-Next: {len(extra)} extra documents, longest {max(extra)} tokens; "
        f"all {len(fn_fit)} fit inputs bounded by {row_bound} tokens")
    say(f"Excluded documents: {len(excluded_ids)}; LongBench v2 fit word range "
        f"{min(fit_words)}..{max(fit_words)}, excluded maximum {max(excluded_words)}")
    say("Fresh overlap is a recorded whole-calibration word statistic; document-role attribution is unavailable")
