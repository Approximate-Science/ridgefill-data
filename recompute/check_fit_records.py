"""LB1a: the fit records (provenance/fits/). Reference models: calibration rows and documents per model (the same for
every split), lambda per split, and every projector sha256 a reference cell declares (results/reference/*/cell.json)
equals the fit record of that model and split; the cells' declared calibration rows equal the fit records'.
Qwen3.8-27B: rows, documents, lambda per split, the number of maps at split 16, and the fit's wall time from the first
to the last line of fit-q4.log. Flash-Next: captured rows (raw + chat), the weighted row count, the chosen lambda of the
released maps (kva-s24.safetensors) and the number of held-out captures."""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, parse, say           # noqa: E402

FITS = {"llama": "llama-3.1-8b-instruct", "qwen35": "qwen3.5-9b", "qwen3": "qwen3-8b"}
args = parse(__doc__)
P = args.root / "provenance" / "fits"
records = {}
for short, model in FITS.items():
    recs = {int(re.search(r"-s(\d+)\.json$", f.name).group(1)): load(f)
            for f in sorted(need(P / "reference").glob(f"fit-sweep-{model}-s*.json"))}
    records[model] = recs
    rows = {r["rows"] for r in recs.values()}
    docs = {r["docs"] for r in recs.values() if "docs" in r}
    if not docs:
        docs = {load(P / "reference" / f"calib-docs-{model}-sweep.json")["count"]}
    if len(rows) != 1 or len(docs) != 1:
        raise SystemExit(f"{model}: the splits' fit records disagree on rows {rows} or documents {docs}")
    say(f"{model}: {rows} rows, {docs} documents, lambda per split "
        f"{ {s: r['lambda_star'] for s, r in sorted(recs.items())} }")
    claim(f"LB1a.{short}.rows", rows.pop())
    claim(f"LB1a.{short}.docs", docs.pop())
    for s, r in sorted(recs.items()):
        claim(f"LB1a.{short}.lambda.s{s}", f"{r['lambda_star']:g}")
checked, bad = 0, []
for cell_json in sorted(need(args.root / "results" / "reference").glob("*/cell.json")):
    cell = load(cell_json)
    model = cell["model"]["name"]
    proj = cell.get("projector") or {}
    for f in proj.get("files", []):
        rec = records[model].get(f["layer"])
        checked += 1
        if rec is None or rec["projector_sha256"] != f["sha256"]:
            bad.append(f"{cell['name']} layer {f['layer']}")
    if proj.get("provenance", {}).get("calib_rows") not in (None, next(iter(records[model].values()))["rows"]):
        bad.append(f"{cell['name']} calib_rows")
if not checked:
    raise SystemExit("no projector sha256 in the reference cells")
say(f"reference cells: {checked} projector hashes checked, mismatches {bad or 'none'}")
claim("LB1a.projector_sha256_mismatches", len(bad))
q = {int(re.search(r"-s(\d+)\.json$", f.name).group(1)): load(f)
     for f in sorted(need(P / "qwen38-27b").glob("fit-sweep-qwen3.8-27b-s*.json"))}
claim("LB1a.qwen38-27b.rows", q[16]["rows"])
claim("LB1a.qwen38-27b.docs", q[16]["docs"])
for s, r in sorted(q.items()):
    claim(f"LB1a.qwen38-27b.lambda.s{s}", f"{r['lambda_star']:g}")
claim("LB1a.qwen38-27b.maps_s16", len(q[16]["per_layer_read_cos"][f"{q[16]['lambda_star']:g}"]))
log = need(P / "qwen38-27b" / "fit-q4.log").read_text()
times = [line[:8] for line in log.splitlines() if re.match(r"\d\d:\d\d:\d\d ", line)]
last = re.search(r"fit: done \S*T(\d\d:\d\d:\d\d)", log)
end = last.group(1) if last else times[-1]
sec = lambda t: int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8])                    # noqa: E731
minutes = (sec(end) - sec(times[0])) % 86400 / 60
say(f"27B fit: {times[0]} to {end}, {minutes:.1f} min for splits {sorted(q)}")
claim("LB1a.qwen38-27b.fit_minutes", f"{minutes:.0f}")
fn = load(P / "flashnext" / "report-big-s24.json")
claim("LB1a.flashnext.rows", sum(fn["rows"]["plain"].values()))
claim("LB1a.flashnext.weighted_rows", f"{fn['train_rows']:.0f}")
claim("LB1a.flashnext.lambda", f"{fn['files']['kva-s24.safetensors']['lambdas']['all']:g}")
claim("LB1a.flashnext.held_captures", len(fn["held"]))
say(f"Flash-Next: rows {fn['rows']['plain']}, weighted {fn['train_rows']:.0f}, lambda "
    f"{fn['files']['kva-s24.safetensors']['lambdas']}")
