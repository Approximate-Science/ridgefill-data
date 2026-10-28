"""X: four numbers whose raw records joined the release in 1.4.

- §9: the ledger entries the results rest on (provenance/ledger/rest-on.tsv: the registrations of the released cells
  and the records that amend them), how many are published in full (cited.jsonl) and how many as hashes only.
- §2.3: terminal-state states captured per layer for the Flash-Next correction (provenance/fits/flashnext/
  sterm-kva-big-s24.json): one state every 2,048 tokens of each calibration prompt's prefix, summed over prompts.
- §5.1: decode-throughput cost of the determinism fix (results/flashnext/determinism-throughput/): per request,
  (generated tokens - 1) / (wall - time to first token); median of three requests per length; cost = 1 - fixed /
  unfixed, at 16K, 32K and 64K.
- Appendix I.2: the development checkpoint's exact arm on the corrected answer format, 60 needle prompts
  (results/development/needles-dev/s2-niahchat/; arm "off"; correct when the reply is the needle's value)."""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say       # noqa: E402

args = parse(__doc__)
R = args.root

rest = list(csv.DictReader(need(R / "provenance/ledger/rest-on.tsv").open(newline=""), delimiter="\t"))
cited = {json.loads(line)["seq"] for line in need(R / "provenance/ledger/cited.jsonl").read_text().splitlines() if line}
full = {int(r["seq"]) for r in rest if r["published"] == "full"}
if full != cited:
    raise SystemExit(f"rest-on.tsv marks {sorted(full ^ cited)} differently from cited.jsonl")
claim("X.ledger.rest_on", len(rest))
claim("X.ledger.full", len(full))
claim("X.ledger.hash_only", sum(r["published"] == "hash only" for r in rest))

st = load(R / "provenance/fits/flashnext/sterm-kva-big-s24.json")
states = sum(len(p["chunks"]) for p in st["prompts"])
counts = {v for layer in st["counts"].values() for v in (layer.values() if isinstance(layer, dict) else [layer])}
if counts != {states}:
    raise SystemExit(f"per-layer state counts {sorted(counts)} differ from the {states} sampled boundaries")
say(f"terminal-state capture: {len(st['prompts'])} prompts, {states} states per layer")
claim("X.sterm.prompts", len(st["prompts"]))
claim("X.sterm.states", states)

T = R / "results/flashnext/determinism-throughput"
cost = {}
for length in (16384, 32768, 65536):
    med = {}
    for arm in ("fixed", "unfixed"):
        rates = []
        for rep in range(3):
            g = load(T / arm / f"gen-{length}.r{rep}.json")
            rates.append((g["n_gen"] - 1) / (g["wall_s"] - g["ttft_s"]))
        med[arm] = statistics.median(rates)
    cost[length] = 100 * (1 - med["fixed"] / med["unfixed"])
    say(f"{length}: decode {med['unfixed']:.3f} -> {med['fixed']:.3f} tok/s, cost {cost[length]:.2f} %")
    claim(f"X.detfix.cost_pct.{length}", num(cost[length], 2))
claim("X.detfix.cost_pct.min", num(min(cost.values()), 2))
claim("X.detfix.cost_pct.max", num(max(cost.values()), 2))

recs = [load(p) for p in sorted((R / "results/development/needles-dev/s2-niahchat").glob("niah-*.json"))]
off = [next(x for x in r["runs"] if x["mode"] == "off") for r in recs]
right = sum(o["text"].strip() == r["value"] for o, r in zip(off, recs))
claim("X.niahchat.exact", f"{right} of {len(recs)}")
