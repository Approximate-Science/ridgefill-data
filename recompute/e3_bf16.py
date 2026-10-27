"""S2: the full-precision BF16 model on 25 documents at 32K chosen by disagreement
(results/qwen38-27b/BF16-SELECTION.json: the 10 largest-KL documents with E3's NLL below stock's, the 10 with it
above, 5 near the median). Per document: mean NLL of stock and E3 on the 512 scored tokens (the KL records) and of
BF16 (bf16/<id>.json), and each arm's mean absolute target log-probability difference from BF16; stock is "closer
to BF16" when its difference is the smaller. Recomputed from the raw records and checked against
results/qwen38-27b/BF16-COMPARISON.json."""
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say            # noqa: E402

args = parse(__doc__)
Q = args.root / "results" / "qwen38-27b"
selection = load(Q / "BF16-SELECTION.json")
stored = {x["id"]: x for x in load(Q / "BF16-COMPARISON.json")}
rows = []
for group, ids in selection.items():
    for i in ids:
        b = load(Q / "bf16" / f"{i}.json")
        k = load(Q / "results" / "kld" / "32768" / f"{i}.json")
        bl, sl, el = b["target_logprobs"], k["stock"]["logprobs"], k["e3"]["logprobs"]
        if not len(bl) == len(sl) == len(el) == 512:
            raise SystemExit(f"{i}: {len(bl)} / {len(sl)} / {len(el)} scored positions, expected 512")
        r = dict(g=group, id=i, s=k["stock"]["nll"], e=k["e3"]["nll"], b=b["nll"],
                 ds=sum(abs(x - y) for x, y in zip(sl, bl)) / 512, de=sum(abs(x - y) for x, y in zip(el, bl)) / 512)
        x = stored[i]
        if x["g"] != group or any(abs(r[k_] - x[k_]) > 1e-9 for k_ in ("s", "e", "b", "ds", "de")):
            raise SystemExit(f"{i}: recomputed values differ from BF16-COMPARISON.json")
        rows.append(r)
closer = sum(r["ds"] < r["de"] for r in rows)
say(f"stock closer to BF16 than E3 on {closer} of {len(rows)} documents")
claim("S2.docs", len(rows))
claim("S2.stock_closer", closer)
g = [r for r in rows if r["g"] == "e3_better"]
for key, arm in (("stock", "s"), ("e3", "e"), ("bf16", "b")):
    v = statistics.mean(r[arm] for r in g)
    say(f"largest-KL group with E3 lower ({len(g)} documents): {key} mean NLL {v:.4f}")
    claim(f"S2.e3_lower_group.{key}", num(v, 3))
claim("S2.e3_lower_group.docs", len(g))
doc = next(r for r in rows if r["id"].startswith("66ecf139"))
for key, arm in (("stock", "s"), ("e3", "e"), ("bf16", "b")):
    claim(f"S2.doc_66ecf139.{key}", num(doc[arm], 3))
say(f"document {doc['id']}: stock {doc['s']:.4f}, E3 {doc['e']:.4f}, BF16 {doc['b']:.4f}")
