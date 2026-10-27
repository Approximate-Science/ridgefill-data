"""LB5, S1: Qwen3.8-27B agreement with stock after the prompt. 100 LongBench v2 documents per length (16,384 and 32,768
prompt tokens), 512 scored positions each (results/qwen38-27b/results/kld/<length>/<id>.json, documents and order from
provenance/inputs/qwen38-27b/kld-index.json). Per-position KL(stock || E3) and top-1 agreement are pooled over the
51,200 positions (median, p99 = the ceil(0.99 n)-th smallest, agreement); mean KL and mean dNLL (E3 - stock) are
document means with the kit's bootstrap (cloud-kit common.bootstrap: random.Random(1), 10,000 resamples, percentile).
S1: the documents split by copyable share (results/qwen38-27b/PER-DOC-KL.json, checked against the KL records)."""
import math
import random
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say, sgn       # noqa: E402

P99_ND = {16384: 3, 32768: 2}                                # the paper prints 0.965 and 12.77


def bootstrap(values, repeats=10000, seed=1):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(repeats))
    return statistics.mean(values), bs[int(.025 * repeats)], bs[min(repeats - 1, int(.975 * repeats))]


args = parse(__doc__)
Q = args.root / "results" / "qwen38-27b"
index = load(args.root / "provenance" / "inputs" / "qwen38-27b" / "kld-index.json")
per_doc = {(x["L"], x["id"]): x for x in load(Q / "PER-DOC-KL.json")}
for length in (16384, 32768):
    items = [load(Q / "results" / "kld" / str(length) / (x["id"] + ".json")) for x in index if x["length"] == length]
    kl = [v for r in items for v in r["e3"]["kl"]]
    top1 = [v for r in items for v in r["e3"]["top1"]]
    if len(items) != 100 or len(kl) != len(top1) != 51200:
        raise SystemExit(f"{length}: {len(items)} documents, {len(kl)} positions; expected 100 and 51,200")
    s = sorted(kl)
    m, lo, hi = bootstrap([r["e3"]["kl_mean"] for r in items])
    d, dlo, dhi = bootstrap([r["dnll"] for r in items])
    say(f"{length}: mean KL {m:.5f} [{lo:.5f}, {hi:.5f}], median {statistics.median(kl):.5f}, "
        f"p99 {s[math.ceil(.99 * len(s)) - 1]:.4f}, top-1 {100 * statistics.mean(top1):.3f} %, "
        f"dNLL {d:+.5f} [{dlo:+.5f}, {dhi:+.5f}], median dNLL {statistics.median(r['dnll'] for r in items):+.5f}")
    claim(f"LB5.{length}.n_positions", len(kl))
    claim(f"LB5.{length}.top1_pct", num(100 * statistics.mean(top1), 2))
    claim(f"LB5.{length}.median_kl", num(statistics.median(kl), 4))
    claim(f"LB5.{length}.p99_kl", num(s[math.ceil(.99 * len(s)) - 1], P99_ND[length]))
    claim(f"LB5.{length}.mean_kl", num(m, 4))
    claim(f"LB5.{length}.mean_kl.ci_lo", num(lo, 4))
    claim(f"LB5.{length}.mean_kl.ci_hi", num(hi, 4))
    claim(f"LB5.{length}.dnll", sgn(d, 4))
    claim(f"LB5.{length}.dnll.ci_lo", sgn(dlo, 4))
    claim(f"LB5.{length}.dnll.ci_hi", sgn(dhi, 4))
    claim(f"LB5.{length}.median_dnll", sgn(statistics.median(r["dnll"] for r in items), 4))
    docs = []
    for r in items:
        x = per_doc[(length, r["id"])]
        if abs(x["kl_mean"] - r["e3"]["kl_mean"]) > 1e-12 or abs(x["dnll"] - r["dnll"]) > 1e-12:
            raise SystemExit(f"PER-DOC-KL.json {length} {r['id']}: kl_mean / dnll differ from the KL record")
        docs.append(x)
    for key, keep in (("lt0.2", lambda c: c < 0.2), ("0.2to0.5", lambda c: 0.2 <= c < 0.5),
                      ("ge0.5", lambda c: c >= 0.5)):
        g = [x for x in docs if keep(x["copyable"])]
        say(f"{length} copyable {key}: {len(g)} documents, mean KL {statistics.mean(x['kl_mean'] for x in g):.4f}, "
            f"mean dNLL {statistics.mean(x['dnll'] for x in g):+.4f}")
        claim(f"S1.{length}.{key}.docs", len(g))
        claim(f"S1.{length}.{key}.mean_kl", num(statistics.mean(x["kl_mean"] for x in g), 4))
        claim(f"S1.{length}.{key}.mean_dnll", sgn(statistics.mean(x["dnll"] for x in g), 4))
