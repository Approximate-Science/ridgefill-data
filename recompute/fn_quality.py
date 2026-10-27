"""LB7c, S5 (quality): Qwen3.8-Flash-Next against exact's own server start. Perplexity gap on novel tokens,
100 (exp(mean paired NLL difference) - 1), per document over the 90 locked documents <= 32K (unit = the source file,
novel-token NLL recomputed from the stored log-probs and novel mask), document-weighted, BCa 5,000 within domain (the
finals' declared statistic, as finals/tables.py's Flash-Next table computes it). RULER: the four needle tasks at
8K-32K plus niah_single_1 at 64K (325 rows) and the filled-region rows (every needle ends before the 2,048-token
tail), mean score per arm, and the filled-region rows where RidgeFill scores below exact."""
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bca as S                                            # noqa: E402
from common import claim, load, need, num, parse, say, sgn # noqa: E402

ARMS = {"ridgefill": ("ridgefill", "LB7c"), "quality": ("graded", "S5")}
TAIL = 2048
args = parse(__doc__)
F = args.root / "results" / "flashnext" / "final-det-20261010" / "results"


def ppl(folder):
    out = {}
    for f in sorted(need(F / folder / "ppl-a").glob("lp-*.json")):
        r = load(f)
        lp, mask = r["logprobs"], r["novel"]
        xs = [-x for x, m in zip(lp, mask) if m]
        v = sum(xs) / len(xs)
        if abs(v - r["nll_novel"]) > 1e-9:
            raise SystemExit(f"{f}: novel-token NLL {v} from the log-probs, the record stores {r['nll_novel']}")
        out[r["source"]] = (r["task"], v)
    if len(out) != 90:
        raise SystemExit(f"{folder}: {len(out)} perplexity documents, expected 90")
    return out


def ruler(folder):
    out = {}
    for f in sorted(need(F / folder / "ruler-a").glob("ruler-*.json")):
        if f.name.startswith("_"):
            continue
        r = load(f)
        out[f"{r['source']}#{r['row']:02d}"] = r
    if len(out) != 325:
        raise SystemExit(f"{folder}: {len(out)} RULER rows, expected 325")
    return out


exact = ppl("exact")
units = sorted(exact)
for arm, (folder, tag) in ARMS.items():
    a = ppl(folder)
    if sorted(a) != units:
        raise SystemExit(f"{folder}: its perplexity documents are not exact's")
    d = np.array([a[u][1] - exact[u][1] for u in units])
    r = S.bca(d, strata=[exact[u][0] for u in units])
    gap = [100 * math.expm1(x) for x in (r.estimate, r.lo, r.hi)]
    say(f"{arm}: perplexity gap {gap[0]:+.3f} % [{gap[1]:+.3f}, {gap[2]:+.3f}] over {len(units)} documents")
    claim(f"{tag}.{arm}.ppl_gap_pct.delta", sgn(gap[0], 2))
    claim(f"{tag}.{arm}.ppl_gap_pct.ci_lo", sgn(gap[1], 2))
    claim(f"{tag}.{arm}.ppl_gap_pct.ci_hi", sgn(gap[2], 2))
ex = ruler("exact")
filled = {u for u, r in ex.items() if all(e <= r["n"] - TAIL for _, e in r["needles"])}
say(f"RULER rows {len(ex)}, filled-region {len(filled)}")
claim("LB7c.ruler.n", len(ex))
claim("LB7c.ruler.n_filled", len(filled))
for arm, folder, tag in (("exact", "exact", "LB7c"), ("ridgefill", "ridgefill", "LB7c"), ("quality", "graded", "S5")):
    rr = ex if arm == "exact" else ruler(folder)
    if sorted(rr) != sorted(ex):
        raise SystemExit(f"{folder}: its RULER rows are not exact's")
    all_ = 100 * sum(r["score"] for r in rr.values()) / len(rr)
    fil = 100 * sum(rr[u]["score"] for u in filled) / len(filled)
    say(f"{arm}: RULER all {all_:.2f}, filled region {fil:.2f}")
    claim(f"{tag}.{arm}.ruler_all", num(all_, 1))
    claim(f"{tag}.{arm}.ruler_filled", num(fil, 1))
    if arm == "ridgefill":
        below = [u for u in sorted(filled) if rr[u]["score"] < ex[u]["score"]]
        say(f"ridgefill below exact on filled rows: {below}")
        claim("LB7c.ridgefill.ruler_filled_below_exact", len(below))
