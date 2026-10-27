"""LB7d: Qwen3.8-Flash-Next quality mode against plain RidgeFill, paired per document over the 90 perplexity documents
(VERIFY check 8): per-document NLL difference (quality - plain), averaged within domain and then across the six
domains, reported as exp(mean) - 1 with a BCa bootstrap, 5,000 resamples within domain (seed 20261004), on novel
tokens and on all tokens. Positive control first: the same code on arm - exact gives the published point estimates
+2.65 % (RidgeFill) and +1.60 % (quality mode); its lower bounds sit at zero and -0.8 % (fn_quality.py prints the
tables' intervals)."""
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, parse, say, sgn      # noqa: E402

args = parse(__doc__)
F = args.root / "results" / "flashnext" / "final-det-20261010" / "results"


def arm(folder):
    out = {}
    for f in sorted(need(F / folder / "ppl-a").glob("lp-*.json")):
        d = load(f)
        out[d["name"]] = d
    return out


E, P, Q = arm("exact"), arm("ridgefill"), arm("graded")
names = sorted(set(E) & set(P) & set(Q))
if len(names) != 90:
    raise SystemExit(f"{len(names)} documents in all three arms, expected 90")
say("documents", len(names))
dom = np.array([E[n]["task"] for n in names])


def stat(v, dom):
    return float(np.mean([np.mean(v[dom == d]) for d in sorted(set(dom))]))


def bca(v, dom, B=5000, seed=20261004):
    ds = sorted(set(dom))
    idx = {d: np.where(dom == d)[0] for d in ds}
    th = stat(v, dom)
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for b in range(B):
        sel = np.concatenate([rng.choice(idx[d], len(idx[d])) for d in ds])
        bs[b] = stat(v[sel], dom[sel])
    z0 = NormalDist().inv_cdf(min(max(np.mean(bs < th), 1e-9), 1 - 1e-9))
    jk = np.array([stat(np.delete(v, k), np.delete(dom, k)) for k in range(len(v))])
    jm = jk.mean()
    a = np.sum((jm - jk) ** 3) / (6 * np.sum((jm - jk) ** 2) ** 1.5)

    def q(al):
        z = NormalDist().inv_cdf(al)
        return NormalDist().cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(bs, [q(0.025), q(0.975)])
    return th, lo, hi


def pct(x):
    return 100 * (np.exp(x) - 1)


for key in ("nll_novel", "nll_all"):
    kind = "novel" if key == "nll_novel" else "all"
    for lab, A, B, ck in (("RidgeFill vs exact", P, E, "posctl.ridgefill_vs_exact"),
                          ("quality mode vs exact", Q, E, "posctl.quality_vs_exact"),
                          ("quality mode vs RidgeFill", Q, P, "quality_vs_ridgefill")):
        v = np.array([A[n][key] - B[n][key] for n in names])
        th, lo, hi = bca(v, dom)
        say(f"{kind:5s} {lab:26s} gap {pct(th):+.2f} % [{pct(lo):+.2f} %, {pct(hi):+.2f} %]")
        if ck.startswith("posctl"):
            if kind == "novel":
                claim(f"LB7d.{ck}.{kind}.delta", sgn(pct(th), 2))
            continue
        claim(f"LB7d.{ck}.{kind}.delta", sgn(pct(th), 2))
        claim(f"LB7d.{ck}.{kind}.ci_lo", sgn(pct(lo), 2))
        claim(f"LB7d.{ck}.{kind}.ci_hi", sgn(pct(hi), 2))
