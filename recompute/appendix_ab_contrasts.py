"""Appendix A.3 question-position/control diagnostics and Appendix B additional contrasts.

Reproduce code/finals/tables.py Stats.contrast: intersect paired sample IDs, sort
units, use sample/document weighting, and resample within RULER/turn-2 task or NLL
domain. bca.py supplies the finals' BCa 95% intervals (5,000 draws, seed 20261004,
half-weight ties, jackknife acceleration, linear quantiles). Question-first versus
question-last pairs only rows filled in both layouts and uses question-first task
labels, as Stats.contrast(name_b='ruler_filled') does. The Llama identity-rows
non-inferiority bound is the one-sided 95% BCa lower end with the binomial-floor
guard. Count unequal raw control scores; test all paired turn-2 deltas by their
maximum absolute value. Read only the raw reference JSON under --root.
All human-readable diagnostics go to stderr; stdout contains only claim lines.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say, sgn   # noqa: E402
from reftree import MODELS, Tree                            # noqa: E402
import bca                                                 # noqa: E402

P = "3/8"
ENDS = ("ci_lo", "ci_hi")
ALL = ("delta", "ci_lo", "ci_hi")
# model, id stem, metric, first arm/split, second arm/split, subset,
# decimal places, printed components. No estimate is a typed input.
CONTRASTS = [
    ("llama", "ridgefill_vs_nofill", "ruler_filled", ("kva", P), ("skip", P), None, 1, ENDS),
    ("llama", "ridgefill_vs_nofill", "nll_novel", ("kva", P), ("skip", P), None, 3, ALL),
    ("llama", "ridgefill_vs_pop", "nll_novel", ("kva", P), ("pop-native", "pop"), None, 3, ENDS),
    ("llama", "ridgefill_vs_identity_half", "ruler_filled", ("kva", "1/2"), ("noproj", "1/2"), None, 1, ALL),
    ("llama", "ridgefill_vs_identity_64k", "ruler_filled", ("kva", P), ("noproj", P), "64k", 1, ALL),
    ("llama", "ridgefill_rows_vs_exact_64k", "ruler_filled", ("kva+qattn10", P), ("exact", "-"), "64k", 1, ALL),
    ("llama", "identity_qblind_vs_identity_rows", "ruler_filled", ("noproj+qblind10", P), ("noproj+qattn10", P), None, 1, ALL),
    ("qwen35", "ridgefill_vs_nofill", "nll_novel", ("kva", P), ("skip", P), None, 3, ENDS),
    ("qwen35", "ridgefill_vs_pop", "nll_novel", ("kva", P), ("pop-native", "pop"), None, 3, ENDS),
    ("qwen35", "identity_rows_vs_exact", "ruler_filled", ("noproj+qattn10", P), ("exact", "-"), None, 1, ENDS),
    ("qwen35", "identity_qblind_vs_identity_rows", "ruler_filled", ("noproj+qblind10", P), ("noproj+qattn10", P), None, 1, ENDS),
    ("qwen35", "ridgefill_rows_vs_identity_rows", "turn2", ("kva+qattn10", P), ("noproj+qattn10", P), None, 1, ENDS),
    ("qwen3", "ridgefill_vs_nofill", "nll_novel", ("kva", P), ("skip", P), None, 3, ALL),
    ("qwen3", "identity_rows_vs_exact", "ruler_filled", ("noproj+qattn10", P), ("exact", "-"), None, 1, ENDS),
    ("qwen3", "ridgefill_rows_vs_identity_rows", "turn2", ("kva+qattn10", P), ("noproj+qattn10", P), None, 1, ALL),
]

args = parse(__doc__)
t = Tree(args.root, [m + suffix for m in MODELS.values()
                     for suffix in ("-ruler", "-nll", "-secondq")])

for short, key, metric, a, b, subset, nd, components in CONTRASTS:
    where = (lambda r: r["length"] == 65536) if subset == "64k" else None
    c = t.contrast("battery", MODELS[short], metric, a, b, where=where)
    cid = f"AB.{short}.{metric}.{key}"
    with redirect_stdout(sys.stderr):
        say(f"{cid}: n={c['n']}, delta={c['est']:.9f}, CI=[{c['lo']:.9f}, {c['hi']:.9f}]")
    if components == ALL:
        claims_ci(cid, c["est"], c["lo"], c["hi"], nd)
    else:
        for component, field in (("ci_lo", "lo"), ("ci_hi", "hi")):
            claim(cid + "." + component, sgn(c[field], nd))

n, est, lo = t.noninferiority_lower("battery", MODELS["llama"], "ruler_filled",
                                    ("noproj+qattn10", P), ("exact", "-"))
with redirect_stdout(sys.stderr):
    say(f"Llama identity rows - exact: n={n}, delta={est:.9f}, one-sided lower={lo:.9f}")
claim("AB.llama.ruler_filled.identity_rows_vs_exact.ni_lower", sgn(lo, 2))

_, units, xa, xb = t.paired("battery", MODELS["llama"], "turn2",
                            ("kva+qattn10", P), ("noproj+qattn10", P))
with redirect_stdout(sys.stderr):
    say(f"Llama turn 2: n={len(units)}, nonzero paired deltas={np.count_nonzero(xa != xb)}")
claim("AB.llama.turn2.ridgefill_rows_vs_identity_rows.max_abs_delta", num(float(np.max(np.abs(xa - xb))), 0))

model = MODELS["qwen3"]
first = t.values("battery", model, "qfirst_filled", "noproj+qattn10", P)
last = t.values("battery", model, "ruler_filled", "noproj+qattn10", P)
units = sorted(set(first) & set(last))
if not units:
    raise SystemExit("no paired filled rows under both question positions")
recs = t.records("battery", model, "ruler-qfirst")
deltas = np.array([first[u] - last[u] for u in units])
c = bca.bca(deltas, strata=[recs[u]["task"] for u in units], weighting="unit")
with redirect_stdout(sys.stderr):
    say(f"Qwen3 question-first identity rows - question-last: n={len(units)}, "
        f"delta={c.estimate:.9f}, CI=[{c.lo:.9f}, {c.hi:.9f}]")
claims_ci("AB.qwen3.ruler_filled.identity_rows_qfirst_vs_qlast", c.estimate, c.lo, c.hi, 1)

for metric, population in (("ruler_filled", "filled"), ("ruler_all", "all_needle")):
    _, units, random, tail = t.paired("battery", model, metric,
                                    ("kva+random", P), ("kva+tailmatch", P),
                                    where=lambda r: 8192 <= r["length"] <= 32768)
    count = int(np.count_nonzero(random != tail))
    with redirect_stdout(sys.stderr):
        say(f"Qwen3 random rows versus longer tail ({population}): {count} unequal of {len(units)}")
    claim(f"AB.qwen3.ruler.{population}.random_vs_longer_tail.unequal", count)
