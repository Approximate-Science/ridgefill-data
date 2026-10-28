"""T: filled-region retrieval in paper sections 6.2, 6.4-6.7 (Tables 22 and 23).

Read reference RULER samples only, retaining needle_filled rows and each arm's actual
population. Control, splice and half-depth means use 8K-32K; rows-minus-identity-rows
uses all available lengths (64K included on Llama and Qwen3.5). Match finals/tables.py:
unit-weighted scores in points, paired BCa with 5,000 resamples within task, seed
20261004, 95%; gap closed is 100*(splice-RidgeFill)/(exact-RidgeFill), evaluated on
paired column means. Count unequal per-row scores, rather than rounded means.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say       # noqa: E402
from reftree import MODELS, SUBSETS, Tree                   # noqa: E402

args = parse(__doc__)
P = "3/8"
t = Tree(args.root, [m + "-ruler" for m in MODELS.values()])
le32k = SUBSETS["le32k"]


def describe(*parts):
    with redirect_stdout(sys.stderr):
        say(*parts)


MEANS = (("qwen35", "tailbudget", "skip+tailbudget", P),
         ("qwen3", "tailbudget", "skip+tailbudget", P),
         ("qwen3", "pop_native", "pop-native", "pop"),
         ("qwen3", "rows", "kva+qattn10", P),
         ("qwen3", "random", "kva+random", P),
         ("qwen3", "longer_tail", "kva+tailmatch", P),
         ("llama", "keys", "kva+kexact", P),
         ("llama", "values", "kva+vexact", P),
         ("qwen3", "keys", "kva+kexact", P),
         ("qwen3", "values", "kva+vexact", P))
for short, key, arm, split in MEANS:
    model = MODELS[short]
    value = t.mean("battery", model, "ruler_filled", arm, split, where=le32k)
    n = len(t.values("battery", model, "ruler_filled", arm, split, where=le32k))
    claim(f"T.{short}.ruler.le32k.{key}", num(value, 1))
    describe(model, arm, "8K-32K filled-region RULER:", value, "n=", n)

for short, model in MODELS.items():
    for key, other, where in (("rows_vs_longer_tail", "kva+tailmatch", le32k),
                               ("rows_vs_identity_rows", "noproj+qattn10", None)):
        c = t.contrast("battery", model, "ruler_filled", ("kva+qattn10", P), (other, P), where=where)
        claims_ci(f"T.{short}.ruler.primary.{key}", c["est"], c["lo"], c["hi"], 1)
        describe(model, "selected rows -", other, ":", c["est"], c["lo"], c["hi"], "n=", c["n"])

model = MODELS["qwen3"]
_, units, random, tail = t.paired("battery", model, "ruler_filled",
                                ("kva+random", P), ("kva+tailmatch", P), where=le32k)
claim("T.qwen3.ruler.le32k.random_vs_tail_discordant", int((random != tail).sum()))
describe(model, "random rows vs longer tail: unequal scores", int((random != tail).sum()), "of", len(units))


def closed(means):
    return 100.0 * (means[..., 0] - means[..., 1]) / (means[..., 2] - means[..., 1])


for key, arm in (("keys", "kva+kexact"), ("values", "kva+vexact")):
    n, est, lo, hi = t.ratio("battery", model, "ruler_filled",
                            [(arm, P), ("kva", P), ("exact", "-")], closed, where=le32k)
    claim(f"T.qwen3.ruler.le32k.{key}.gap_closed", num(est, 1) + " %")
    describe(model, arm, "retrieval gap closed (%):", est, "; BCa endpoints:", lo, hi, "n=", n)

# Restrict to the mechanism table's population: the records carrying the key splice.
splice_rows = lambda r: le32k(r) and P in r["arms"].get("kva+kexact", {})  # noqa: E731
_, units, exact, ridgefill = t.paired("battery", model, "ruler_filled",
                                   ("exact", "-"), ("kva", P), where=splice_rows)
claim("T.qwen3.ruler.le32k.exact_vs_ridgefill_discordant", int((exact != ridgefill).sum()))
describe(model, "exact vs RidgeFill: unequal scores", int((exact != ridgefill).sum()), "of", len(units))

model = MODELS["llama"]
c = t.contrast("battery", model, "ruler_filled", ("kva", "1/2"), ("noproj", "1/2"), where=le32k)
claims_ci("T.llama.ruler.s16.ridgefill_vs_identity", c["est"], c["lo"], c["hi"], 1)
describe(model, "half-depth RidgeFill - identity:", c["est"], c["lo"], c["hi"], "n=", c["n"])
for key, arm in (("ridgefill", "kva"), ("identity", "noproj")):
    value = t.mean("battery", model, "ruler_filled", arm, "1/2", where=le32k)
    claim(f"T.llama.ruler.s16.{key}", num(value, 1))
    describe(model, "half-depth", arm, "filled-region RULER:", value)
