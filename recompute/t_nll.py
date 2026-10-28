"""T: novel-token loss in paper sections 6.1, 6.2, 6.4, 6.5 and 6.6 (Tables 17, 22, 23).

Read the reference samples, average paired arm-minus-exact losses equally over documents,
and reproduce the finals/tables.py SplitFiller, battery contrasts and mechanism formulas.
Intervals use reftree's paired BCa bootstrap: 5,000 resamples within domain, seed 20261004,
95%, unit weighting. Selected rows and tail scales route to their causal reruns, including
those runs' own exact and RidgeFill arms. Loss removed is a ratio of paired column means,
100 * (1 - (splice - exact) / (RidgeFill - exact)), never a mean of document ratios.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say, sgn  # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
P = "3/8"
t = Tree(args.root, [m + suffix for m in MODELS.values() for suffix in ("-nll", "-nll-causal")])
DEPTHS = (("1/4", 8), ("3/8", 12), ("1/2", 16), ("5/8", 20))


def describe(*parts):
    # Keep stdout claim-only so the documented direct invocation works with gate.py.
    with redirect_stdout(sys.stderr):
        say(*parts)


for short, model in MODELS.items():
    value = t.mean("battery", model, "nll_novel", "pop-native", "pop", vs_exact=True)
    claim(f"T.{short}.nll.pop_native", sgn(value, 3))
    describe(model, "POP native - exact, novel NLL:", value)
    for frac, layer32 in DEPTHS:
        layer = {"1/4": 9, "3/8": 13, "1/2": 18, "5/8": 22}[frac] if short == "qwen3" else layer32
        arms = (("nofill", "skip"),) if short == "llama" else (
            ("ridgefill", "kva"), ("identity", "noproj"), ("nofill", "skip"))
        for key, arm in arms:
            value = t.mean("battery", model, "nll_novel", arm, frac, vs_exact=True)
            claim(f"T.{short}.nll.s{layer}.{key}", sgn(value, 3))
            describe(model, f"{arm}@{frac} - exact, novel NLL:", value)
        if frac != P:
            c = t.contrast("battery", model, "nll_novel", ("kva", frac), ("noproj", frac))
            key = f"T.{short}.nll.s{layer}.ridgefill_vs_identity"
            claim(key + ".ci_lo", sgn(c["lo"], 3))
            claim(key + ".ci_hi", sgn(c["hi"], 3))
            describe(model, frac, "RidgeFill - identity:", c["est"], c["lo"], c["hi"], "n=", c["n"])

    c = t.contrast("battery", model, "nll_novel", ("kva", P), ("skip+tailbudget", P))
    key = f"T.{short}.nll.primary.ridgefill_vs_tailbudget"
    if short == "llama":
        claims_ci(key, c["est"], c["lo"], c["hi"], 3)
    else:
        claim(key + ".delta", sgn(c["est"], 3))
    describe(model, "RidgeFill - no fill with longer tail:", c["est"], c["lo"], c["hi"], "n=", c["n"])

    c = t.contrast("battery", model, "nll_novel", ("kva", P), ("pop-native", "pop"))
    key = f"T.{short}.nll.primary.ridgefill_vs_pop"
    if short == "qwen3":
        claims_ci(key, c["est"], c["lo"], c["hi"], 3)
    else:
        claim(key + ".ci_lo", sgn(c["lo"], 3))
        claim(key + ".ci_hi", sgn(c["hi"], 3))
    describe(model, "RidgeFill - POP native:", c["est"], c["lo"], c["hi"], "n=", c["n"])

model = MODELS["llama"]
value = t.mean("battery", model, "nll_novel", "kva+qattn10", P, vs_exact=True)
claim("T.llama.nll.primary.rows", sgn(value, 3))
describe(model, "causal rows - own exact, novel NLL:", value)

# The paper follows the document worst under plain RidgeFill, rather than selecting a
# new worst document under the rows arm. Choose it from the original run, then read
# both scores of its rows contrast inside the causal run.
_, units, base, exact = t.paired("battery", model, "nll_novel", ("kva", P), ("exact", "-"))
worst = max(zip(units, base - exact), key=lambda pair: pair[1])[0]
row_values = t.values("battery", model, "nll_novel_causal", "kva+qattn10", P)
row_exact = t.values("battery", model, "nll_novel_causal", "exact", "-")
claim("T.llama.nll.primary.rows_on_worst_document", sgn(row_values[worst] - row_exact[worst], 3))
describe("Plain RidgeFill's worst document:", worst, "; causal rows - exact:", row_values[worst] - row_exact[worst])

SPLICES = (("keys", "kva+kexact"), ("values", "kva+vexact"), ("both", "kvexact"),
           ("key_scale", "kva+ktail"), ("value_scale", "kva+vtail"))
for short, model in MODELS.items():
    for key, arm in SPLICES:
        if short == "qwen3" and key == "value_scale":
            continue  # This run was not part of the design and is not an assigned claim.
        value = t.mean("battery", model, "nll_novel", arm, P, vs_exact=True)
        claim(f"T.{short}.nll.primary.{key}", sgn(value, 3))
        describe(model, arm, "- own exact, novel NLL:", value)


def removed(means):
    return 100.0 * (1.0 - (means[..., 0] - means[..., 2]) / (means[..., 1] - means[..., 2]))


for short, key, arm in (("qwen35", "key_scale", "kva+ktail"),
                        ("qwen3", "values", "kva+vexact"), ("qwen3", "key_scale", "kva+ktail")):
    n, est, lo, hi = t.ratio("battery", MODELS[short], "nll_novel",
                            [(arm, P), ("kva", P), ("exact", "-")], removed)
    if lo is None or hi is None:
        raise SystemExit(f"{short} {arm}: assigned loss-removed interval is undefined")
    key = f"T.{short}.nll.primary.{key}.loss_removed"
    claim(key, num(est, 1) + " %")
    claim(key + ".ci_lo", num(lo, 1))
    claim(key + ".ci_hi", num(hi, 1))
    describe(MODELS[short], arm, "loss removed (%):", est, lo, hi, "n=", n)
