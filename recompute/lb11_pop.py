"""LB11: RidgeFill at 5/8 depth against POP as published (identity fill at POP's own split, one-token tail).
(1) The ruler-deep cells (8K-32K filled-region rows): arm means, RidgeFill at 5/8 - POP native with its one-sided
95 % non-inferiority bound at the registered -3 point margin (the more conservative of the BCa bound and the exact
binomial floor), and RidgeFill - identity at 5/8 (BCa). (2) Llama's 30 fresh documents: RidgeFill at 5/8 - POP
native, novel-token NLL (BCa within domain). finals/tables.py deep_panel and fresh_panel."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say, sgn  # noqa: E402
from reftree import Tree                                   # noqa: E402

args = parse(__doc__)
DEEP = {"llama": "llama-3.1-8b-instruct", "qwen3": "qwen3-8b"}
t = Tree(args.root, [m + "-ruler-deep" for m in DEEP.values()] + ["llama-3.1-8b-instruct-nll-fresh"])
for short, model in DEEP.items():
    for key, arm, split in (("exact", "exact", "-"), ("pop", "pop-native", "pop"), ("identity", "noproj", "5/8"),
                            ("ridgefill", "kva", "5/8")):
        v = t.values("battery", model, "deep_filled", arm, split)
        claim(f"LB11.{short}.{key}", num(sum(v.values()) / len(v), 1))
        say(f"{model} ruler-deep {arm}@{split}: {sum(v.values()) / len(v):.2f} (n={len(v)})")
    n, est, lo = t.noninferiority_lower("battery", model, "deep_filled", ("kva", "5/8"), ("pop-native", "pop"))
    say(f"{model}: RidgeFill@5/8 - POP native {est:+.2f}, one-sided lower bound {lo:+.3f} (margin -3), n={n}")
    claim(f"LB11.{short}.n", n)
    claim(f"LB11.{short}.vs_pop.delta", sgn(est, 1))
    claim(f"LB11.{short}.vs_pop.ni_lower", sgn(lo, 2))
    c = t.contrast("battery", model, "deep_filled", ("kva", "5/8"), ("noproj", "5/8"))
    say(f"{model}: RidgeFill - identity at 5/8 {c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]")
    claims_ci(f"LB11.{short}.vs_identity", c["est"], c["lo"], c["hi"], 1)
model = "llama-3.1-8b-instruct"
scope = "dataset:" + t.cells[model + "-nll-fresh"]["dataset"]
c = t.contrast(scope, model, "nll_novel", ("kva", "5/8"), ("pop-native", "pop"))
say(f"fresh documents: RidgeFill@5/8 - POP native {c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n={c['n']}")
claim("LB11.fresh.n", c["n"])
claims_ci("LB11.fresh.vs_pop", c["est"], c["lo"], c["hi"], 3)
