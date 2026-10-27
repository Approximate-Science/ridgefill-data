"""S8: depth dependence on Llama-3.1-8B. Novel-token NLL against exact (document-weighted mean of arm - exact) for
RidgeFill and the identity fill at the four split depths (S 8 / 12 / 16 / 20 = 1/4, 3/8, 1/2, 5/8), and on the 30 fresh
documents RidgeFill - identity at 1/2 and 5/8 (BCa within domain; finals/tables.py split curve and fresh panel)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, parse, say, sgn       # noqa: E402
from reftree import Tree                                   # noqa: E402

args = parse(__doc__)
model = "llama-3.1-8b-instruct"
t = Tree(args.root, [model + "-nll", model + "-nll-fresh"])
for frac, layer in (("1/4", 8), ("3/8", 12), ("1/2", 16), ("5/8", 20)):
    for key, arm in (("ridgefill", "kva"), ("identity", "noproj")):
        v = t.mean("battery", model, "nll_novel", arm, frac, vs_exact=True)
        say(f"{arm}@{frac} (S{layer}) - exact, novel tokens: {v:+.4f}")
        claim(f"S8.llama.{key}.s{layer}", sgn(v, 3))
scope = "dataset:" + t.cells[model + "-nll-fresh"]["dataset"]
for frac, layer in (("1/2", 16), ("5/8", 20)):
    c = t.contrast(scope, model, "nll_novel", ("kva", frac), ("noproj", frac))
    say(f"fresh: RidgeFill - identity at {frac}: {c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n={c['n']}")
    claims_ci(f"S8.fresh.s{layer}", c["est"], c["lo"], c["hi"], 3)
