"""R6: Llama-3.1-8B at the primary split 3/8 (S12): on how many documents RidgeFill's novel-token NLL is worse than no
fill's (kva vs skip, both at 3/8), by document length, and the worst document (largest RidgeFill - exact)."""
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, parse, say, sgn                  # noqa: E402
from reftree import Tree                                   # noqa: E402

args = parse(__doc__)
model = "llama-3.1-8b-instruct"
t = Tree(args.root, [model + "-nll"])
recs = t.records("battery", model, "longppl")
fill = t.values("battery", model, "nll_novel", "kva", "3/8")
skip = t.values("battery", model, "nll_novel", "skip", "3/8")
exact = t.values("battery", model, "nll_novel", "exact", "-")
units = sorted(set(fill) & set(skip) & set(exact))
count = defaultdict(lambda: [0, 0])
for u in units:
    band = "8k_32k" if recs[u]["length"] <= 32768 else "64k" if recs[u]["length"] <= 65536 else "128k"
    count[band][1] += 1
    count[band][0] += fill[u] > skip[u]
for band in ("8k_32k", "64k", "128k"):
    say(f"{band}: RidgeFill worse than no fill on {count[band][0]} of {count[band][1]} documents")
    claim(f"R6.worse.{band}", count[band][0])
    claim(f"R6.docs.{band}", count[band][1])
claim("R6.worse.ge64k", count["64k"][0] + count["128k"][0])
claim("R6.docs.ge64k", count["64k"][1] + count["128k"][1])
worst = max(units, key=lambda u: fill[u] - exact[u])
say(f"worst document {recs[worst]['doc']}: RidgeFill {fill[worst] - exact[worst]:+.4f}, no fill "
    f"{skip[worst] - exact[worst]:+.4f} (nats vs exact)")
claim("R6.worst.doc", recs[worst]["doc"])
claim("R6.worst.ridgefill", sgn(fill[worst] - exact[worst], 3))
claim("R6.worst.nofill", sgn(skip[worst] - exact[worst], 3))
