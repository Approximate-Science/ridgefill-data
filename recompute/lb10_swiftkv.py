"""LB10: SwiftKV against RidgeFill on Llama-3.1-8B at SwiftKV's split 16 (1/2 depth), novel-token NLL, paired by
document, BCa 5,000 within domain (finals/tables.py battery panel B): SwiftKV as released (its one-token tail) -
RidgeFill, and SwiftKV's maps with RidgeFill's 4,096-token exact tail - RidgeFill."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, parse, say            # noqa: E402
from reftree import Tree                                   # noqa: E402

args = parse(__doc__)
model = "llama-3.1-8b-instruct"
t = Tree(args.root, [model + "-nll"])
for key, arm in (("native", "swiftkv-native"), ("maps_tail", "swiftkv")):
    c = t.contrast("battery", model, "nll_novel", (arm, "1/2"), ("kva", "1/2"))
    say(f"{arm} - RidgeFill at 1/2, novel tokens: {c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n={c['n']}")
    claim(f"LB10.{key}.n", c["n"])
    claims_ci(f"LB10.{key}", c["est"], c["lo"], c["hi"], 3)
