"""S7: the cost at the primary split (3/8): RidgeFill - identity on novel-token and key-token NLL per model, paired by
document, BCa 5,000 within domain; key tokens on the documents with key tokens (finals/tables.py battery panel B)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claims_ci, parse, say                   # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
t = Tree(args.root, [m + "-nll" for m in MODELS.values()])
for short, model in MODELS.items():
    for kind in ("novel", "key"):
        c = t.contrast("battery", model, "nll_" + kind, ("kva", "3/8"), ("noproj", "3/8"))
        say(f"{model} {kind}: RidgeFill - identity {c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n={c['n']}")
        claims_ci(f"S7.{short}.{kind}", c["est"], c["lo"], c["hi"], 3)
