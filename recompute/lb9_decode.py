"""LB9: decode tracks exact more closely with the fitted fill. 20 documents x 512 greedy steps per model at the
primary split: mean per-step KL(exact || arm) per arm, and the paired RidgeFill - identity greedy-agreement contrast
(points, BCa 5,000, prompt-weighted; finals/tables.py)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say       # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
t = Tree(args.root, [m + "-decode" for m in MODELS.values()])
for short, model in MODELS.items():
    kl_f = t.mean("battery", model, "kl", "kva", "3/8")
    kl_i = t.mean("battery", model, "kl", "noproj", "3/8")
    c = t.contrast("battery", model, "agree", ("kva", "3/8"), ("noproj", "3/8"))
    say(f"{model}: KL {kl_f:.4f} (RidgeFill) vs {kl_i:.4f} (identity); agreement RidgeFill - identity "
        f"{c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}")
    claim(f"LB9.{short}.n", c["n"])
    claim(f"LB9.{short}.kl.ridgefill", num(kl_f, 3))
    claim(f"LB9.{short}.kl.identity", num(kl_i, 3))
    claims_ci(f"LB9.{short}.agree", c["est"], c["lo"], c["hi"], 1)
