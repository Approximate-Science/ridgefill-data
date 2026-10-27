"""LB8: the fitted fill against the identity fill on filled-region needle retrieval at the primary split (3/8),
matched split and tail. RULER rows with every needle before the tail, pooled over tasks and lengths (Llama and
Qwen3.5 8K-64K, Qwen3-8B 8K-32K); paired BCa, 5,000 resamples within task (finals/tables.py battery panel B)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say       # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
t = Tree(args.root, [m + "-ruler" for m in MODELS.values()])
for short, model in MODELS.items():
    c = t.contrast("battery", model, "ruler_filled", ("kva", "3/8"), ("noproj", "3/8"))
    fill = t.mean("battery", model, "ruler_filled", "kva", "3/8")
    ident = t.mean("battery", model, "ruler_filled", "noproj", "3/8")
    say(f"{model}: RidgeFill {fill:.1f} vs identity {ident:.1f}; RidgeFill - identity {c['est']:+.2f} "
        f"[{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}")
    claim(f"LB8.{short}.n", c["n"])
    claims_ci(f"LB8.{short}", c["est"], c["lo"], c["hi"], 1)
    claim(f"LB8.{short}.ridgefill", num(fill, 1))
    claim(f"LB8.{short}.identity", num(ident, 1))
