"""S9: attention-selected exact prefix rows at the primary split, filled-region RULER rows (finals/tables.py battery
panel B and C3): RidgeFill + rows - exact (Llama and Qwen3.5: estimate and the one-sided 95 % non-inferiority bound,
the more conservative of BCa and the binomial floor, margin -2; Qwen3-8B: BCa interval), rows - random rows, and
question-blind rows - rows (Llama and Qwen3.5). BCa 5,000 within task."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, parse, say, sgn       # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
t = Tree(args.root, [m + "-ruler" for m in MODELS.values()])
P = "3/8"
for short, model in MODELS.items():
    if short == "qwen3":
        c = t.contrast("battery", model, "ruler_filled", ("kva+qattn10", P), ("exact", "-"))
        say(f"{model}: rows - exact {c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}")
        claims_ci(f"S9.{short}.rows_vs_exact", c["est"], c["lo"], c["hi"], 1)
    else:
        n, est, lo = t.noninferiority_lower("battery", model, "ruler_filled", ("kva+qattn10", P), ("exact", "-"))
        say(f"{model}: rows - exact {est:+.2f}, one-sided lower bound {lo:+.3f} (margin -2), n={n}")
        claim(f"S9.{short}.rows_vs_exact.delta", sgn(est, 1))
        claim(f"S9.{short}.rows_vs_exact.ni_lower", sgn(lo, 2))
    c = t.contrast("battery", model, "ruler_filled", ("kva+qattn10", P), ("kva+random", P))
    say(f"{model}: rows - random rows {c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}")
    claims_ci(f"S9.{short}.rows_vs_random", c["est"], c["lo"], c["hi"], 1)
    if short != "qwen3":
        c = t.contrast("battery", model, "ruler_filled", ("kva+qblind10", P), ("kva+qattn10", P))
        say(f"{model}: question-blind rows - rows {c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}")
        claims_ci(f"S9.{short}.qblind_vs_rows", c["est"], c["lo"], c["hi"], 1)
