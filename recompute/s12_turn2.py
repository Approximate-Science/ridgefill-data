"""S12: a second question on the once-filled cache (turn 2 of the secondq cells): turn-2 RULER accuracy of
RidgeFill + rows and of exact per model (sample-weighted means of the stored per-sample turn2 scores)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, num, parse, say                  # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
t = Tree(args.root, [m + "-secondq" for m in MODELS.values()])
for short, model in MODELS.items():
    for key, arm, split in (("rows", "kva+qattn10", "3/8"), ("exact", "exact", "-")):
        v = t.values("battery", model, "turn2", arm, split)
        mean = sum(v.values()) / len(v)
        say(f"{model} turn 2 {arm}@{split}: {mean:.2f} (n={len(v)})")
        claim(f"S12.{short}.{key}", num(mean, 1))
