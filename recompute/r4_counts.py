"""R4: Qwen3-8B's exact prefix rows against RidgeFill alone, per sample, on the filled-region RULER rows at the
primary split (8K-32K, n = 138): how many samples the attention-selected rows (kva+qattn10) and the random rows
(kva+random) make worse or better than RidgeFill (kva), with the losses by task and by length."""
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, parse, say                       # noqa: E402
from reftree import Tree                                   # noqa: E402

args = parse(__doc__)
model = "qwen3-8b"
t = Tree(args.root, [model + "-ruler"])
recs = t.records("battery", model, "ruler")
base = t.values("battery", model, "ruler_filled", "kva", "3/8")
for key, arm in (("rows", "kva+qattn10"), ("random", "kva+random")):
    v = t.values("battery", model, "ruler_filled", arm, "3/8")
    units = sorted(set(v) & set(base))
    worse = [u for u in units if v[u] < base[u]]
    better = [u for u in units if v[u] > base[u]]
    by_task = Counter(recs[u]["task"] for u in worse)
    by_len = Counter(recs[u]["length"] for u in worse)
    say(f"{arm} vs kva on {len(units)} samples: worse {len(worse)}, better {len(better)}; losses by task "
        f"{dict(sorted(by_task.items()))}, by length {dict(sorted(by_len.items()))}")
    claim(f"R4.{key}.n", len(units))
    claim(f"R4.{key}.worse", len(worse))
    claim(f"R4.{key}.better", len(better))
    if key == "rows":
        claim("R4.rows.worse_single_needle", by_task.get("niah_single_1", 0))
