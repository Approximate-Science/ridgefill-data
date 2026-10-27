"""S11: LongBench on Llama-3.1-8B, 8 tasks x 50 items at the primary split: task-weighted arm means (the mean of the
task means, LongBench's average) and RidgeFill - identity, paired by item, BCa 5,000 within task (finals/tables.py).
Uses the stored per-item scores; the gold answers are not in the repository and are not needed."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say       # noqa: E402
from reftree import Tree                                   # noqa: E402

args = parse(__doc__)
model = "llama-3.1-8b-instruct"
t = Tree(args.root, [model + "-longbench"])
for key, arm, split in (("exact", "exact", "-"), ("nofill", "skip", "3/8"), ("identity", "noproj", "3/8"),
                        ("pop", "pop-native", "pop"), ("ridgefill", "kva", "3/8"),
                        ("identity_rows", "noproj+qattn10", "3/8"), ("ridgefill_rows", "kva+qattn10", "3/8")):
    v = t.mean("battery", model, "longbench", arm, split)
    say(f"{arm}@{split}: {v:.2f}")
    claim(f"S11.{key}", num(v, 1))
c = t.contrast("battery", model, "longbench", ("kva", "3/8"), ("noproj", "3/8"))
say(f"RidgeFill - identity {c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}")
claim("S11.n", c["n"])
claims_ci("S11.ridgefill_vs_identity", c["est"], c["lo"], c["hi"], 1)
