"""R1, R2: filled-region RULER at 8K-32K by arm (Llama-3.1-8B n = 136, Qwen3.5-9B n = 139): the four needle tasks,
samples whose every needle lies before the 4,096-token tail, arm scores at the primary split 3/8 ("-" exact, "pop"
POP native), and Llama's RidgeFill - identity, paired BCa 5,000 within task (the finals' declared statistic,
bca.py). The 2026-10-26 scratch script (ruler1t.py) read the records in directory-listing order and drew its
resamples with Generator.choice, so its printed interval depended on the file system's listing order; this script
sorts the units, as the finals tables do."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, num, parse, say       # noqa: E402
from reftree import Tree                                   # noqa: E402

TASKS4 = ("niah_single_1", "niah_multikey_1", "niah_multivalue", "niah_multiquery")
SEL = lambda r: r["task"] in TASKS4 and r["length"] <= 32768                          # noqa: E731
ARMS = [("exact", "exact", "-"), ("identity", "noproj", "3/8"), ("ridgefill", "kva", "3/8"),
        ("ridgefill_rows", "kva+qattn10", "3/8"), ("identity_rows", "noproj+qattn10", "3/8"),
        ("pop", "pop-native", "pop")]
RUNS = [("R1", "llama-3.1-8b-instruct",
         ARMS + [("nofill", "skip", "3/8"), ("nofill_longer_tail", "skip+tailbudget", "3/8")], True),
        ("R2", "qwen3.5-9b", ARMS, False)]

args = parse(__doc__)
t = Tree(args.root, [m + "-ruler" for _, m, _, _ in RUNS])
for prefix, model, arms, contrast in RUNS:
    n = None
    for key, arm, split in arms:
        v = t.values("battery", model, "ruler_filled", arm, split, where=SEL)
        n = n or len(v)
        if len(v) != n:
            raise SystemExit(f"{model} {arm}@{split}: {len(v)} filled-region rows at 8K-32K, the others {n}")
        say(f"{model} {arm}@{split}: {sum(v.values()) / len(v):.2f} (n={len(v)})")
        claim(f"{prefix}.{key}", num(sum(v.values()) / len(v), 1))
    claim(f"{prefix}.n", n)
    if contrast:
        c = t.contrast("battery", model, "ruler_filled", ("kva", "3/8"), ("noproj", "3/8"), where=SEL)
        say(f"{model}: RidgeFill - identity {c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}")
        claims_ci(f"{prefix}.ridgefill_vs_identity", c["est"], c["lo"], c["hi"], 1)
