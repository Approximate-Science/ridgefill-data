"""R3: SwiftKV against RidgeFill on key tokens, Llama-3.1-8B at split 16 (depth 1/2). Key tokens: the documents with
key tokens under exact (80 of 126), document-weighted; paired BCa 5,000 within domain (the finals' declared
statistic, bca.py). Arms and split are arguments (q8doc.py of 2026-10-26 generalised). Positive control first: the
same code on novel tokens must give the published LB10 contrasts, +0.010 [-0.004, +0.025] (SwiftKV as released -
RidgeFill) and -0.024 [-0.037, -0.015] (SwiftKV maps + exact tail - RidgeFill); those claim lines are R3.posctl.*.
The 2026-10-26 intervals came from a q8doc.py variant (Generator.choice, seed 20261004) that read the records in
directory-listing order, so they depended on the file system; this script sorts the units, as the finals tables do."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, parse, say, sgn       # noqa: E402
from reftree import Tree                                   # noqa: E402

LABEL = {"swiftkv-native": "native", "swiftkv": "maps_tail", "kva": "ridgefill", "noproj": "identity"}


def options(p):
    p.add_argument("--model", default="llama-3.1-8b-instruct")
    p.add_argument("--split", default="1/2", help="split key of the filled arms (1/2 = S16 on Llama)")
    p.add_argument("--base", default="kva", help="the arm the others are contrasted with")
    p.add_argument("--arms", default="swiftkv-native,swiftkv")


args = parse(__doc__, options)
t = Tree(args.root, [args.model + "-nll"])
base = (args.base, args.split)
for kind in ("novel", "key"):
    tag = "posctl" if kind == "novel" else "key"
    for arm in [args.base] + args.arms.split(","):
        v = t.mean("battery", args.model, "nll_" + kind, arm, args.split, vs_exact=True)
        say(f"{kind:5s} {arm}@{args.split} - exact {v:+.4f}")
        claim(f"R3.{LABEL.get(arm, arm)}.{kind}_vs_exact", sgn(v, 3))
    for arm in args.arms.split(","):
        c = t.contrast("battery", args.model, "nll_" + kind, (arm, args.split), base)
        say(f"{kind:5s} {arm} - {args.base} {c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n={c['n']}"
            + ("  (positive control: must equal LB10)" if kind == "novel" else ""))
        key = f"R3.{tag}.{LABEL.get(arm, arm)}_vs_{LABEL.get(args.base, args.base)}"
        claim(key + ".n", c["n"])
        claims_ci(key, c["est"], c["lo"], c["hi"], 3)
