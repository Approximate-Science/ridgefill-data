"""S10: where the error lives, the splices at the primary split (finals/tables.py mechanism table). Loss removed:
100 (1 - mean(arm - exact) / mean(RidgeFill - exact)) over the NLL documents (novel tokens, BCa within domain; no
interval when the statistic is not finite on some resample). Gap closed: 100 (arm - RidgeFill) / (exact - RidgeFill)
on the filled-region RULER rows the splices ran on (8K-32K)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, num, parse, say                  # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
P = "3/8"
t = Tree(args.root, [m + s for m in MODELS.values() for s in ("-nll", "-nll-causal", "-ruler")])
LOSS = {"llama": ("kva+kexact", "kva+vexact", "kva+ktail"), "qwen35": ("kva+kexact", "kva+vexact", "kvexact"),
        "qwen3": ("kva+kexact", "kvexact")}
GAP = {"llama": ("kva+kexact",)}
KEY = {"kva+kexact": "keys", "kva+vexact": "values", "kvexact": "both", "kva+ktail": "key_scale"}


def removed(m):
    return 100.0 * (1.0 - (m[..., 0] - m[..., 2]) / (m[..., 1] - m[..., 2]))


def closed(m):
    return 100.0 * (m[..., 0] - m[..., 1]) / (m[..., 2] - m[..., 1])


for short, model in MODELS.items():
    for arm in LOSS[short]:
        n, est, lo, hi = t.ratio("battery", model, "nll_novel", [(arm, P), ("kva", P), ("exact", "-")], removed)
        say(f"{model} {arm}: loss removed {est:.2f} % [{lo}, {hi}] n={n}")
        claim(f"S10.{short}.{KEY[arm]}.loss_removed", num(est, 1))
        if lo is not None:
            claim(f"S10.{short}.{KEY[arm]}.loss_removed.ci_lo", num(lo, 1))
            claim(f"S10.{short}.{KEY[arm]}.loss_removed.ci_hi", num(hi, 1))
    for arm in GAP.get(short, ()):
        rows = lambda r, arm=arm: P in r["arms"].get(arm, {})                              # noqa: E731
        n, est, lo, hi = t.ratio("battery", model, "ruler_filled", [(arm, P), ("kva", P), ("exact", "-")], closed,
                                 where=rows)
        say(f"{model} {arm}: RULER gap closed {est:.2f} % [{lo}, {hi}] n={n}")
        claim(f"S10.{short}.{KEY[arm]}.gap_closed", num(est, 1))
